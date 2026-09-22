#!/usr/bin/env python3
"""
Proposes a scratch block-tree loss core, offline, and caches it for
the live organism's next propose_scratch_core cycle to pick up --
same architecture as dispatch_tanzania.py and fetch_exoplanet_data.py:
never called from inside organism.py's render loop, output lands in
state/ for a later cycle to read.

scratch_blocks.py's random_candidate_tree() already generates a
structurally-valid random tree inline, live, in-process -- that isn't
being replaced. What this adds is a SECOND, offline proposer that can
afford to think harder about which tree to propose, because it isn't
paying rent in the per-frame budget.

Closes the loop between proposing and the real outcome: neural_learner
.py's self_evolve() logs every scratch-tree attempt's real accept/
reject verdict to state/scratch_history.json (see _log_scratch_history
()). This tool reads that same file and folds the last few entries
into the Gemini prompt as a digest, so each call knows what's already
been tried and whether it helped, instead of proposing blind every
time. This is NOT gradient-based learning -- nothing here updates
Gemini's weights, and it's still a single stateless request each run --
it's in-context learning: recent real results steer the next proposal
the same way a person skimming a lab notebook would.

Two backends, same contract -- (tree_dict, source_label):

  propose_via_gemini(api_key)     -- real API call (generateContent,
                                      responseMimeType=application/json
                                      so the model returns the tree
                                      directly, no markdown-fence
                                      stripping needed). Any failure --
                                      network, auth, unparsable
                                      response -- returns None rather
                                      than raising, so main() falls
                                      back to the local emulation
                                      exactly as if no key were
                                      configured.

  propose_via_local_emulation()   -- works today, no dependencies.
                                      Samples several candidates from
                                      the same random_candidate_tree()
                                      generator and keeps the most
                                      promising one by a cheap local
                                      proxy, rather than trusting a
                                      single draw. Whatever backend
                                      eventually runs here, the tree it
                                      hands back is NEVER trusted on
                                      its say-so -- main() re-validates
                                      and re-gradient-checks it before
                                      caching, the same defense-in-
                                      depth attach_simulator() already
                                      applies when loading a persisted
                                      scratch_tree from disk.

Usage:
    python3 propose_scratch_tree.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ORGANISM_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = ORGANISM_DIR / "state"
MODULES_DIR = ORGANISM_DIR / "modules"
ENV_PATH = ORGANISM_DIR / "proposer.env"
PROPOSAL_PATH = STATE_DIR / "scratch_proposal.json"
HISTORY_PATH = STATE_DIR / "scratch_history.json"
HISTORY_DIGEST_ENTRIES = 8

sys.path.insert(0, str(MODULES_DIR))

import scratch_blocks

SAMPLES_PER_RUN = 12
PROBE = np.linspace(-2.0, 2.0, 9)

GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
GEMINI_TIMEOUT = 30.0

UNARY_OPS = ("neg", "abs", "square", "sqrt", "exp", "log1p", "sign", "tanh", "sigmoid")
BINARY_OPS = ("add", "sub", "mul", "div", "min", "max", "pow")

def _load_scratch_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return history if isinstance(history, list) else []


def _format_history_digest(
    history: list[dict], limit: int = HISTORY_DIGEST_ENTRIES
) -> str:
    """
    Turns the last few real evaluate_candidate_real() outcomes into a
    short digest for the prompt. This is what actually closes the loop:
    without it, every Gemini call is stateless, with no way to know
    what it already proposed or whether any of it helped train the
    network better than the current config.
    """
    if not history:
        return ""

    lines = []
    for entry in history[-limit:]:
        verdict = "ACCEPTED" if entry.get("accepted") else "rejected"
        score = entry.get("score")
        score_str = f"{score:.4g}" if isinstance(score, (int, float)) else "n/a"
        source = entry.get("source", "?")
        tree_json = json.dumps(entry.get("tree"))
        lines.append(f"- [{source}] {verdict} (score {score_str}): {tree_json}")

    return (
        "\n\nRecent proposals and their REAL outcomes (lower score is "
        "better; ACCEPTED means it beat the current network's score "
        "after both sides got equal additional training -- don't "
        "repeat a rejected shape verbatim, and consider building on "
        "what got accepted):\n" + "\n".join(lines) + "\n"
    )


def _build_prompt(history_digest: str) -> str:
    return f"""You are proposing a new loss "core penalty" function C(e) for a \
small neural network that predicts orbital positions, where e = \
prediction - target (a residual). This is exploratory: propose \
something DIFFERENT from plain squared error or absolute error, that \
might train better on some residual distributions than either does.

Return ONLY a JSON object describing an expression tree in this exact \
grammar, nothing else:
  - {{"op": "<name>", "args": [<node>, <node>, ...]}} for an operator
  - {{"var": "e"}} for the residual itself (the only variable available)
  - {{"const": <number>}} for a literal constant

Unary ops (exactly 1 arg): {", ".join(UNARY_OPS)}
Binary ops (exactly 2 args): {", ".join(BINARY_OPS)}

Constraints: at most 24 nodes total, at most 6 levels deep. Use ONLY \
the op names listed above -- no other function names are valid. \
Prefer something that behaves sensibly (finite, roughly increasing in \
|e|) across e in roughly [-3, 3], since it will be numerically \
gradient-checked before ever being used.

Example of valid output shape (do not just copy this -- propose \
something different):
{{"op": "mul", "args": [{{"const": 0.5}}, {{"op": "square", "args": [{{"var": "e"}}]}}]}}
{history_digest}
Output only the JSON object, no markdown fences, no explanation."""


def _load_gemini_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()

    if not ENV_PATH.exists():
        return None

    try:
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == "GEMINI_API_KEY":
                value = value.strip().strip('"').strip("'")
                return value or None
    except OSError:
        return None

    return None


def propose_via_gemini(api_key: str, history_digest: str = "") -> tuple[dict, str] | None:
    """
    Real call to the Gemini API. Any failure -- network, auth, an
    unparsable response -- returns None rather than raising, so
    main() falls back to propose_via_local_emulation() exactly as if
    no key were configured. Whatever comes back is treated as
    untrusted external input: main() re-validates and re-gradient-
    checks it before ever caching it, same as every other external
    input this project accepts (Tanzania's sweep results, the NASA
    Exoplanet Archive rows).
    """
    body = json.dumps({
        "contents": [{"parts": [{"text": _build_prompt(history_digest)}]}],
        "generationConfig": {
            "temperature": 0.9,
            "responseMimeType": "application/json",
        },
    }).encode("utf-8")

    request = urllib.request.Request(
        GEMINI_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=GEMINI_TIMEOUT) as response:
            raw = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        print(f"Gemini request failed: HTTP {exc.code} -- {detail}", file=sys.stderr)
        return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Gemini request failed: {exc}", file=sys.stderr)
        return None

    try:
        text = raw["candidates"][0]["content"]["parts"][0]["text"]
        tree = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        print(f"Gemini response unparsable: {exc} -- raw: {raw}", file=sys.stderr)
        return None

    if not isinstance(tree, dict):
        print(f"Gemini response wasn't a JSON object: {tree!r}", file=sys.stderr)
        return None

    return tree, "gemini"


def _count_nodes(tree: dict) -> int:
    if "op" in tree:
        return 1 + sum(_count_nodes(a) for a in tree["args"])
    return 1


def _distinct_ops(tree: dict) -> set[str]:
    if "op" in tree:
        ops = {tree["op"]}
        for a in tree["args"]:
            ops |= _distinct_ops(a)
        return ops
    return set()


def propose_via_local_emulation(samples: int = SAMPLES_PER_RUN) -> tuple[dict, str]:
    """
    Draws several candidates from random_candidate_tree() and keeps the
    most promising gradient-clean one, instead of trusting a single
    draw the way the live in-process proposer has to (it can't afford
    to sample 12 trees and gradient-check each one every generation).

    The proxy score is NOT the real evaluate_candidate_real() -- that
    needs a live NeuralLearner and real training steps, out of scope
    for an offline tool with no network samples to train on. It's a
    cheap structural preference: more distinct ops used (richer
    composition) minus a small penalty per node (Occam's-razor bias
    against needless depth), and a hard floor against single-node
    trees -- the exact degenerate case ({'var': 'e'}, a no-op identity
    "loss") that surfaced this session's evaluate_candidate_real bugs
    by getting accepted on sampling noise alone.
    """
    rng = np.random.default_rng()

    candidates = []
    for _ in range(samples):
        tree = scratch_blocks.random_candidate_tree(rng, max_ops=3)

        try:
            problems = scratch_blocks.check_gradients(tree, {"e": PROBE})
        except Exception:
            continue

        if problems:
            continue

        node_count = _count_nodes(tree)
        if node_count <= 1:
            continue

        score = len(_distinct_ops(tree)) - 0.1 * node_count
        candidates.append((score, tree))

    if not candidates:
        # Every sample degenerated or failed its gradient check --
        # fall back to a single bare draw, same behavior as before
        # this tool existed.
        return scratch_blocks.random_candidate_tree(rng, max_ops=3), "local_emulation_fallback"

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1], "local_emulation"


def main() -> int:
    api_key = _load_gemini_key()
    history = _load_scratch_history()
    history_digest = _format_history_digest(history)

    result = propose_via_gemini(api_key, history_digest) if api_key else None

    if result is None:
        tree, source = propose_via_local_emulation()
    else:
        tree, source = result

    # Defense in depth: never trust a proposer's own say-so, whichever
    # backend produced it. Re-run the same checks main-line
    # _propose_scratch_core() would run on this tree in-process.
    try:
        scratch_blocks.validate(tree)
        problems = scratch_blocks.check_gradients(tree, {"e": PROBE})
    except scratch_blocks.InvalidBlockTree as exc:
        print(f"Proposed tree failed validation: {exc}")
        return 1

    if problems:
        print("Proposed tree failed its own gradient re-check:")
        for problem in problems:
            print(" ", problem)
        return 1

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "tree": tree,
        "source": source,
        "proposed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "consumed": False,
    }

    temp = PROPOSAL_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(PROPOSAL_PATH)

    print(f"Proposed via {source}: {json.dumps(tree)}")
    print(f"Cached to {PROPOSAL_PATH}")
    print("Picked up on the organism's next propose_scratch_core cycle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
