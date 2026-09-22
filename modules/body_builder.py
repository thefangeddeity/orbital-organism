from __future__ import annotations

"""
Body-builder: the per-body structural peer to NeuralLearner's own self-
evolution. Where NeuralLearner grows PREDICTION accuracy by proposing,
evaluating, and accepting candidate loss functions and hyperparameters
(modules/neural_learner.py's evaluate_candidate_real()/self_evolve()),
this grows REPRESENTATION fidelity the same way -- propose a candidate
texture-generation parameter set for one body, dispatch the real
compute to Tanzania, evaluate it against a real yardstick, accept only
if it's genuinely better, and persist the outcome either way.

Honest scope for this first pass: the move vocabulary is limited to
what tools/dispatch_tanzania.py's generate_world_textures task can
actually produce today -- resolution, octave count, seed. Rotation,
rings, moons, and sourcing real public elevation data are natural next
moves; none of them are implemented here, and this module does not
pretend otherwise.

The fitness signal is deliberately NOT a visual-quality judgment --
there is no honest way for this system to know whether a texture
"looks better". It is real detail (resolution x octaves, a genuinely
monotonic measure of how much information was actually computed) that
still fits inside a real, measured memory budget. That mirrors
NeuralLearner's own reference-metric discipline: judged against one
fixed, stable yardstick, not a moving or invented one.

Only bodies real_systems.registers_as_body() accepts (Ceres-or-larger)
are eligible -- the same size floor the rest of the organism's world-
tracking uses, not a second, disconnected threshold.
"""

import json
import math
import random
from pathlib import Path
from typing import Any

STATE_DIR = Path(__file__).resolve().parents[1] / "state"
STATE_PATH = STATE_DIR / "body_builder.json"
HISTORY_PATH = STATE_DIR / "body_builder_history.json"
WORLD_TEXTURES_PATH = STATE_DIR / "world_textures.json"
ORGANISM_CONFIG_PATH = STATE_DIR.parent / "organism.json"

MAX_HISTORY = 200

# Real, measured memory cost of a candidate's heightmap, in KB
# (resolution^2 float64 values). Distinct from OrganicBudget's own
# render-cost model (a different concern -- how expensive a body is to
# DRAW each frame at a given fidelity level) -- this is about how much
# raw generated data is reasonable to dispatch, cache, and ship back
# over SSH at all.
MAX_HEIGHTMAP_KB = 500.0

MIN_RESOLUTION = 32
MAX_RESOLUTION = 256
MIN_OCTAVES = 2
MAX_OCTAVES = 8

DEFAULT_RESOLUTION = 64
DEFAULT_OCTAVES = 4


def _heightmap_kb(resolution: int) -> float:
    return (resolution * resolution * 8) / 1024.0


def _detail_score(resolution: int, octaves: int) -> float:
    """
    A genuinely monotonic proxy for how much real detail was computed --
    NOT a claim about how good it looks. Higher resolution and more
    octaves are strictly more computation and strictly more information
    in the returned array; this is measurable, unlike "quality".
    """
    return float(resolution) * float(octaves)


def _stable_seed(name: str) -> int:
    import hashlib

    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


BODY_MOVES = {
    "increase_resolution": lambda p: {**p, "resolution": min(MAX_RESOLUTION, int(p["resolution"] * 1.5))},
    "decrease_resolution": lambda p: {**p, "resolution": max(MIN_RESOLUTION, int(p["resolution"] / 1.5))},
    "increase_octaves": lambda p: {**p, "octaves": min(MAX_OCTAVES, p["octaves"] + 1)},
    "decrease_octaves": lambda p: {**p, "octaves": max(MIN_OCTAVES, p["octaves"] - 1)},
    "reroll_seed": lambda p: {**p, "seed": random.randint(0, 2**32 - 1)},
}


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


def load_state() -> dict:
    return _read_json(STATE_PATH, {"bodies": {}})


def save_state(state: dict) -> None:
    _write_json(STATE_PATH, state)


def _log_history(entry: dict) -> None:
    history = _read_json(HISTORY_PATH, [])
    if not isinstance(history, list):
        history = []
    history.append(entry)
    history = history[-MAX_HISTORY:]
    _write_json(HISTORY_PATH, history)


def _persist_accepted_heightmap(body_name: str, params: dict, heightmap) -> None:
    """
    An accepted candidate's real heightmap goes straight into
    state/world_textures.json -- the SAME file organism.py's renderer
    already reads for L2/L3 fidelity (see organism.py's
    _load_world_textures()). This is the actual point of body-builder,
    not a side effect: its real, continuously-improving output becomes
    the organism's own texture fidelity directly, rather than sitting
    in a separate file nothing else consumes. A rejected candidate's
    heightmap is never written here -- only genuinely-accepted detail
    should ever reach the renderer.
    """
    config = _read_json(ORGANISM_CONFIG_PATH, {})
    world_mode = config.get("world", {}).get("mode", "solar_system")

    data = _read_json(WORLD_TEXTURES_PATH, {})
    if not isinstance(data, dict) or data.get("world_mode") != world_mode:
        # A cache for a different world (or none yet) -- start fresh
        # rather than mixing textures from two different worlds under
        # one world_mode label.
        data = {"world_mode": world_mode, "textures": {}}

    # Per-body resolution, not a single top-level one -- organism.py's
    # _load_world_textures() only ever reads each body's own
    # tex["heightmap"] (via np.asarray, shape inferred directly), never
    # a top-level "resolution"/"octaves" field, and different bodies
    # legitimately evolve to different resolutions independently here.
    data.setdefault("textures", {})[body_name] = {
        "resolution": params["resolution"],
        "heightmap": heightmap,
    }

    _write_json(WORLD_TEXTURES_PATH, data)


def current_params(state: dict, body_name: str) -> dict:
    body_state = state.get("bodies", {}).get(body_name)
    if body_state is not None:
        return dict(body_state["current_params"])
    return {
        "resolution": DEFAULT_RESOLUTION,
        "octaves": DEFAULT_OCTAVES,
        "seed": _stable_seed(body_name),
    }


def propose_candidate(params: dict, rng: random.Random) -> tuple[str, dict]:
    move_name = rng.choice(list(BODY_MOVES))
    candidate = BODY_MOVES[move_name](params)
    return move_name, candidate


def evaluate_candidate(
    state: dict, body_name: str, move_name: str, candidate_params: dict,
    generate_fn,
) -> dict:
    """
    generate_fn(body_name, candidate_params) -> heightmap (2-D
    sequence, or None on failure) is injected rather than imported
    directly, so this module never has to know HOW the real generation
    happens (SSH dispatch, local test stub, whatever) -- same
    separation-of-concerns reasoning as NeuralLearner accepting an
    already-built candidate list rather than constructing one itself.
    """
    body_state = state.setdefault("bodies", {}).get(body_name)
    if body_state is None:
        body_state = {
            "generation": 0,
            "accepted": 0,
            "current_params": current_params(state, body_name),
            "best_detail_score": 0.0,
        }
        state["bodies"][body_name] = body_state

    generation = body_state["generation"] + 1
    body_state["generation"] = generation

    projected_kb = _heightmap_kb(candidate_params["resolution"])

    if projected_kb > MAX_HEIGHTMAP_KB:
        outcome = {
            "body": body_name, "generation": generation, "move": move_name,
            "accepted": False, "reason": f"projected {projected_kb:.0f}KB over {MAX_HEIGHTMAP_KB:.0f}KB cap",
            "candidate_params": candidate_params,
        }
        _log_history(outcome)
        return outcome

    heightmap = generate_fn(body_name, candidate_params)

    if heightmap is None:
        outcome = {
            "body": body_name, "generation": generation, "move": move_name,
            "accepted": False, "reason": "generation failed",
            "candidate_params": candidate_params,
        }
        _log_history(outcome)
        return outcome

    candidate_score = _detail_score(candidate_params["resolution"], candidate_params["octaves"])
    baseline_score = body_state["best_detail_score"]

    accepted = candidate_score > baseline_score

    if accepted:
        body_state["current_params"] = candidate_params
        body_state["best_detail_score"] = candidate_score
        body_state["accepted"] += 1
        _persist_accepted_heightmap(body_name, candidate_params, heightmap)

    outcome = {
        "body": body_name, "generation": generation, "move": move_name,
        "accepted": accepted,
        "candidate_score": candidate_score, "baseline_score": baseline_score,
        "candidate_params": candidate_params,
        "projected_kb": projected_kb,
    }
    _log_history(outcome)
    return outcome
