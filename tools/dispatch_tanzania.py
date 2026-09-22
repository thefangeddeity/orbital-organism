#!/usr/bin/env python3
"""
Run ON 7elwe (or wherever this repo lives). Ships the current network
state + a batch of training samples to Tanzania over SSH/scp, runs
remote_runner.py there against real numpy on real hardware, and pulls
the result back.

This is deliberately NOT part of organism.py's real-time loop -- SSH
round-trips are seconds, not milliseconds, and the render loop has a
frame budget it can't blow. Run this manually, or schedule it (cron /
Task Scheduler) for real unattended operation; either way its output
lands in state/tanzania_results/ for a human or the next organism boot
to consult, never injected into a live running process mid-flight.

Usage:
    python3 dispatch_tanzania.py explore_mutation_space
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ORGANISM_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = ORGANISM_DIR / "state"
MODULES_DIR = ORGANISM_DIR / "modules"
TOOLS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(MODULES_DIR))

REMOTE_HOST = "tanzania"
REMOTE_DIR = "orbital_dispatch"

# Below this, a "better" combination is noise, not a real finding --
# without a floor, an infinitesimal improvement would get written into
# self_program.json as an accepted mutation on every single dispatch.
MIN_IMPROVEMENT_TO_ADOPT = 0.02


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, **kwargs)


def check_reachable() -> bool:
    result = _run([
        "ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes",
        REMOTE_HOST, "echo ok",
    ])
    return result.returncode == 0 and "ok" in result.stdout


def generate_samples(count_days: float = 3 * 365.25) -> list[dict]:
    """
    Independent, authoritative training data -- computed here from the
    real Kepler engine (orbital-sandbox.py), not copied from a live
    process's memory (which isn't persisted anywhere). Any valid set of
    solar-system position samples is equally legitimate ground truth
    for this prediction task.
    """
    import importlib.util

    sim_path = ORGANISM_DIR.parents[1] / "orbital-sandbox.py"
    spec = importlib.util.spec_from_file_location("sim", sim_path)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)

    import numpy as np

    samples = []
    prev = {}
    times = np.linspace(0, count_days * 86400, 400)

    for t in times:
        current = {}
        for body in sim.PLANETS:
            orbit = sim.planet_orbit(body)
            pos, _vel = orbit.state_at_time(t)
            x_au = pos[0] / sim.AU_KM
            y_au = pos[1] / sim.AU_KM
            features = [x_au, y_au, float(body.ecc), float(body.a_au) / 2.0]
            target = [x_au, y_au]
            if body.name in prev:
                samples.append({"x": prev[body.name], "y": target})
            current[body.name] = features
        prev = current

    return samples


def _load_live_learner():
    """
    Constructs the actual live NeuralLearner via attach_simulator(True),
    which loads weights AND replays self_program.json -- including any
    active scratch tree -- straight from disk, the identical code path
    every real learner process uses to figure out its own current state.

    Shared by build_payload() (to label the sweep's baseline row with
    the REAL current core/weighting/activation, not a hardcoded guess)
    and local_reference_score() (the zero-training fallback score), so
    there's exactly one place that determines "what is the organism
    currently running" -- not two that could quietly drift apart.
    """
    from neural_learner import NeuralLearner

    learner = NeuralLearner()
    learner.attach_simulator(True)
    return learner


def build_payload(task: str) -> dict:
    state_path = STATE_DIR / "neural_learner.json"
    state = json.loads(state_path.read_text(encoding="utf-8-sig"))

    learner = _load_live_learner()

    return {
        "task": task,
        "parameters": state["parameters"],
        "network": state["network"],
        # The learner's ACTUAL current variant, not a hardcoded guess --
        # this is what lets adopt_if_better() find the matching, fairly-
        # trained row in the sweep instead of comparing against an
        # untrained score. See adopt_if_better()'s docstring.
        "baseline_loss_variant": learner.get_loss_function().name,
        "baseline_activation_variant": learner.active_activation_variant,
        "baseline_is_scratch": learner.active_scratch_tree is not None,
        "samples": generate_samples(),
    }


def local_reference_score(payload: dict) -> float:
    """
    Zero-training score of the current live network on
    loss_blocks.REFERENCE_METRIC.

    This alone is NOT a fair baseline for adoption anymore: every row in
    the remote sweep (including whichever one matches the live config)
    receives TRAIN_STEPS_PER_COMBO of real training first, so comparing
    a trained sweep row against this untrained number has exactly the
    "training itself looks like the improvement" confound that was
    found and fixed in evaluate_candidate_real() locally (noop won
    20/20 against an untrained baseline there). adopt_if_better() now
    prefers the sweep's own matching row -- trained the same amount --
    as the real baseline, and falls back to this number only when the
    live config is a scratch tree with no named row in the sweep to
    match against.
    """
    learner = _load_live_learner()

    learner.observation_samples = [
        (np.asarray(s["x"], dtype=float), np.asarray(s["y"], dtype=float))
        for s in payload["samples"]
    ]

    return learner._reference_score()


def adopt_if_better(result: dict, baseline: float, payload: dict) -> dict:
    """
    If the sweep's best combination clears the CURRENT config by a real
    margin -- both sides trained the same amount -- write it into
    self_program.json as an accepted mutation with provenance, so the
    next boot's _replay_accepted_variants() picks it up (never injected
    into a currently-running process; see the module docstring).

    Prefers, as the baseline, the sweep row matching the live learner's
    own (loss_variant, activation_variant) -- it went through the exact
    same TRAIN_STEPS_PER_COMBO training burst from the exact same
    starting weights as every other row, so it's a fair "current config,
    trained" comparison point rather than "current config, untrained".
    Falls back to the untrained `baseline` score only when the live
    config is a scratch tree (payload["baseline_is_scratch"]) or, should
    it happen, no matching row turns up in the sweep -- both flagged in
    the returned report so a caller can see which comparison was used.

    Returns a small report dict describing what happened, for main()
    to print.
    """
    best = result.get("best")
    if not best:
        return {"adopted": False, "reason": "no sweep results"}

    reference_after = best.get("reference_score")
    if reference_after is None:
        return {"adopted": False, "reason": "no comparable baseline"}

    matched_baseline = None
    if not payload.get("baseline_is_scratch"):
        for entry in result.get("results", []):
            if (
                entry["loss_variant"] == payload.get("baseline_loss_variant")
                and entry["activation_variant"]
                == payload.get("baseline_activation_variant")
            ):
                matched_baseline = entry["reference_score"]
                break

    if matched_baseline is not None:
        true_baseline = matched_baseline
        baseline_kind = "trained-matched"
    else:
        # Scratch-tree baseline (no named row can match it), or,
        # unexpectedly, no matching row found -- fall back to the
        # untrained score. Known to slightly overstate any adopted
        # improvement, since the sweep side got a training burst this
        # side didn't; MIN_IMPROVEMENT_TO_ADOPT still gates it so a
        # marginal case won't adopt purely on that gap.
        true_baseline = baseline
        baseline_kind = "untrained-fallback"

    if true_baseline <= 0:
        return {"adopted": False, "reason": "no comparable baseline"}

    improvement = (true_baseline - reference_after) / true_baseline

    if improvement < MIN_IMPROVEMENT_TO_ADOPT:
        return {
            "adopted": False,
            "reason": f"improvement {improvement:.1%} below "
                      f"{MIN_IMPROVEMENT_TO_ADOPT:.0%} floor "
                      f"(baseline: {baseline_kind})",
        }

    core, _, weighting = best["loss_variant"].partition("/")
    commands = [
        f"switch_core_{core}",
        f"switch_weighting_{weighting}",
        f"switch_activation_{best['activation_variant']}",
    ]

    program_path = STATE_DIR / "self_program.json"
    try:
        data = json.loads(program_path.read_text(encoding="utf-8-sig"))
    except Exception:
        data = {"generation": 0, "accepted": 0}

    generation = int(data.get("generation", 0)) + 1
    accepted = int(data.get("accepted", 0)) + 1

    payload_out = {
        "commands": commands,
        "generation": generation,
        "accepted": accepted,
        "last_score": float(reference_after),
        "success_rate": accepted / max(generation, 1),
        "generations_since_accepted": 0,
        "source": "tanzania_sweep",
        "baseline_kind": baseline_kind,
    }

    temp = program_path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")
    temp.replace(program_path)

    return {
        "adopted": True,
        "commands": commands,
        "improvement": improvement,
        "reference_before": true_baseline,
        "reference_after": reference_after,
        "baseline_kind": baseline_kind,
    }


def deploy_code() -> None:
    _run(["ssh", REMOTE_HOST, f"mkdir -p {REMOTE_DIR}"])

    for name in ("neural_learner.py", "lego.py", "loss_blocks.py"):
        result = _run([
            "scp", str(MODULES_DIR / name),
            f"{REMOTE_HOST}:{REMOTE_DIR}/{name}",
        ])
        if result.returncode != 0:
            raise RuntimeError(f"scp {name} failed: {result.stderr}")

    result = _run([
        "scp", str(TOOLS_DIR / "remote_runner.py"),
        f"{REMOTE_HOST}:{REMOTE_DIR}/remote_runner.py",
    ])
    if result.returncode != 0:
        raise RuntimeError(f"scp remote_runner.py failed: {result.stderr}")


def dispatch(task: str, payload: dict | None = None) -> dict:
    if not check_reachable():
        raise RuntimeError(f"{REMOTE_HOST} is not reachable over SSH right now")

    deploy_code()

    # Accepts an already-built payload so the caller can compute
    # local_reference_score() without rebuilding it a second time.
    # generate_samples() is deterministic (fixed time grid, no
    # randomness) so calling build_payload() twice would actually
    # produce identical samples either way -- this is purely to avoid
    # redoing the simulation work, not a correctness requirement.
    if payload is None:
        payload = build_payload(task)

    local_payload_path = STATE_DIR / "_tanzania_payload.json"
    local_payload_path.write_text(json.dumps(payload), encoding="utf-8")

    scp_result = _run([
        "scp", str(local_payload_path),
        f"{REMOTE_HOST}:{REMOTE_DIR}/payload.json",
    ])
    if scp_result.returncode != 0:
        raise RuntimeError(f"scp payload failed: {scp_result.stderr}")

    run_result = subprocess.run(
        [
            "ssh", REMOTE_HOST,
            f"cd {REMOTE_DIR} && python3 remote_runner.py payload.json",
        ],
        capture_output=True, text=True, timeout=120,
    )

    if run_result.returncode != 0:
        raise RuntimeError(f"remote run failed: {run_result.stderr}")

    return json.loads(run_result.stdout)


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "explore_mutation_space"

    print(f"Dispatching '{task}' to {REMOTE_HOST}...")
    started = time.perf_counter()

    payload = build_payload(task)
    baseline = local_reference_score(payload)

    result = dispatch(task, payload=payload)

    elapsed = time.perf_counter() - started
    print(f"Done in {elapsed:.1f}s.")

    results_dir = STATE_DIR / "tanzania_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = results_dir / f"{task}-{timestamp}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Result written to {out_path}")
    print()

    if result.get("best"):
        best = result["best"]
        print(f"Best combination: loss={best['loss_variant']} "
              f"activation={best['activation_variant']} "
              f"-> reference_score={best['reference_score']:.6f} "
              f"(current network, untrained: {baseline:.6f})")
        print(f"sample count: {result['sample_count']}, "
              f"combinations tried: {result.get('combinations_tried', len(result['results']))}")

        adoption = adopt_if_better(result, baseline, payload)
        print()

        if adoption["adopted"]:
            print(
                f"ADOPTED -> {adoption['commands']} "
                f"({adoption['improvement']:.1%} improvement over "
                f"{adoption['baseline_kind']} baseline "
                f"{adoption['reference_before']:.6f}). "
                f"Takes effect on the organism's next boot."
            )
        else:
            print(f"Not adopted: {adoption['reason']}")


if __name__ == "__main__":
    main()
