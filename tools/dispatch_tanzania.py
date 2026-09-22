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

ORGANISM_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = ORGANISM_DIR / "state"
MODULES_DIR = ORGANISM_DIR / "modules"
TOOLS_DIR = Path(__file__).resolve().parent

REMOTE_HOST = "tanzania"
REMOTE_DIR = "orbital_dispatch"


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


def build_payload(task: str) -> dict:
    state_path = STATE_DIR / "neural_learner.json"
    state = json.loads(state_path.read_text(encoding="utf-8-sig"))

    return {
        "task": task,
        "parameters": state["parameters"],
        "network": state["network"],
        "baseline_loss_variant": "mse",
        "baseline_activation_variant": "relu",
        "samples": generate_samples(),
    }


def deploy_code() -> None:
    _run(["ssh", REMOTE_HOST, f"mkdir -p {REMOTE_DIR}"])

    for name in ("neural_learner.py", "lego.py"):
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


def dispatch(task: str) -> dict:
    if not check_reachable():
        raise RuntimeError(f"{REMOTE_HOST} is not reachable over SSH right now")

    deploy_code()

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

    result = dispatch(task)

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
              f"-> validation_loss={best['validation_loss']:.6f}")
        print(f"(sample count: {result['sample_count']}, "
              f"baseline was {result['baseline']['loss_variant']}/"
              f"{result['baseline']['activation_variant']})")


if __name__ == "__main__":
    main()
