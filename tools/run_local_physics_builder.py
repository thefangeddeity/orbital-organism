#!/usr/bin/env python3
"""
Run ON 7elwe. The continuous counterpart to organism.py's own live
self-evolution loop, but for local-physics-builder (modules/
local_physics_builder.py) -- the third branch: a SECOND, independent
NeuralLearner trained on real near-surface ballistics (cannon's engine)
instead of the universe-scale Kepler ephemeris organism's own brain
learns.

Dispatches to Tanzania exclusively (TRAINING_HOST below) -- this is
the heavier of the two per-body workloads (real training + self-
evolution, not just procedural noise generation), which is exactly why
body-builder's texture generation moved to Tina instead: Tanzania stays
dedicated to this.

Each cycle trains a genuinely fresh network from a new random seed
(tools/remote_runner.py's train_local_physics task -- re-seeded per
call, not resumed) and keeps it only if its final validation loss beats
the best ever seen for that body. Earth only, for now -- cannon's
engine hardcodes Earth's gravity model.

Usage:
    python3 run_local_physics_builder.py [--interval SECONDS] [--cycles N]

    --interval: seconds to sleep between cycles (default 120 -- each
                real training run is itself ~10-15s of dispatch time,
                so this doesn't need as tight a cadence as body-
                builder's texture generation).
    --cycles:   stop after N cycles instead of running forever
                (omit to run until interrupted -- Ctrl+C).
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
import time
from pathlib import Path

ORGANISM_DIR = Path(__file__).resolve().parent.parent
MODULES_DIR = ORGANISM_DIR / "modules"
TOOLS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(MODULES_DIR))
sys.path.insert(0, str(TOOLS_DIR))

import fleet_dispatch as fd
import local_physics_builder as lpb

TRAINING_HOST = "tanzania"


def dispatch_training(body_name: str, seed: int):
    payload = {"task": "train_local_physics", "body": body_name, "seed": seed}
    try:
        return fd.dispatch(TRAINING_HOST, "train_local_physics", payload)
    except Exception as exc:
        print(f"  [{body_name}] dispatch to {TRAINING_HOST} failed: {exc}", file=sys.stderr)
        return None


def run(interval_s: float, max_cycles: int | None) -> None:
    rng = random.Random()
    state = lpb.load_state()

    bodies = list(lpb.SUPPORTED_BODIES)
    print(f"local-physics-builder: {len(bodies)} supported body(ies): {', '.join(bodies)}")
    print(f"host: {TRAINING_HOST}")
    print(f"interval: {interval_s:.0f}s between cycles"
          + (f", stopping after {max_cycles} cycles" if max_cycles else ", running until interrupted"))
    print()

    body_cycle = itertools.cycle(bodies)
    cycle_count = 0

    while max_cycles is None or cycle_count < max_cycles:
        body_name = next(body_cycle)
        seed = lpb.propose_seed(rng)

        started = time.perf_counter()
        outcome = lpb.evaluate_attempt(state, body_name, seed, dispatch_training)
        elapsed = time.perf_counter() - started

        lpb.save_state(state)

        verdict = "ACCEPTED" if outcome["accepted"] else "rejected"
        if outcome.get("reason"):
            detail = f" -- {outcome['reason']}"
        else:
            detail = f" loss={outcome['final_loss']:.6f} (best={outcome['baseline']:.6f})"
        print(
            f"[{time.strftime('%H:%M:%S')}] {body_name:10s} gen={outcome['generation']:4d} "
            f"seed={outcome['seed']:>10d} {verdict:8s} ({elapsed:.1f}s){detail}"
        )

        cycle_count += 1

        if max_cycles is None or cycle_count < max_cycles:
            time.sleep(interval_s)

    print()
    print(f"Stopped after {cycle_count} cycles.")
    for name in bodies:
        body_state = state["bodies"].get(name, {})
        print(
            f"  {name:10s} accepted={body_state.get('accepted', 0)}/"
            f"{body_state.get('generation', 0)}  "
            f"best_final_loss={body_state.get('best_final_loss')}  "
            f"best_seed={body_state.get('best_seed')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=120.0)
    parser.add_argument("--cycles", type=int, default=None)
    args = parser.parse_args()

    run(args.interval, args.cycles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
