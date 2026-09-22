#!/usr/bin/env python3
"""
Run ON 7elwe. The continuous counterpart to organism.py's own live
self-evolution loop, but for body-builder (modules/body_builder.py):
round-robins through every Ceres-or-larger body in the currently
configured world, proposing one candidate texture-generation move per
cycle, dispatching the real generation to Tanzania, and persisting the
outcome -- meant to be started once and left running for hours
unattended, same "let it run all day" shape as organism.py itself.

Deliberately a SEPARATE, standalone process, not wired into organism
.py's own render loop -- this dispatches to Tanzania every cycle (a
real SSH round-trip, seconds not milliseconds) and organism.py's frame
budget has no room for that, the same reasoning dispatch_tanzania.py's
own module docstring already gives for keeping ITS dispatches out of
the render loop.

Usage:
    python3 run_body_builder.py [--interval SECONDS] [--cycles N]

    --interval: seconds to sleep between cycles (default 60).
    --cycles:   stop after N cycles instead of running forever
                (omit to run until interrupted -- Ctrl+C).
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import random
import sys
import time
from pathlib import Path

ORGANISM_DIR = Path(__file__).resolve().parent.parent
MODULES_DIR = ORGANISM_DIR / "modules"
TOOLS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(MODULES_DIR))
sys.path.insert(0, str(TOOLS_DIR))

import body_builder as bb
import dispatch_tanzania as dt
from real_systems import build_world, registers_as_body


def _load_sim():
    sim_path = ORGANISM_DIR.parents[1] / "orbital-sandbox.py"
    spec = importlib.util.spec_from_file_location("sim", sim_path)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    return sim


def eligible_bodies() -> list[str]:
    """
    Every body real_systems.registers_as_body() accepts, for whichever
    world.mode organism.json is currently configured with -- reads the
    live config fresh each call rather than caching it, so a config
    change (e.g. switching to proxima mode) takes effect on this
    process's next cycle without needing a restart.
    """
    sim = _load_sim()
    config = json.loads(
        (ORGANISM_DIR / "organism.json").read_text(encoding="utf-8-sig")
    )
    world_mode = config.get("world", {}).get("mode", "solar_system")
    world = build_world(sim, world_mode)

    names = []
    if registers_as_body(world.SUN.radius):
        names.append(world.SUN.name)
    for body in world.PLANETS:
        if registers_as_body(body.radius):
            names.append(body.name)
    return names


def generate_via_tanzania(body_name: str, params: dict):
    """
    The generate_fn body_builder.evaluate_candidate() calls -- real
    dispatch to Tanzania, reusing generate_world_textures exactly as
    it already exists (no remote_runner.py changes needed), scoped to
    just this one body and this one candidate's params. Returns None
    on any failure (unreachable, remote error, malformed response) so
    the caller treats it as a rejected candidate, not a crash.
    """
    payload = {
        "task": "generate_world_textures",
        "bodies": [{"name": body_name, "seed": params["seed"]}],
        "resolution": params["resolution"],
        "octaves": params["octaves"],
    }
    try:
        result = dt.dispatch("generate_world_textures", payload=payload)
        heightmap = result.get("textures", {}).get(body_name, {}).get("heightmap")
        return heightmap
    except Exception as exc:
        print(f"  [{body_name}] dispatch failed: {exc}", file=sys.stderr)
        return None


def run(interval_s: float, max_cycles: int | None) -> None:
    rng = random.Random()
    state = bb.load_state()

    bodies = eligible_bodies()
    if not bodies:
        print("No Ceres-or-larger bodies found in the current world. Nothing to do.")
        return

    print(f"body-builder: {len(bodies)} eligible bodies: {', '.join(bodies)}")
    print(f"interval: {interval_s:.0f}s between cycles"
          + (f", stopping after {max_cycles} cycles" if max_cycles else ", running until interrupted"))
    print()

    body_cycle = itertools.cycle(bodies)
    cycle_count = 0

    while max_cycles is None or cycle_count < max_cycles:
        body_name = next(body_cycle)
        params = bb.current_params(state, body_name)
        move, candidate = bb.propose_candidate(params, rng)

        started = time.perf_counter()
        outcome = bb.evaluate_candidate(
            state, body_name, move, candidate,
            generate_fn=generate_via_tanzania,
        )
        elapsed = time.perf_counter() - started

        bb.save_state(state)

        verdict = "ACCEPTED" if outcome["accepted"] else "rejected"
        reason = f" -- {outcome['reason']}" if outcome.get("reason") else ""
        print(
            f"[{time.strftime('%H:%M:%S')}] {body_name:10s} gen={outcome['generation']:4d} "
            f"{move:20s} {verdict:8s} ({elapsed:.1f}s){reason}"
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
            f"params={body_state.get('current_params')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--cycles", type=int, default=None)
    args = parser.parse_args()

    run(args.interval, args.cycles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
