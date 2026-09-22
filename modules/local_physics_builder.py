from __future__ import annotations

"""
local-physics-builder: the third branch, alongside organism (universe-
scale brain) and body-builder (pure visual fidelity). Where organism's
brain predicts a body's place in the universe, this trains a SECOND,
independent NeuralLearner instance on real near-surface ballistics
(cannon's engine, dispatched via tools/remote_runner.py's
train_local_physics task) -- what happens inside a body's own sphere
of influence, not its orbit around anything else.

Each dispatch trains a genuinely fresh network from scratch (see
remote_runner.py's run_train_local_physics() -- re-seeded per call, not
resumed) and reports back a real final_validation_loss. This module's
job is exactly body_builder.py's shape applied to a different
yardstick: propose a candidate (here, just a new seed -- the move
vocabulary the brain-transplant experiment already explored was
training budget/architecture, not yet exposed as movable knobs here),
evaluate the real dispatched result, accept only if it's a genuine
improvement over the best ever seen for that body, persist either way.

Earth only, for now -- cannon/planet.py hardcodes its gravity model to
Earth, so there is no other body this can honestly train against yet.
"""

import json
import random
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parents[1] / "state"
STATE_PATH = STATE_DIR / "local_physics_builder.json"
HISTORY_PATH = STATE_DIR / "local_physics_builder_history.json"

MAX_HISTORY = 200

# cannon's engine only models Earth's gravity today (planet.py's
# MU_EARTH_M3_S2/R_EARTH_M are module constants, not parameters) --
# listed explicitly rather than derived, so it's obvious this is a
# real, known limitation and not an oversight.
SUPPORTED_BODIES = ("Earth",)


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


def propose_seed(rng: random.Random) -> int:
    return rng.randint(0, 2**32 - 1)


def evaluate_attempt(state: dict, body_name: str, seed: int, dispatch_fn) -> dict:
    """
    dispatch_fn(body_name, seed) -> real result dict from
    train_local_physics (or None on failure) is injected rather than
    imported directly, same separation-of-concerns reasoning as
    body_builder.evaluate_candidate()'s generate_fn.
    """
    body_state = state.setdefault("bodies", {}).get(body_name)
    if body_state is None:
        body_state = {
            "generation": 0, "accepted": 0,
            "best_final_loss": float("inf"), "best_seed": None,
        }
        state["bodies"][body_name] = body_state

    generation = body_state["generation"] + 1
    body_state["generation"] = generation

    result = dispatch_fn(body_name, seed)

    if result is None:
        outcome = {
            "body": body_name, "generation": generation, "seed": seed,
            "accepted": False, "reason": "dispatch failed",
        }
        _log_history(outcome)
        return outcome

    final_loss = float(result["final_validation_loss"])
    baseline = body_state["best_final_loss"]
    accepted = final_loss < baseline

    if accepted:
        body_state["best_final_loss"] = final_loss
        body_state["best_seed"] = seed
        body_state["accepted"] += 1

    outcome = {
        "body": body_name, "generation": generation, "seed": seed,
        "accepted": accepted,
        "final_loss": final_loss, "baseline": baseline,
        "self_evolve_accepted": result.get("self_evolve_accepted"),
        "self_evolve_cycles": result.get("self_evolve_cycles"),
        "active_loss_variant": result.get("active_loss_variant"),
        "active_activation_variant": result.get("active_activation_variant"),
    }
    _log_history(outcome)
    return outcome
