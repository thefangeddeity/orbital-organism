#!/usr/bin/env python3
"""
Runs ON Tanzania (or any worker with SSH access + python3 + numpy).

Not a reimplementation of the learner's math -- imports the actual
NeuralLearner class from neural_learner.py (shipped alongside this
file by dispatch_tanzania.py) so whatever runs here is guaranteed
identical to what runs locally on 7elwe. No drift risk between "what
was tested" and "what actually ran remotely".

Usage:
    python3 remote_runner.py payload.json

payload.json shape:
    {
        "task": "explore_mutation_space",
        "parameters": {...NeuralLearner.parameters...},
        "network": {"w1": [...], "b1": [...], ...},
        "samples": [{"x": [x_au, y_au, ecc, a_au_half], "y": [x_au, y_au]}, ...]
    }

Prints one JSON object to stdout: the result. Nothing else goes to
stdout (diagnostics go to stderr) so the caller can parse stdout
directly as the result.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import loss_blocks
from neural_learner import NeuralLearner


def build_learner(payload: dict) -> NeuralLearner:
    learner = NeuralLearner()
    learner.sim = True  # only used as a truthiness guard, never dereferenced here
    learner.parameters.update(payload["parameters"])

    network = payload["network"]
    learner.w1 = np.asarray(network["w1"], dtype=float)
    learner.b1 = np.asarray(network["b1"], dtype=float)
    learner.w2 = np.asarray(network["w2"], dtype=float)
    learner.b2 = np.asarray(network["b2"], dtype=float)
    learner.w3 = np.asarray(network["w3"], dtype=float)
    learner.b3 = np.asarray(network["b3"], dtype=float)
    learner.hidden_width = learner.w1.shape[1]

    learner.observation_samples = [
        (
            np.asarray(sample["x"], dtype=float),
            np.asarray(sample["y"], dtype=float),
        )
        for sample in payload["samples"]
    ]

    return learner


# Fixed for now -- a real future direction is letting the organism
# itself learn/adjust this (more steps = more confident per combo but
# a slower sweep; the right tradeoff likely depends on how much spare
# compute Tanzania actually has at dispatch time, which the organism
# could in principle sense and adapt to, the same way it already
# tunes its own hyperparameters locally).
TRAIN_STEPS_PER_COMBO = 40


def run_explore_mutation_space(payload: dict) -> dict:
    """
    Try every core x weighting x activation combination -- the full
    block-composer space (6 x 4 x 3 = 72), versus the 12 the old
    hardcoded 4-name x 3-activation grid covered. This is exactly the
    kind of wide search a single local evolution cycle (one candidate,
    one mutation) could never afford to try, which is the actual case
    for dispatching it here instead of running it locally.

    Each combination gets a short real training burst (real gradient
    steps via the actual _train_step(), not a reimplementation) from
    the SAME starting weights before being scored. An earlier version
    of this just re-scored the current frozen weights under each
    combination's formula -- but loss/weighting choice has no effect
    on a forward pass at all (only weights and activation do), so that
    approach could only ever meaningfully compare activations. Every
    core/weighting pair scored identically to every other pair sharing
    an activation, confirmed by inspecting real output before this fix
    (both showed reference_score=0.011932 despite different
    weightings). Training briefly under each combination is what
    actually tests "would this loss function train the network
    better", which is the entire point of the sweep.

    Ranked by loss_blocks.REFERENCE_METRIC (plain square/uniform), not
    each combination's own native loss value -- the same fitness-
    yardstick fix applied locally in evaluate_candidate_real(). Ranking
    24 loss variants by their own reported values would repeat the
    exact unit-mismatch bug that fix closed: MAE-family values run
    systematically smaller than square-family values for sub-1
    residuals, so ranking by native loss would just rediscover
    "switch away from square" every time, regardless of which
    combination actually predicts best.

    Scored on a held-out batch drawn with its own dedicated RNG seed,
    separate from each combo's training draws -- not reused from
    _make_batch(), which pulls from self.rng and would otherwise risk
    the held-out set overlapping the very data a combo just trained on.
    """
    base = build_learner(payload)

    if len(base.observation_samples) < 2:
        return {
            "task": "explore_mutation_space",
            "sample_count": 0,
            "results": [],
            "best": None,
        }

    holdout_rng = np.random.default_rng(999)
    holdout_size = min(
        int(base.parameters["validation_size"]),
        len(base.observation_samples),
    )
    holdout_indices = holdout_rng.integers(
        0, len(base.observation_samples), size=holdout_size
    )
    holdout_x = np.asarray(
        [base.observation_samples[i][0] for i in holdout_indices]
    )
    holdout_target = np.asarray(
        [base.observation_samples[i][1] for i in holdout_indices]
    )

    results = []

    for core in loss_blocks.CORE_PENALTIES:
        for weighting in loss_blocks.WEIGHTINGS:
            for activation_name in base.ACTIVATION_VARIANTS:
                learner = build_learner(payload)
                learner.active_loss_variant = f"{core}/{weighting}"
                learner.active_activation_variant = activation_name

                for _ in range(TRAIN_STEPS_PER_COMBO):
                    learner._train_step()

                prediction, _ = learner._forward(holdout_x)

                reference_score = loss_blocks.REFERENCE_METRIC.value(
                    prediction, holdout_target
                )
                native_loss = learner.get_loss_function().value(
                    prediction, holdout_target
                )

                results.append({
                    "loss_variant": f"{core}/{weighting}",
                    "activation_variant": activation_name,
                    "reference_score": float(reference_score),
                    "validation_loss": float(native_loss),
                })

    results.sort(key=lambda r: r["reference_score"])

    return {
        "task": "explore_mutation_space",
        "sample_count": len(base.observation_samples),
        "combinations_tried": len(results),
        "baseline": {
            "loss_variant": payload.get("baseline_loss_variant", "square/uniform"),
            "activation_variant": payload.get("baseline_activation_variant", "relu"),
        },
        "results": results,
        "best": results[0] if results else None,
    }


# ======================================================================
# WORLD-BUILDING: procedural surface textures for L2/L3 fidelity.
#
# Multi-octave VALUE noise (bilinear-upsampled random grids, summed
# across octaves at doubling frequency) -- not Perlin/simplex, which
# would need an extra dependency neither side of this dispatch has
# installed. This is a real, deterministic (given a seed), genuinely
# more-expensive-at-higher-resolution computation -- the actual reason
# it's worth affording here rather than computing inline at boot: it
# runs once per body, gets cached, and the render loop never touches
# it again.
# ======================================================================

TEXTURE_RESOLUTION = 96
TEXTURE_OCTAVES = 5


def _value_noise_2d(
    rng: np.random.Generator, shape: tuple, octaves: int, persistence: float = 0.5
) -> np.ndarray:
    height, width = shape
    total = np.zeros(shape)
    amplitude = 1.0
    max_amplitude = 0.0
    freq = 4

    for _ in range(octaves):
        grid_h, grid_w = max(2, freq), max(2, freq)
        grid = rng.uniform(-1.0, 1.0, size=(grid_h, grid_w))

        ys = np.linspace(0, grid_h - 1, height)
        xs = np.linspace(0, grid_w - 1, width)
        y0 = np.floor(ys).astype(int)
        y1 = np.clip(y0 + 1, 0, grid_h - 1)
        x0 = np.floor(xs).astype(int)
        x1 = np.clip(x0 + 1, 0, grid_w - 1)
        wy = (ys - y0)[:, None]
        wx = (xs - x0)[None, :]

        top = grid[y0][:, x0] * (1 - wx) + grid[y0][:, x1] * wx
        bottom = grid[y1][:, x0] * (1 - wx) + grid[y1][:, x1] * wx
        layer = top * (1 - wy) + bottom * wy

        total += amplitude * layer
        max_amplitude += amplitude
        amplitude *= persistence
        freq *= 2

    return total / max_amplitude


def run_generate_world_textures(payload: dict) -> dict:
    """
    One procedural heightmap per body -- normalized to [-1, 1], meant
    to drive BOTH a facecolor tint (L2: mix the body's base color with
    a shade offset from the heightmap) and real mesh displacement
    (L3: perturb each vertex's radius by the same heightmap), so both
    fidelity levels read as textured/cratered views of the SAME
    generated surface, not two unrelated effects.

    Resolution/octaves are generous relative to what would be sane to
    block the organism's own boot on locally (see TEXTURE_RESOLUTION/
    TEXTURE_OCTAVES) -- this is the actual justification for dispatching
    world-building here at all: not because it's needed live, but
    because it's a one-time cost worth spending real compute on.
    """
    bodies = payload.get("bodies", [])
    resolution = int(payload.get("resolution", TEXTURE_RESOLUTION))
    octaves = int(payload.get("octaves", TEXTURE_OCTAVES))

    textures = {}
    for body in bodies:
        name = body["name"]
        seed = int(body.get("seed", abs(hash(name)) % (2**32)))
        rng = np.random.default_rng(seed)
        heightmap = _value_noise_2d(rng, (resolution, resolution), octaves)
        textures[name] = {
            "resolution": resolution,
            "heightmap": heightmap.tolist(),
        }

    return {
        "task": "generate_world_textures",
        # Echoed straight through, not re-derived -- the renderer that
        # eventually reads this cache needs to know which world it was
        # generated for, to refuse a stale/mismatched cache rather than
        # silently applying solar-system textures to a proxima world
        # (or vice versa) after a world.mode switch.
        "world_mode": payload.get("world_mode"),
        "resolution": resolution,
        "octaves": octaves,
        "textures": textures,
    }


# ======================================================================
# LOCAL-PHYSICS-BUILDER: a second brain instance, trained on real near-
# surface ballistics instead of the multi-body Kepler ephemeris the
# main organism's own brain learns. Third branch of the organism/
# body-builder/local-physics-builder split: organism's brain handles a
# body's place in the universe, this handles what happens inside its
# sphere of influence. Ported from the brain-transplant experiment
# (proved this works at all -- see that project's results/) into the
# real dispatch infrastructure instead of a disconnected standalone
# script with its own stale copy of these modules.
#
# Earth only, for now -- cannon/planet.py hardcodes MU_EARTH_M3_S2 and
# R_EARTH_M as module constants, so this genuinely can't cover another
# body yet without cannon's own engine gaining a way to swap gravity
# parameters. Not pretended otherwise.
# ======================================================================

LOCAL_PHYSICS_LENGTH_SCALE_M = 1000.0
LOCAL_PHYSICS_VELOCITY_SCALE_MS = 100.0
LOCAL_PHYSICS_ELEVATIONS_DEG = (25.0, 35.0, 45.0, 55.0, 65.0)
LOCAL_PHYSICS_TRAINING_STEPS = 4000
LOCAL_PHYSICS_SELF_EVOLVE_CYCLES = 30
LOCAL_PHYSICS_SAMPLE_INTERVAL_S = 0.05


def _generate_ballistics_samples(gun_mod, physics_mod):
    samples = []
    charges = (gun_mod.CHARGE_REFERENCE_KG * f for f in (0.85, 1.0, 1.15))

    for elevation_deg in LOCAL_PHYSICS_ELEVATIONS_DEG:
        for charge_kg in charges:
            initial_state, _bore = gun_mod.fire(
                gun_mod.GRIBEAUVAL_12PDR, gun_mod.IRON_SHOT_12PDR,
                charge_kg=charge_kg, elevation_rad=math.radians(elevation_deg),
            )
            muzzle = initial_state.pos
            result = physics_mod.integrate_flight(
                initial_state, drag_enabled=True, ground_enabled=True,
                sample_interval_s=LOCAL_PHYSICS_SAMPLE_INTERVAL_S,
            )

            prev = None
            for _t, state in result.samples:
                rel_x = (state.pos.x - muzzle.x) / LOCAL_PHYSICS_LENGTH_SCALE_M
                rel_y = (state.pos.y - muzzle.y) / LOCAL_PHYSICS_LENGTH_SCALE_M
                vx = state.vel.x / LOCAL_PHYSICS_VELOCITY_SCALE_MS
                vy = state.vel.y / LOCAL_PHYSICS_VELOCITY_SCALE_MS
                if prev is not None:
                    samples.append({"x": list(prev), "y": [rel_x, rel_y]})
                prev = (rel_x, rel_y, vx, vy)

    return samples


def run_train_local_physics(payload: dict) -> dict:
    """
    One real training + self-evolution run of a SECOND, independent
    NeuralLearner instance against real cannon ballistics data --
    reusing the exact same brain classes (NeuralLearner, loss_blocks,
    scratch_blocks) the main organism's own brain uses, unmodified.
    Each call trains a FRESH network from scratch (no persisted state
    between calls -- local_physics_builder.py on the caller side
    tracks whether each run's result is an improvement worth keeping,
    the training itself is stateless per dispatch).
    """
    cannon_lib_dir = Path(__file__).resolve().parent / "cannon_lib"
    sys.path.insert(0, str(cannon_lib_dir))
    from cannon import gun, physics as cannon_physics

    samples = _generate_ballistics_samples(gun, cannon_physics)

    learner = NeuralLearner()
    learner.sim = True  # truthiness guard only

    # NeuralLearner.__init__ fixes self.rng at seed 42 always -- every
    # dispatch would otherwise produce a BIT-IDENTICAL weight init,
    # training trajectory, and self-evolution choices, with nothing to
    # explore across repeat calls. Re-seeded from the payload BEFORE
    # _initialize_network() (which draws the initial weights from
    # self.rng), so a different payload seed genuinely means a
    # different, independent training run.
    seed = int(payload.get("seed", 42))
    learner.rng = np.random.default_rng(seed)

    # A fresh network, not one loaded from shipped weights (unlike
    # build_learner() above, which the other tasks use) -- this trains
    # from scratch every call, so the weights need real initialization
    # here rather than staying at their post-__init__ None/0-d state.
    learner._initialize_network()
    # self_evolve() needs state_path set (it derives self_program_path
    # from state_path.parent) -- a fresh temp dir each call, not this
    # host's real state/, so this stays genuinely stateless: no self_
    # program.json persisting across dispatches and quietly biasing a
    # "fresh" training run with a previous run's accepted mutations.
    import tempfile
    scratch_dir = Path(tempfile.mkdtemp(prefix="local_physics_"))
    learner.state_path = scratch_dir / "neural_learner.json"

    learner.observation_samples = [
        (np.asarray(s["x"], dtype=float), np.asarray(s["y"], dtype=float))
        for s in samples
    ]

    baseline_loss = learner._validation_loss()

    for _ in range(LOCAL_PHYSICS_TRAINING_STEPS):
        learner._train_step()

    trained_loss = learner._validation_loss()

    evolution_log = []
    for _ in range(LOCAL_PHYSICS_SELF_EVOLVE_CYCLES):
        result = learner.self_evolve()
        evolution_log.append({
            "generation": result["generation"],
            "accepted": result["accepted"],
            "score": result["score"],
        })

    final_loss = learner._validation_loss()
    accepted_count = sum(1 for e in evolution_log if e["accepted"])

    return {
        "task": "train_local_physics",
        "body": payload.get("body", "Earth"),
        "sample_count": len(samples),
        "baseline_validation_loss": float(baseline_loss),
        "trained_validation_loss": float(trained_loss),
        "final_validation_loss": float(final_loss),
        "self_evolve_cycles": LOCAL_PHYSICS_SELF_EVOLVE_CYCLES,
        "self_evolve_accepted": accepted_count,
        "active_loss_variant": learner.active_loss_variant,
        "active_activation_variant": learner.active_activation_variant,
    }


TASKS = {
    "explore_mutation_space": run_explore_mutation_space,
    "generate_world_textures": run_generate_world_textures,
    "train_local_physics": run_train_local_physics,
}


def main():
    if len(sys.argv) != 2:
        print("usage: remote_runner.py payload.json", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    task = payload.get("task")

    handler = TASKS.get(task)
    if handler is None:
        print(json.dumps({"error": f"unknown task '{task}'"}))
        sys.exit(1)

    result = handler(payload)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
