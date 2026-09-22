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


TASKS = {
    "explore_mutation_space": run_explore_mutation_space,
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
