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


def run_explore_mutation_space(payload: dict) -> dict:
    """
    Try every loss x feature x activation combination against a fixed
    held-out slice of the shipped samples. Reports validation loss for
    each -- a wider sweep than 7elwe would ever attempt live, since
    this machine isn't resource-capped the same way.

    Feature/activation variants are recorded per-combination but not
    actually re-extracted (the shipped samples already encode the
    "basic" feature layout) -- this first pass sweeps loss x
    activation, which is what's actually swappable post-hoc on a fixed
    dataset without re-deriving features from raw positions. Feature
    variant sweeps are future scope, noted honestly rather than faked.
    """
    base = build_learner(payload)

    results = []

    for loss_name in base.LOSS_VARIANTS:
        for activation_name in base.ACTIVATION_VARIANTS:
            learner = build_learner(payload)
            learner.active_loss_variant = loss_name
            learner.active_activation_variant = activation_name

            loss = learner._validation_loss()

            results.append({
                "loss_variant": loss_name,
                "activation_variant": activation_name,
                "validation_loss": float(loss),
            })

    results.sort(key=lambda r: r["validation_loss"])

    return {
        "task": "explore_mutation_space",
        "sample_count": len(base.observation_samples),
        "baseline": {
            "loss_variant": payload.get("baseline_loss_variant", "mse"),
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
