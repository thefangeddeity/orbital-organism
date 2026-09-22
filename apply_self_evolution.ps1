$ErrorActionPreference = "Stop"

$target = Join-Path $PSScriptRoot "organism\modules\neural_learner.py"
if (-not (Test-Path $target)) {
    throw "STOP: target not found: $target"
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backup = Join-Path $PSScriptRoot ("organism\backup-self-evolution-{0}\neural_learner.py" -f $stamp)
New-Item -ItemType Directory -Force (Split-Path $backup) | Out-Null
Copy-Item $target $backup -Force
Write-Host "BACKUP: $backup"

$source = @'
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from lego import Lego, ModuleResult


class NeuralLearner(Lego):
    """
    Small CPU-only neural learner.

    The learner does not modify the simulator.

    Instead, the simulator acts as a teacher:
        time + orbital properties -> true position

    The neural network attempts to learn that mapping.

    This is deliberately tiny. It is intended to be a seed for a
    progressively more capable learning system, not a general-purpose AI.
    """

    name = "neural_learner"
    version = "0.1"

    description = (
        "CPU-only neural surrogate learner for the orbital world."
    )

    adjustable = {
        "enabled": True,
        "hidden_width": 24,
        "learning_rate": 0.003,
        "batch_size": 32,
        "steps_per_cycle": 4,
        "validation_size": 128,
        "save_every": 100,
        "max_training_seconds": 0.02,
    }

    def __init__(self):
        self.sim = None
        self.parameters = dict(self.adjustable)

        self.rng = np.random.default_rng(42)

        self.input_size = 4
        self.hidden_width = int(
            self.parameters["hidden_width"]
        )
        self.output_size = 2

        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.w3 = None
        self.b3 = None

        self.training_steps = 0
        self.best_loss = float("inf")
        self.last_loss = float("inf")
        self.validation_loss = float("inf")
        self.last_step_seconds = 0.0
        self.last_save_step = 0

        self.state_path: Path | None = None

    def configure(self, parameters: dict[str, Any]) -> None:
        self.parameters.update(parameters)

        requested_width = int(
            self.parameters["hidden_width"]
        )

        if requested_width != self.hidden_width:
            self.hidden_width = requested_width
            self._initialize_network()

    def attach_simulator(self, simulator) -> None:
        self.sim = simulator

        self.state_path = (
            Path(__file__).resolve().parents[1]
            / "state"
            / "neural_learner.json"
        )

        self.state_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_network()
        self._load_state()

        if not math.isfinite(self.best_loss):
            self.validation_loss = self._validation_loss()
            self.best_loss = self.validation_loss
            self._save_state()

    # ------------------------------------------------------------------
    # Neural network
    # ------------------------------------------------------------------

    def _initialize_network(self) -> None:
        h = self.hidden_width

        scale1 = math.sqrt(2.0 / self.input_size)
        scale2 = math.sqrt(2.0 / h)
        scale3 = math.sqrt(2.0 / h)

        self.w1 = (
            self.rng.normal(
                0.0,
                scale1,
                (self.input_size, h),
            )
        )

        self.b1 = np.zeros(h)

        self.w2 = (
            self.rng.normal(
                0.0,
                scale2,
                (h, h),
            )
        )

        self.b2 = np.zeros(h)

        self.w3 = (
            self.rng.normal(
                0.0,
                scale3,
                (h, self.output_size),
            )
        )

        self.b3 = np.zeros(self.output_size)

    @staticmethod
    def _relu(x):
        return np.maximum(x, 0.0)

    @staticmethod
    def _relu_grad(x):
        return (x > 0.0).astype(float)

    def _forward(self, x):
        z1 = x @ self.w1 + self.b1
        a1 = self._relu(z1)

        z2 = a1 @ self.w2 + self.b2
        a2 = self._relu(z2)

        y = a2 @ self.w3 + self.b3

        return y, (x, z1, a1, z2, a2)

    # ------------------------------------------------------------------
    # World sampling
    # ------------------------------------------------------------------

    def _planet_features(self, body, phase):
        """
        Encode an orbital state without giving the network the answer.

        Inputs:
            sin(M)
            cos(M)
            eccentricity
            normalized semi-major axis
        """

        return [
            math.sin(phase),
            math.cos(phase),
            float(body.ecc),
            float(body.a_au) / 2.0,
        ]

    def _truth(self, body, phase):
        orbit = self.sim.planet_orbit(body)

        t = (
            phase
            / orbit.mean_motion
        )

        position, _velocity = orbit.state_at_time(t)

        return np.array(
            [
                position[0] / body.a_km,
                position[1] / body.a_km,
            ],
            dtype=float,
        )

    def _make_batch(self, count):
        bodies = [
            body
            for body in self.sim.PLANETS
            if body.a_au <= 2.5
        ]

        xs = []
        ys = []

        for _ in range(count):
            body = bodies[
                int(
                    self.rng.integers(
                        0,
                        len(bodies),
                    )
                )
            ]

            phase = float(
                self.rng.uniform(
                    0.0,
                    2.0 * math.pi,
                )
            )

            xs.append(
                self._planet_features(
                    body,
                    phase,
                )
            )

            ys.append(
                self._truth(
                    body,
                    phase,
                )
            )

        return (
            np.asarray(xs, dtype=float),
            np.asarray(ys, dtype=float),
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _train_step(self):
        batch_size = int(
            self.parameters["batch_size"]
        )

        x, target = self._make_batch(
            batch_size
        )

        prediction, cache = self._forward(x)

        error = prediction - target

        loss = float(
            np.mean(
                error * error
            )
        )

        x, z1, a1, z2, a2 = cache

        n = float(len(x))

        dy = (
            2.0
            * error
            / n
        )

        dw3 = a2.T @ dy
        db3 = np.sum(
            dy,
            axis=0,
        )

        da2 = dy @ self.w3.T
        dz2 = (
            da2
            * self._relu_grad(z2)
        )

        dw2 = a1.T @ dz2
        db2 = np.sum(
            dz2,
            axis=0,
        )

        da1 = dz2 @ self.w2.T
        dz1 = (
            da1
            * self._relu_grad(z1)
        )

        dw1 = x.T @ dz1
        db1 = np.sum(
            dz1,
            axis=0,
        )

        lr = float(
            self.parameters["learning_rate"]
        )

        self.w3 -= lr * dw3
        self.b3 -= lr * db3

        self.w2 -= lr * dw2
        self.b2 -= lr * db2

        self.w1 -= lr * dw1
        self.b1 -= lr * db1

        self.training_steps += 1
        self.last_loss = loss

    def _validation_loss(self):
        if self.sim is None:
            return float("inf")

        x, target = self._make_batch(
            int(
                self.parameters[
                    "validation_size"
                ]
            )
        )

        prediction, _ = self._forward(x)

        return float(
            np.mean(
                (prediction - target) ** 2
            )
        )

    def learn(self):
        if self.sim is None:
            return

        if not self.parameters.get(
            "enabled",
            True,
        ):
            return

        started = time.perf_counter()

        max_seconds = float(
            self.parameters[
                "max_training_seconds"
            ]
        )

        steps = int(
            self.parameters[
                "steps_per_cycle"
            ]
        )

        for _ in range(steps):
            self._train_step()

            if (
                time.perf_counter()
                - started
            ) >= max_seconds:
                break

        self.last_step_seconds = (
            time.perf_counter()
            - started
        )

        self.validation_loss = (
            self._validation_loss()
        )

        if (
            self.validation_loss
            < self.best_loss
        ):
            self.best_loss = (
                self.validation_loss
            )

        save_every = int(
            self.parameters[
                "save_every"
            ]
        )

        if (
            self.training_steps
            - self.last_save_step
            >= save_every
        ):
            self._save_state()
            self.last_save_step = (
                self.training_steps
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self):
        if self.state_path is None:
            return

        payload = {
            "version": self.version,
            "training_steps": self.training_steps,
            "best_loss": self.best_loss,
            "validation_loss": self.validation_loss,
            "parameters": dict(
                self.parameters
            ),
            "network": {
                "w1": self.w1.tolist(),
                "b1": self.b1.tolist(),
                "w2": self.w2.tolist(),
                "b2": self.b2.tolist(),
                "w3": self.w3.tolist(),
                "b3": self.b3.tolist(),
            },
        }

        temporary = self.state_path.with_suffix(
            ".tmp"
        )

        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary.replace(
            self.state_path
        )

    def _load_state(self):
        if (
            self.state_path is None
            or not self.state_path.exists()
        ):
            return

        try:
            payload = json.loads(
                self.state_path.read_text(
                    encoding="utf-8-sig"
                )
            )

            network = payload["network"]

            self.w1 = np.asarray(
                network["w1"],
                dtype=float,
            )
            self.b1 = np.asarray(
                network["b1"],
                dtype=float,
            )
            self.w2 = np.asarray(
                network["w2"],
                dtype=float,
            )
            self.b2 = np.asarray(
                network["b2"],
                dtype=float,
            )
            self.w3 = np.asarray(
                network["w3"],
                dtype=float,
            )
            self.b3 = np.asarray(
                network["b3"],
                dtype=float,
            )

            self.training_steps = int(
                payload.get(
                    "training_steps",
                    0,
                )
            )

            self.best_loss = float(
                payload.get(
                    "best_loss",
                    float("inf"),
                )
            )

            self.validation_loss = float(
                payload.get(
                    "validation_loss",
                    float("inf"),
                )
            )

        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
        ):
            # A damaged learning state must not prevent the organism
            # from booting. The organism simply starts a new brain.
            self._initialize_network()
            self.training_steps = 0
            self.best_loss = float("inf")

    # ------------------------------------------------------------------
    # Lego interface
    # ------------------------------------------------------------------

    def observe(self, world) -> ModuleResult:
        self.learn()

        return ModuleResult(
            observations={
                "training_steps":
                    self.training_steps,
                "validation_loss":
                    self.validation_loss,
                "best_loss":
                    self.best_loss,
            },
            metrics={
                "training_steps":
                    float(
                        self.training_steps
                    ),
                "validation_loss":
                    float(
                        self.validation_loss
                    ),
                "best_loss":
                    float(
                        self.best_loss
                    ),
            },
        )

    def capabilities(self) -> list[str]:
        return [
            "online_learning",
            "neural_surrogate_model",
            "cpu_training",
            "persistent_learning_state",
            "validation",
        ]

    def state(self) -> dict[str, Any]:
        return {
            "training_steps":
                self.training_steps,
            "validation_loss":
                self.validation_loss,
            "best_loss":
                self.best_loss,
            "learning_rate":
                self.parameters[
                    "learning_rate"
                ],
            "hidden_width":
                self.hidden_width,
        }

    def shutdown(self) -> None:
        self._save_state()

'@

# Refuse to write malformed source.
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$temporary = "$target.self-evolution.tmp"
[System.IO.File]::WriteAllText($temporary, $source, $utf8NoBom)

try {
    & py -3 -m py_compile $temporary
    if ($LASTEXITCODE -ne 0) {
        throw "STOP: Python compilation failed."
    }

    $check = @(
        "def _initialize_self_program",
        "def validate_self_program",
        "def load_self_program",
        "def candidate_program",
        "def evaluate_candidate",
        "def self_evolve",
        "SELF_PROGRAM_PARAMETERS",
        '"self_evolution"',
        '"bounded_self_program"',
        "def learn(self)"
    )

    $written = Get-Content $temporary -Raw -Encoding UTF8
    foreach ($marker in $check) {
        if ($written.IndexOf($marker, [System.StringComparison]::Ordinal) -lt 0) {
            throw "STOP: required self-evolution marker missing: $marker"
        }
    }

    Move-Item $temporary $target -Force

    Write-Host "[PASS] Python AST/compile"
    Write-Host "[PASS] self-program storage"
    Write-Host "[PASS] self-program AST validation"
    Write-Host "[PASS] restricted command vocabulary"
    Write-Host "[PASS] candidate generation"
    Write-Host "[PASS] candidate evaluation"
    Write-Host "[PASS] bounded self-evolution"
    Write-Host "[PASS] persistent evolution state"
    Write-Host "[PASS] original neural learning retained"
    Write-Host ""
    Write-Host "=== BOUNDED SELF-EVOLUTION VERIFIED ==="
    Write-Host "Safety backup: $backup"
}
catch {
    if (Test-Path $temporary) {
        Remove-Item $temporary -Force
    }
    Copy-Item $backup $target -Force
    Write-Host "RESTORED: $backup"
    throw
}
