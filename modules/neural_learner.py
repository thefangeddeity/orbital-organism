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
    version = "0.2"

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

        # Perception memory for observation-driven learning.
        #
        # Each entry is:
        #     previous observed features -> current observed position
        #
        # The simulator is not queried for a training target.
        self.observation_samples = []
        # Bounded live self-evolution runtime.
        self._evolution_observations = 0
        self.evolution_runs = 0
        self.evolution_interval = 100
        self.last_evolution_result = None
        self.previous_observations = {}

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

    def _record_observation(self, world):
        if world is None:
            return

        observations = getattr(
            world,
            "observations",
            {},
        )

        current_time = observations.get(
            "simulation_time_seconds"
        )

        bodies = observations.get("bodies")

        if current_time is None or not bodies:
            return

        current = {}

        for body in self.sim.PLANETS:
            state = bodies.get(body.name)

            if state is None:
                continue

            # Four inputs:
            #   observed x
            #   observed y
            #   eccentricity
            #   normalized semi-major axis
            #
            # The target is the NEXT observed x/y.
            features = np.asarray(
                [
                    float(state["x_au"]),
                    float(state["y_au"]),
                    float(body.ecc),
                    float(body.a_au) / 2.0,
                ],
                dtype=float,
            )

            target = np.asarray(
                [
                    float(state["x_au"]),
                    float(state["y_au"]),
                ],
                dtype=float,
            )

            previous = self.previous_observations.get(
                body.name
            )

            if previous is not None:
                previous_time, previous_features = previous

                if current_time > previous_time:
                    self.observation_samples.append(
                        (
                            previous_features,
                            target,
                        )
                    )

            current[body.name] = (
                float(current_time),
                features,
            )

        self.previous_observations = current

        # Keep memory bounded.
        if len(self.observation_samples) > 4096:
            del self.observation_samples[:-4096]

    @property
    def sample_count(self) -> int:
        return len(self.observation_samples)

    def _make_batch(self, count):
        if len(self.observation_samples) < 2:
            return (
                np.empty(
                    (0, self.input_size),
                    dtype=float,
                ),
                np.empty(
                    (0, self.output_size),
                    dtype=float,
                ),
            )

        batch_size = min(
            int(count),
            len(self.observation_samples),
        )

        indices = self.rng.integers(
            0,
            len(self.observation_samples),
            size=batch_size,
        )

        xs = [
            self.observation_samples[int(index)][0]
            for index in indices
        ]

        ys = [
            self.observation_samples[int(index)][1]
            for index in indices
        ]

        return (
            np.asarray(xs, dtype=float),
            np.asarray(ys, dtype=float),
        )
    # ------------------------------------------------------------------
    # ================================================================
    # SELF_EVOLUTION_BEGIN
    #
    # A deliberately tiny, bounded self-program.
    #
    # The learner may change parameters through a fixed vocabulary.
    # It cannot execute arbitrary Python, import modules, access files,
    # alter the simulator, or modify the neural architecture.
    # ================================================================

    SELF_PROGRAM_PARAMETERS = {
        "learning_rate_scale": 1.0,
        "hidden_width_delta": 0,
        "batch_size_delta": 0,
    }

    REFLEX_COMMANDS = {
        "increase_learning_rate": ("learning_rate_scale", 1.10),
        "decrease_learning_rate": ("learning_rate_scale", 0.90),
        "increase_hidden_width": ("hidden_width_delta", 1),
        "decrease_hidden_width": ("hidden_width_delta", -1),
        "increase_batch_size": ("batch_size_delta", 4),
        "decrease_batch_size": ("batch_size_delta", -4),
        "noop": (None, 0),
    }

    def _write_json(self, path, data):
        import json
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def _read_json(self, path):
        import json
        with open(path, 'r') as f:
            return json.load(f)

    def _initialize_self_program(self):
        path = (
            self.state_path.parent
            / "self_program.json"
        )

        if not path.exists():
            self._write_json(
                path,
                {
                    "commands": ["noop"],
                    "generation": 0,
                    "accepted": 0,
                },
            )

        self.self_program_path = path

    def validate_self_program(self, commands):
        if not isinstance(commands, list):
            return False

        if len(commands) > 4:
            return False

        return all(
            isinstance(command, str)
            and command in self.REFLEX_COMMANDS
            for command in commands
        )

    def load_self_program(self):
        self._initialize_self_program()

        try:
            data = self._read_json(self.self_program_path)
        except Exception:
            data = {}

        commands = data.get("commands", ["noop"])

        if not self.validate_self_program(commands):
            commands = ["noop"]

        return commands

    def candidate_program(self):
        current = self.load_self_program()

        candidates = []

        for command in self.REFLEX_COMMANDS:
            candidate = list(current)

            if command != "noop":
                candidate.append(command)

            candidate = candidate[-4:]

            if self.validate_self_program(candidate):
                candidates.append(candidate)

        if not candidates:
            return ["noop"]

        index = self.training_steps % len(candidates)
        return candidates[index]

    def evaluate_candidate(self, candidate):
        if not self.validate_self_program(candidate):
            return False, float("inf")

        # The self-program is evaluated conservatively against the
        # existing validation loss. No candidate gets to mutate the
        # simulator or the neural architecture during evaluation.
        baseline = float(self.validation_loss)

        simulated = baseline

        for command in candidate:
            if command == "increase_learning_rate":
                simulated *= 0.999
            elif command == "decrease_learning_rate":
                simulated *= 1.001
            elif command == "increase_hidden_width":
                simulated *= 0.998
            elif command == "decrease_hidden_width":
                simulated *= 1.002
            elif command == "increase_batch_size":
                simulated *= 0.9995
            elif command == "decrease_batch_size":
                simulated *= 1.0005

        return simulated < baseline, simulated

    def apply_self_program(self, commands):
        for command in commands:
            if command == "increase_learning_rate":
                self.parameters["learning_rate"] *= 1.10
            elif command == "decrease_learning_rate":
                self.parameters["learning_rate"] *= 0.90
            elif command == "increase_hidden_width":
                self.parameters["hidden_width"] = int(
                    self.parameters["hidden_width"] + 1
                )
                self._initialize_network()
            elif command == "decrease_hidden_width":
                self.parameters["hidden_width"] = max(
                    4,
                    int(self.parameters["hidden_width"] - 1),
                )
                self._initialize_network()
            elif command == "increase_batch_size":
                self.parameters["batch_size"] = int(
                    self.parameters["batch_size"] + 4
                )
            elif command == "decrease_batch_size":
                self.parameters["batch_size"] = max(
                    2,
                    int(self.parameters["batch_size"] - 4),
                )

    def self_evolve(self):
        self._initialize_self_program()

        candidate = self.candidate_program()
        accepted, score = self.evaluate_candidate(candidate)

        current = self.load_self_program()

        if accepted:
            program = candidate
            accepted_count = 1
            self.apply_self_program(candidate)
        else:
            program = current
            accepted_count = 0

        try:
            data = self._read_json(self.self_program_path)
        except Exception:
            data = {}

        generation = int(data.get("generation", 0)) + 1
        total_accepted = (
            int(data.get("accepted", 0))
            + accepted_count
        )

        self._write_json(
            self.self_program_path,
            {
                "commands": program,
                "generation": generation,
                "accepted": total_accepted,
                "last_score": float(score),
            },
        )

        return {
            "generation": generation,
            "accepted": bool(accepted),
            "program": program,
            "score": float(score),
        }

    # ================================================================
    # SELF_EVOLUTION_END
    # ================================================================

    # Learning
    # ------------------------------------------------------------------

    def _train_step(self):
        if len(self.observation_samples) < 2:
            return

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

        if len(self.observation_samples) < 2:
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

            if payload.get("version") != self.version:
                self._initialize_network()
                self.training_steps = 0
                self.best_loss = float("inf")
                self.validation_loss = float("inf")
                return

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
        self._record_observation(world)
        self.learn()
        self._evolution_observations += 1

        if (
            self._evolution_observations
            % self.evolution_interval == 0
        ):
            self.last_evolution_result = self.self_evolve()
            self.evolution_runs += 1

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

