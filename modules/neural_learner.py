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

        # Code variant tracking: what loss, features, and activations are active
        self.active_loss_variant = "mse"
        self.active_feature_variant = "basic"
        self.active_activation_variant = "relu"

        # Track evolution genealogy
        self.code_variants_tried = []
        self.evolution_success_rate = 0.0

    # ================================================================
    # CODE VARIANT LIBRARY
    # ================================================================
    # Safe, pre-written loss functions, feature extractors, and
    # activation functions that the organism can try.
    # ================================================================

    @staticmethod
    def loss_mse(prediction, target):
        """Mean Squared Error."""
        error = prediction - target
        return float(np.mean(error * error))

    @staticmethod
    def loss_mae(prediction, target):
        """Mean Absolute Error."""
        error = np.abs(prediction - target)
        return float(np.mean(error))

    @staticmethod
    def loss_huber(prediction, target, delta=0.5):
        """Huber loss: robust to outliers."""
        error = prediction - target
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return float(
            np.mean(
                0.5 * quadratic**2 + delta * linear
            )
        )

    @staticmethod
    def loss_weighted_mse(prediction, target):
        """MSE with higher weight on larger displacements."""
        error = prediction - target
        magnitude = np.sqrt(np.sum(target**2, axis=1, keepdims=True))
        weight = 1.0 + magnitude
        return float(
            np.mean(weight * (error**2))
        )

    # Feature variants
    @staticmethod
    def features_basic(x_au, y_au, ecc, a_au):
        """Basic 4-element feature vector."""
        return np.asarray(
            [x_au, y_au, ecc, a_au / 2.0],
            dtype=float,
        )

    @staticmethod
    def features_extended(x_au, y_au, ecc, a_au):
        """Extended features: add normalized position magnitude and eccentricity squared."""
        r = np.sqrt(x_au**2 + y_au**2)
        return np.asarray(
            [
                x_au,
                y_au,
                ecc,
                a_au / 2.0,
                r,
                ecc**2,
            ],
            dtype=float,
        )

    @staticmethod
    def features_phase_based(x_au, y_au, ecc, a_au):
        """Phase-based: convert to angle + distance."""
        r = np.sqrt(x_au**2 + y_au**2)
        phase = np.arctan2(y_au, x_au)
        return np.asarray(
            [
                np.sin(phase),
                np.cos(phase),
                r,
                ecc,
            ],
            dtype=float,
        )

    # Activation variants
    @staticmethod
    def activation_relu(x):
        return np.maximum(x, 0.0)

    @staticmethod
    def activation_tanh(x):
        return np.tanh(x)

    @staticmethod
    def activation_gelu(x):
        return x * (0.5 * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
        )))

    # Lookup dictionaries for variants
    LOSS_VARIANTS = {
        "mse": loss_mse,
        "mae": loss_mae,
        "huber": loss_huber,
        "weighted_mse": loss_weighted_mse,
    }

    FEATURE_VARIANTS = {
        "basic": features_basic,
        "extended": features_extended,
        "phase_based": features_phase_based,
    }

    ACTIVATION_VARIANTS = {
        "relu": activation_relu,
        "tanh": activation_tanh,
        "gelu": activation_gelu,
    }

    def get_loss_function(self):
        return self.LOSS_VARIANTS.get(
            self.active_loss_variant,
            self.LOSS_VARIANTS["mse"],
        )

    def get_feature_function(self):
        return self.FEATURE_VARIANTS.get(
            self.active_feature_variant,
            self.FEATURE_VARIANTS["basic"],
        )

    def get_activation_function(self):
        return self.ACTIVATION_VARIANTS.get(
            self.active_activation_variant,
            self.ACTIVATION_VARIANTS["relu"],
        )

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

    def _activation_gradient(self, x):
        """Gradient of active activation function."""
        if self.active_activation_variant == "tanh":
            a = np.tanh(x)
            return 1.0 - a**2
        elif self.active_activation_variant == "gelu":
            # Approximate GELU gradient
            cdf = 0.5 * (1.0 + np.tanh(
                np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
            ))
            pdf = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
            return cdf + x * pdf * np.sqrt(2.0 / np.pi)
        else:  # ReLU
            return (x > 0.0).astype(float)

    def _forward(self, x):
        activation = self.get_activation_function()

        z1 = x @ self.w1 + self.b1
        a1 = activation(z1)

        z2 = a1 @ self.w2 + self.b2
        a2 = activation(z2)

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
        "switch_loss_mse": ("active_loss_variant", "mse"),
        "switch_loss_mae": ("active_loss_variant", "mae"),
        "switch_loss_huber": ("active_loss_variant", "huber"),
        "switch_loss_weighted": ("active_loss_variant", "weighted_mse"),
        "switch_features_basic": ("active_feature_variant", "basic"),
        "switch_features_extended": ("active_feature_variant", "extended"),
        "switch_features_phase": ("active_feature_variant", "phase_based"),
        "switch_activation_relu": ("active_activation_variant", "relu"),
        "switch_activation_tanh": ("active_activation_variant", "tanh"),
        "switch_activation_gelu": ("active_activation_variant", "gelu"),
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

    def snapshot_state(self):
        """Deep snapshot of network weights, parameters, and code variants."""
        return {
            "w1": self.w1.copy(),
            "b1": self.b1.copy(),
            "w2": self.w2.copy(),
            "b2": self.b2.copy(),
            "w3": self.w3.copy(),
            "b3": self.b3.copy(),
            "parameters": dict(self.parameters),
            "active_loss_variant": self.active_loss_variant,
            "active_feature_variant": self.active_feature_variant,
            "active_activation_variant": self.active_activation_variant,
            "validation_loss": self.validation_loss,
        }

    def restore_state(self, snapshot):
        """Restore network weights, parameters, and code variants from snapshot."""
        self.w1 = snapshot["w1"].copy()
        self.b1 = snapshot["b1"].copy()
        self.w2 = snapshot["w2"].copy()
        self.b2 = snapshot["b2"].copy()
        self.w3 = snapshot["w3"].copy()
        self.b3 = snapshot["b3"].copy()
        self.parameters = dict(snapshot["parameters"])
        self.active_loss_variant = snapshot["active_loss_variant"]
        self.active_feature_variant = snapshot["active_feature_variant"]
        self.active_activation_variant = snapshot["active_activation_variant"]
        self.validation_loss = snapshot["validation_loss"]

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

        if len(commands) > 6:
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

            candidate = candidate[-6:]

            if self.validate_self_program(candidate):
                candidates.append(candidate)

        if not candidates:
            return ["noop"]

        index = self.training_steps % len(candidates)
        return candidates[index]

    def evaluate_candidate_real(self, candidate):
        """
        Real transactional validation.

        Snapshot current state, apply candidate mutations,
        run a validation pass, measure actual loss.
        If worse or on exception, rollback and return False.
        """
        if not self.validate_self_program(candidate):
            return False, float("inf")

        baseline = float(self.validation_loss)

        # Take snapshot before mutation
        snapshot = self.snapshot_state()

        try:
            # Apply candidate mutations
            self.apply_self_program(candidate)

            # Run actual validation on real data
            validation_loss = self._validation_loss()

            if validation_loss >= baseline:
                # No improvement: rollback
                self.restore_state(snapshot)
                return False, validation_loss

            # Improved! Keep the mutations
            return True, validation_loss

        except Exception as e:
            # Crash or numerical instability: rollback and reject
            self.restore_state(snapshot)
            return False, float("inf")

    def apply_self_program(self, commands):
        """Apply mutations from accepted candidate program."""
        for command in commands:
            if command == "increase_learning_rate":
                self.parameters["learning_rate"] *= 1.10
            elif command == "decrease_learning_rate":
                self.parameters["learning_rate"] *= 0.90
            elif command == "increase_hidden_width":
                self.parameters["hidden_width"] = int(
                    self.parameters["hidden_width"] + 1
                )
                self.hidden_width = int(
                    self.parameters["hidden_width"]
                )
                self._initialize_network()
            elif command == "decrease_hidden_width":
                self.parameters["hidden_width"] = max(
                    4,
                    int(self.parameters["hidden_width"] - 1),
                )
                self.hidden_width = int(
                    self.parameters["hidden_width"]
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
            elif command.startswith("switch_loss_"):
                loss_name = command.replace("switch_loss_", "")
                if loss_name in self.LOSS_VARIANTS:
                    self.active_loss_variant = loss_name
            elif command.startswith("switch_features_"):
                feature_name = command.replace("switch_features_", "")
                if feature_name in self.FEATURE_VARIANTS:
                    self.active_feature_variant = feature_name
            elif command.startswith("switch_activation_"):
                activation_name = command.replace("switch_activation_", "")
                if activation_name in self.ACTIVATION_VARIANTS:
                    self.active_activation_variant = activation_name

    def self_evolve(self):
        """
        Transactional evolution with real validation.

        Generate candidate, apply mutations to a copy, validate on real data,
        rollback if worse, commit if better.
        """
        self._initialize_self_program()

        candidate = self.candidate_program()

        # Real transactional validation
        accepted, score = self.evaluate_candidate_real(candidate)

        current = self.load_self_program()

        if accepted:
            program = candidate
            accepted_count = 1
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

        # Track genealogy
        self.code_variants_tried.append({
            "generation": generation,
            "candidate": candidate,
            "accepted": accepted,
            "score": float(score),
        })

        # Keep genealogy bounded
        if len(self.code_variants_tried) > 100:
            del self.code_variants_tried[:-100]

        # Update success rate
        if total_accepted > 0:
            self.evolution_success_rate = (
                total_accepted / max(generation, 1)
            )

        self._write_json(
            self.self_program_path,
            {
                "commands": program,
                "generation": generation,
                "accepted": total_accepted,
                "last_score": float(score),
                "success_rate": float(
                    self.evolution_success_rate
                ),
            },
        )

        return {
            "generation": generation,
            "accepted": bool(accepted),
            "program": program,
            "score": float(score),
            "baseline": float(
                self.validation_loss
            ),
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

        loss_fn = self.get_loss_function()
        loss = loss_fn(prediction, target)

        error = prediction - target

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
            * self._activation_gradient(z2)
        )

        dw2 = a1.T @ dz2
        db2 = np.sum(
            dz2,
            axis=0,
        )

        da1 = dz2 @ self.w2.T
        dz1 = (
            da1
            * self._activation_gradient(z1)
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

        loss_fn = self.get_loss_function()
        return loss_fn(prediction, target)

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
            "self_evolution",
            "code_mutation",
            "transactional_rollback",
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
            "active_loss_variant":
                self.active_loss_variant,
            "active_feature_variant":
                self.active_feature_variant,
            "active_activation_variant":
                self.active_activation_variant,
            "evolution_runs":
                self.evolution_runs,
            "evolution_success_rate":
                self.evolution_success_rate,
        }

    def shutdown(self) -> None:
        self._save_state()

