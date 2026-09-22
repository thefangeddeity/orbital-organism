from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

import loss_blocks
import scratch_blocks
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
        "grad_clip": 5.0,
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
        # Tracked SEPARATELY from best_loss, on the fixed REFERENCE_METRIC
        # rather than whatever loss is currently active -- an unconstrained
        # scratch-tree core has no non-negativity guarantee, so best_loss
        # itself can go negative (confirmed live: it did), which then
        # permanently zeroes OrganicBudget's own improvement calculation
        # (it treats a <=0 baseline as "no comparable signal" and reports
        # 0% forever after). OrganicBudget.consider_upgrade() is fed this
        # instead of best_loss -- same "judge growth on one stable
        # yardstick, not whatever's currently active" fix already applied
        # to evaluate_candidate_real() earlier this session.
        self.best_reference_score = float("inf")
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

        # Scratch-block-proposed core penalty, when set, takes priority
        # over active_loss_variant's named core (its weighting still
        # applies -- see get_loss_function()). None means "not using a
        # proposed core, use the named one instead".
        self.active_scratch_tree: dict | None = None

        # What _propose_scratch_core() actually tried this cycle --
        # {"tree", "source"} or None. Deliberately NOT part of
        # snapshot_state()/restore_state(): a rejected candidate's
        # active_scratch_tree gets rolled back, but we still want to
        # know what was attempted and that it lost, so self_evolve()
        # can log it to scratch_history.json regardless of outcome.
        self._last_scratch_attempt: dict | None = None

        # Track evolution genealogy
        self.code_variants_tried = []
        self.evolution_success_rate = 0.0
        self.generations_since_accepted = 0

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
        """
        Returns a loss_blocks.ComposedLoss, not a plain function --
        callers use .value(prediction, target) and
        .grad(prediction, target). Accepts both legacy single names
        ("mse") and composed names ("huber/magnitude").

        A scratch-proposed core (self.active_scratch_tree) takes
        priority over the named core when set, but still goes through
        the same weighting system -- weighting is orthogonal to
        whether the core came from the fixed library or was proposed.
        """
        current = loss_blocks.from_name(self.active_loss_variant)

        if self.active_scratch_tree is not None:
            return loss_blocks.ComposedLoss(
                weighting=current.weighting,
                custom_core=scratch_blocks.as_core_penalty(self.active_scratch_tree),
            )

        return current

    def _set_loss_core(self, core: str) -> None:
        current = loss_blocks.from_name(self.active_loss_variant)
        self.active_loss_variant = f"{core}/{current.weighting}"
        # Explicitly switching to a named core overrides any proposed
        # one -- otherwise a switch_core_* command would silently do
        # nothing while a scratch tree stayed in charge.
        self.active_scratch_tree = None

    def _propose_scratch_core(self) -> bool:
        """
        Generates a random block-tree candidate core penalty and, if
        it survives validation and the gradient self-check, adopts it
        as active_scratch_tree. Returns whether it was adopted.

        A tree failing the check is not an error -- see
        scratch_blocks.py's own self-test: ~30% of random trees are
        genuinely pathological (division near zero, evaluated at a
        sign()/min() kink) and SHOULD be rejected. Rejection here just
        means this particular mutation attempt is a no-op, same as any
        other candidate that doesn't pass evaluate_candidate_real().
        """
        consumed = self._consume_offline_scratch_proposal()

        if consumed is not None:
            tree, source = consumed
        else:
            tree = scratch_blocks.random_candidate_tree(self.rng, max_ops=3)
            source = "local_random"

        # A representative range of residuals to check the gradient
        # over, not the live batch -- this only needs to catch
        # pathological trees (near-zero divisions, kinks), not measure
        # real performance, which evaluate_candidate_real() does next
        # with the real reference metric.
        probe = np.linspace(-2.0, 2.0, 9)

        try:
            problems = scratch_blocks.check_gradients(tree, {"e": probe})
        except Exception:
            return False

        if problems:
            return False

        self.active_scratch_tree = tree
        # Recorded regardless of what evaluate_candidate_real() decides
        # next -- see the __init__ comment on why this survives a
        # rollback that active_scratch_tree itself doesn't.
        self._last_scratch_attempt = {"tree": tree, "source": source}
        return True

    def _consume_offline_scratch_proposal(self) -> tuple[dict, str] | None:
        """
        Picks up a tree left by tools/propose_scratch_tree.py, if a
        fresh, unconsumed one is waiting. That tool runs offline
        (against the real Gemini API when a key is configured, a local
        best-of-N search otherwise), so its result can only ever reach
        this live process by leaving state/ behind for a later cycle
        to read -- never injected mid-flight, same rule dispatch_
        tanzania.py's results follow.

        Re-validated here regardless of what proposed it: an external
        proposer's own say-so is never trusted, the same defense-in-
        depth attach_simulator() applies when reloading a persisted
        scratch_tree from self_program.json.

        Marks the proposal consumed (not deleted) so its provenance --
        which source produced it, when -- survives for later
        inspection, and so a crash between reading and marking it
        doesn't cause the same proposal to be silently skipped forever
        (it would just be tried again next cycle, harmless either way).
        """
        if self.state_path is None:
            return None

        proposal_path = self.state_path.parent / "scratch_proposal.json"

        try:
            data = self._read_json(proposal_path)
        except Exception:
            return None

        if not data or data.get("consumed"):
            return None

        tree = data.get("tree")
        if tree is None:
            return None

        try:
            scratch_blocks.validate(tree)
        except scratch_blocks.InvalidBlockTree:
            return None

        data["consumed"] = True
        try:
            temp = proposal_path.with_suffix(".tmp")
            temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temp.replace(proposal_path)
        except OSError:
            pass

        source = data.get("source") or "offline_unknown"
        return tree, source

    def _log_scratch_history(
        self, attempt: dict, accepted: bool, score: float, generation: int
    ) -> None:
        """
        Records a scratch-tree proposal's real outcome -- closes the
        loop tools/propose_scratch_tree.py's Gemini prompt was missing:
        without this, every Gemini call was stateless, with no way to
        know what it had already proposed or whether any of it helped.
        That tool reads this same file to build a digest for its next
        prompt.

        Only called for attempts that passed validation/gradient-check
        and got a real evaluate_candidate_real() verdict -- a tree
        rejected for being structurally pathological (~30% of random
        trees, expected per scratch_blocks.py's own self-test) never
        reaches here, since that says nothing about the shape being a
        bad LOSS FUNCTION, just an invalid expression.
        """
        if self.state_path is None:
            return

        history_path = self.state_path.parent / "scratch_history.json"

        try:
            history = self._read_json(history_path)
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []

        history.append({
            "generation": generation,
            "tree": attempt["tree"],
            "source": attempt["source"],
            "accepted": bool(accepted),
            "score": float(score) if math.isfinite(score) else None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        history = history[-self.MAX_SCRATCH_HISTORY:]

        try:
            temp = history_path.with_suffix(".tmp")
            temp.write_text(json.dumps(history, indent=2), encoding="utf-8")
            temp.replace(history_path)
        except OSError:
            pass

    def _set_loss_weighting(self, weighting: str) -> None:
        current = loss_blocks.from_name(self.active_loss_variant)
        self.active_loss_variant = f"{current.core}/{weighting}"

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

        if not math.isfinite(self.best_reference_score):
            self.best_reference_score = self._reference_score()
            self._save_state()

        # Seed the in-memory success counter from self_program.json's
        # "accepted" field, which is already correctly persisted on
        # every evolution cycle regardless of restarts -- simpler than
        # reconciling a separate live "generation" (attempts) counter
        # across a restart, and arguably the more meaningful number
        # anyway: how many times it has actually improved, not how many
        # times it merely tried.
        self._initialize_self_program()

        try:
            data = self._read_json(self.self_program_path)
            self.evolution_runs = int(data.get("accepted", 0))
            self.evolution_success_rate = float(data.get("success_rate", 0.0))
            self.generations_since_accepted = int(
                data.get("generations_since_accepted", 0)
            )
            self._replay_accepted_variants(data.get("commands", []))

            # scratch_tree is data, not a replayable command name (see
            # self_evolve()'s write for why "propose_scratch_core" the
            # command can't be replayed the way switch_core_* can be).
            # Re-validated on load, not just trusted -- defense in
            # depth against a hand-edited or corrupted state file
            # introducing something outside the block vocabulary.
            scratch_tree = data.get("scratch_tree")
            if scratch_tree is not None:
                try:
                    scratch_blocks.validate(scratch_tree)
                    self.active_scratch_tree = scratch_tree
                except scratch_blocks.InvalidBlockTree:
                    self.active_scratch_tree = None
        except Exception:
            pass

    def _replay_accepted_variants(self, commands: list[str]) -> None:
        """
        Reconstruct which loss/feature/activation variant was actually
        active when the process last shut down.

        This was a real gap: active_loss_variant etc. were never saved
        or loaded at all, so every restart silently reset to mse/basic
        /relu even after self-evolution had accepted something better
        -- discovered when a live run had already accepted
        switch_loss_huber (self_program.json: generation 3, accepted
        1) but a fresh boot would have started back at mse regardless.

        Deliberately does NOT reuse apply_self_program() here: that
        also handles the parameter-delta commands (learning_rate,
        hidden_width, batch_size), which are already correctly
        captured in the persisted `parameters` dict via _load_state().
        Replaying those too would double-apply the scaling and, for
        hidden_width, re-trigger _initialize_network() and destroy the
        weights that were just correctly loaded. Only the pure
        variant-switch commands are safe to replay verbatim.
        """
        for command in commands:
            if command.startswith("switch_loss_"):
                name = command.replace("switch_loss_", "")
                # "switch_loss_weighted" -> legacy name "weighted_mse";
                # the old lookup checked self.LOSS_VARIANTS, whose keys
                # don't include the bare "weighted" suffix, so this
                # exact command silently no-opped before.
                legacy_name = {"weighted": "weighted_mse"}.get(name, name)
                if legacy_name in loss_blocks.LEGACY_ALIASES:
                    self.active_loss_variant = legacy_name
            elif command.startswith("switch_core_"):
                core_name = command.replace("switch_core_", "")
                if core_name in loss_blocks.CORE_PENALTIES:
                    self._set_loss_core(core_name)
            elif command.startswith("switch_weighting_"):
                weighting_name = command.replace("switch_weighting_", "")
                if weighting_name in loss_blocks.WEIGHTINGS:
                    self._set_loss_weighting(weighting_name)
            elif command.startswith("switch_features_"):
                name = command.replace("switch_features_", "")
                if name in self.FEATURE_VARIANTS:
                    self.active_feature_variant = name
            elif command.startswith("switch_activation_"):
                name = command.replace("switch_activation_", "")
                if name in self.ACTIVATION_VARIANTS:
                    self.active_activation_variant = name

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

    def _make_holdout_batch(self, count):
        """
        Like _make_batch(), but drawn with a dedicated RNG that never
        shares state with self.rng -- self.rng is what _train_step()
        draws from too, so a held-out set sampled from it could
        coincidentally overlap the very steps a candidate trains on
        between the before/after comparison in evaluate_candidate_real().
        Fixed seed, so this always draws the same relative slice
        (same discipline tools/remote_runner.py's sweep uses with its
        own dedicated holdout RNG) -- not meant to be the same physical
        samples forever, just a reproducible sampling pattern applied
        to whatever the current (possibly-evicted, capped at 4096)
        pool is.
        """
        if len(self.observation_samples) < 2:
            return (
                np.empty((0, self.input_size), dtype=float),
                np.empty((0, self.output_size), dtype=float),
            )

        holdout_rng = np.random.default_rng(20260921)
        batch_size = min(int(count), len(self.observation_samples))
        indices = holdout_rng.integers(
            0, len(self.observation_samples), size=batch_size
        )

        xs = [self.observation_samples[int(i)][0] for i in indices]
        ys = [self.observation_samples[int(i)][1] for i in indices]

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

    # Real gradient steps run between applying a candidate and scoring
    # it -- most mutations (core/weighting/scratch-core switches,
    # learning_rate, batch_size) have literally zero effect on a
    # forward pass by themselves; only activation and hidden_width
    # changes touch it directly. Confirmed by direct test: applying
    # switch_core_huber with no training in between left the reference
    # score bit-for-bit identical. Small and fixed for now (this check
    # runs roughly every evolution_interval observations, inline in
    # the render loop's tick, so it has to stay cheap) -- same future
    # self-tuning candidate as TRAIN_STEPS_PER_COMBO in
    # tools/remote_runner.py.
    LOCAL_CANDIDATE_TRAIN_STEPS = 12

    # Bounds scratch_history.json -- oldest entries drop first. Recent
    # results are what an offline proposer can actually act on; an
    # unbounded log isn't more useful and would grow forever.
    MAX_SCRATCH_HISTORY = 50

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
        # Block-based composer: core penalty and weighting mutate
        # independently, 6 x 4 = 24 reachable combinations versus the
        # 4 fixed pairs above (kept for backward compatibility with
        # already-persisted self_program.json histories).
        "switch_core_square": ("active_loss_variant", "core:square"),
        "switch_core_absolute": ("active_loss_variant", "core:absolute"),
        "switch_core_huber": ("active_loss_variant", "core:huber"),
        "switch_core_logcosh": ("active_loss_variant", "core:logcosh"),
        "switch_core_quartic": ("active_loss_variant", "core:quartic"),
        "switch_core_pseudohuber": ("active_loss_variant", "core:pseudohuber"),
        "switch_weighting_uniform": ("active_loss_variant", "weighting:uniform"),
        "switch_weighting_magnitude": ("active_loss_variant", "weighting:magnitude"),
        "switch_weighting_inverse": ("active_loss_variant", "weighting:inverse"),
        "switch_weighting_logmagnitude": ("active_loss_variant", "weighting:logmagnitude"),
        "switch_features_basic": ("active_feature_variant", "basic"),
        "switch_features_extended": ("active_feature_variant", "extended"),
        "switch_features_phase": ("active_feature_variant", "phase_based"),
        "switch_activation_relu": ("active_activation_variant", "relu"),
        "switch_activation_tanh": ("active_activation_variant", "tanh"),
        "switch_activation_gelu": ("active_activation_variant", "gelu"),
        # Genuinely new, not a recombination of the fixed library --
        # generates a random block-tree (modules/scratch_blocks.py),
        # validates it, gradient-checks it, and only then lets it
        # compete for acceptance like any other candidate.
        "propose_scratch_core": ("active_scratch_tree", "generated"),
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
            # json round-trip, not the dict object itself -- a shallow
            # copy would leave the snapshot aliased to the live tree,
            # so a later apply_self_program() mutating it in place
            # (it doesn't today, but nothing should rely on that) would
            # corrupt the snapshot too. Trees are capped at 24 nodes,
            # so this is cheap.
            "active_scratch_tree": (
                json.loads(json.dumps(self.active_scratch_tree))
                if self.active_scratch_tree is not None else None
            ),
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
        self.active_scratch_tree = snapshot.get("active_scratch_tree")
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

        # NOT self.training_steps -- that advances by a near-constant
        # ~424/generation (steps_per_cycle * evolution_interval +
        # evaluate_candidate_real()'s own training bursts), and
        # gcd(424, len(candidates)) trapped the cycle in a closed loop.
        # Confirmed live with len(candidates)=28: only residues
        # {0,4,8,...,24} were EVER reachable, permanently excluding 21
        # of 28 commands -- including propose_scratch_core -- not
        # rarely selecting them, never selecting them, for 1386+ real
        # generations. self_program.json's generation count increments
        # by exactly 1 every single call, so gcd(1, N) = 1 always --
        # every command gets a fair turn every len(candidates)
        # generations, immune to any such collision.
        try:
            generation = int(
                self._read_json(self.self_program_path).get("generation", 0)
            )
        except Exception:
            generation = self.training_steps  # unreadable state: degrade, don't crash

        index = generation % len(candidates)
        return candidates[index]

    def _reference_score(self, batch=None):
        """
        Fixed yardstick for evolution accept/reject, deliberately
        independent of whichever loss function is currently active.

        evaluate_candidate_real() used to compare self.validation_loss
        (under the OLD active variant) against a freshly computed
        _validation_loss() (under whatever the candidate just switched
        to) whenever the candidate included a loss-changing command.
        Those two numbers are in different units -- MAE runs
        systematically smaller than MSE for residuals under 1, so
        switching mse -> mae read as a large "improvement" from the
        unit change alone, regardless of whether the network actually
        predicts better. Every candidate is judged on this one stable
        metric instead, no matter what it trains on.
        """
        return self._validation_loss(metric=loss_blocks.REFERENCE_METRIC, batch=batch)

    def evaluate_candidate_real(self, candidate):
        """
        Real transactional validation.

        Snapshot current state, apply candidate mutations,
        run a validation pass, measure actual loss.
        If worse or on exception, rollback and return False.
        """
        if not self.validate_self_program(candidate):
            return False, float("inf")

        # ONE shared, dedicated-RNG held-out batch for scoring both
        # runs -- not drawn from self.rng (which the training bursts
        # below also draw from). Confirmed by direct test: calling
        # _reference_score() repeatedly with ZERO mutations applied
        # produced a 0.137-0.244 spread from sampling noise alone,
        # comfortably large enough to flip an accept/reject decision
        # regardless of whether a candidate does anything at all.
        batch = self._make_holdout_batch(int(self.parameters["validation_size"]))

        pristine_snapshot = self.snapshot_state()
        rng_state = self.rng.bit_generator.state

        try:
            # baseline = the CURRENT (unmutated) config, given the
            # SAME number of additional training steps the candidate
            # is about to get -- not a zero-training baseline. Most
            # mutations (core/weighting/scratch-core switches,
            # learning_rate, batch_size) have no effect on the forward
            # pass by themselves (confirmed: applying switch_core_huber
            # alone left the reference score bit-for-bit identical), so
            # SOME training is required to test them at all. But
            # comparing "mutated + trained" against "unmutated,
            # untrained" would make training itself look like the
            # mutation's doing -- confirmed by direct test: noop
            # (which apply_self_program() doesn't even have a branch
            # for -- it's a genuine no-op) "won" 20/20 times against a
            # zero-training baseline, which only proves training helps,
            # not that any particular mutation does. Both runs get the
            # same LOCAL_CANDIDATE_TRAIN_STEPS from the same starting
            # weights, isolating the mutation's own effect.
            for _ in range(self.LOCAL_CANDIDATE_TRAIN_STEPS):
                self._train_step()

            baseline = self._reference_score(batch=batch)

            # Back to the exact untouched starting point -- weights,
            # parameters, AND the RNG stream, so the candidate run
            # below draws the identical sequence of training
            # mini-batches the baseline run just did. Without resetting
            # rng_state too, the two runs would differ in which samples
            # they trained on, reintroducing a smaller version of the
            # same sampling-noise problem the holdout batch already
            # fixed for scoring.
            self.restore_state(pristine_snapshot)
            self.rng.bit_generator.state = rng_state

            self.apply_self_program(candidate)

            for _ in range(self.LOCAL_CANDIDATE_TRAIN_STEPS):
                self._train_step()

            # Judged on the fixed reference metric, not whatever loss
            # the candidate just switched to.
            reference_after = self._reference_score(batch=batch)

            if reference_after >= baseline:
                # No improvement: rollback to the untouched start, not
                # to the post-baseline-training state.
                self.restore_state(pristine_snapshot)
                return False, reference_after

            # Improved. Keep the mutation (and the training it just
            # received), and refresh the displayed validation_loss/
            # best_loss under whatever variant is now active -- a
            # switch_loss_*/switch_core_*/switch_weighting_* command
            # just changed what that number means.
            self.validation_loss = self._validation_loss()
            if self.validation_loss < self.best_loss:
                self.best_loss = self.validation_loss

            return True, reference_after

        except Exception:
            # Crash or numerical instability: rollback and reject
            self.restore_state(pristine_snapshot)
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
                legacy_name = {"weighted": "weighted_mse"}.get(loss_name, loss_name)
                if legacy_name in loss_blocks.LEGACY_ALIASES:
                    self.active_loss_variant = legacy_name
                    self.active_scratch_tree = None
            elif command == "propose_scratch_core":
                self._propose_scratch_core()
            elif command.startswith("switch_core_"):
                core_name = command.replace("switch_core_", "")
                if core_name in loss_blocks.CORE_PENALTIES:
                    self._set_loss_core(core_name)
            elif command.startswith("switch_weighting_"):
                weighting_name = command.replace("switch_weighting_", "")
                if weighting_name in loss_blocks.WEIGHTINGS:
                    self._set_loss_weighting(weighting_name)
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

        # Cleared before every cycle so a later check of this attribute
        # can only ever reflect THIS candidate's attempt, never a stale
        # one left over from a previous cycle that didn't propose a
        # scratch core at all.
        self._last_scratch_attempt = None

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

        if self._last_scratch_attempt is not None:
            self._log_scratch_history(
                self._last_scratch_attempt, accepted, score, generation
            )

        # Plateau tracking: how many generations since the last accepted
        # mutation. Local search tries exactly one candidate per cycle
        # (a single parameter/variant change) -- this is the signal
        # that hill-climbing has stalled and a wider search (more than
        # one cycle could ever afford to try) might actually be worth
        # dispatching somewhere with real spare compute.
        if accepted:
            generations_since_accepted = 0
        else:
            generations_since_accepted = (
                int(data.get("generations_since_accepted", 0)) + 1
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
                "generations_since_accepted": generations_since_accepted,
                # Unconditional, not "only if accepted": if this cycle
                # was rejected, restore_state() already reverted
                # active_scratch_tree to whatever it was before this
                # attempt, so writing it here always reflects the true
                # current state either way. Distinct from `commands`,
                # since "propose_scratch_core" (the command NAME) is
                # non-deterministic -- replaying it would generate a
                # DIFFERENT random tree, not reconstruct the one that
                # was actually tested and accepted. The specific tree
                # has to be persisted as data, not implied by a command
                # name the way every other mutation can be.
                "scratch_tree": self.active_scratch_tree,
            },
        )

        self.generations_since_accepted = generations_since_accepted

        return {
            "generation": generation,
            "accepted": bool(accepted),
            "program": program,
            "score": float(score),
            "baseline": float(
                self.validation_loss
            ),
            "generations_since_accepted": generations_since_accepted,
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
        loss = loss_fn.value(prediction, target)

        x, z1, a1, z2, a2 = cache

        # The actual derivative of whatever loss is active, not a
        # hardcoded MSE gradient -- previously this was always
        # 2*error/n regardless of active_loss_variant, so switching to
        # MAE/Huber/etc. changed the reported loss but silently kept
        # training on the MSE gradient. Value and grad now come from
        # the same composed-loss block, so they cannot disagree.
        dy = loss_fn.grad(prediction, target)

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

        # Global-norm gradient clipping. This hand-rolled MLP has no
        # batch normalization and trains on unnormalized, heterogeneous
        # -scale AU inputs (Mercury ~0.4 AU next to Neptune ~30 AU in
        # the same batch) for hundreds of unclipped SGD steps -- exactly
        # the conditions that blow a plain ReLU net up to astronomical
        # weight values within the first few steps. Clipping the
        # combined gradient norm preserves direction while bounding
        # magnitude, the standard fix for this failure mode.
        clip = float(self.parameters.get("grad_clip", 5.0))

        grads = [dw1, db1, dw2, db2, dw3, db3]

        total_norm = math.sqrt(
            sum(float(np.sum(g * g)) for g in grads)
        )

        if total_norm > clip and total_norm > 0.0:
            scale = clip / total_norm
            dw1, db1, dw2, db2, dw3, db3 = (g * scale for g in grads)

        lr = float(
            self.parameters["learning_rate"]
        )

        # Transactional step: snapshot, apply, and only commit if the
        # result is still finite -- same rollback-on-failure philosophy
        # as self-evolution's candidate mutations, applied here to
        # ordinary gradient steps as the last line of defense against
        # a step that still diverges despite clipping.
        pre_step = (
            self.w1.copy(), self.b1.copy(),
            self.w2.copy(), self.b2.copy(),
            self.w3.copy(), self.b3.copy(),
        )

        self.w3 -= lr * dw3
        self.b3 -= lr * db3

        self.w2 -= lr * dw2
        self.b2 -= lr * db2

        self.w1 -= lr * dw1
        self.b1 -= lr * db1

        if not all(
            np.all(np.isfinite(w))
            for w in (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)
        ):
            self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = pre_step
            return

        self.training_steps += 1
        self.last_loss = loss

    def _validation_loss(self, metric=None, batch=None):
        if self.sim is None:
            return float("inf")

        if len(self.observation_samples) < 2:
            return float("inf")

        if batch is not None:
            x, target = batch
        else:
            x, target = self._make_batch(
                int(
                    self.parameters[
                        "validation_size"
                    ]
                )
            )

        prediction, _ = self._forward(x)

        loss_fn = metric or self.get_loss_function()
        return loss_fn.value(prediction, target)

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

        reference_score = self._reference_score()
        if reference_score < self.best_reference_score:
            self.best_reference_score = reference_score

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
            "best_reference_score": self.best_reference_score,
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

            loaded_w1 = np.asarray(network["w1"], dtype=float)
            loaded_b1 = np.asarray(network["b1"], dtype=float)
            loaded_w2 = np.asarray(network["w2"], dtype=float)
            loaded_b2 = np.asarray(network["b2"], dtype=float)
            loaded_w3 = np.asarray(network["w3"], dtype=float)
            loaded_b3 = np.asarray(network["b3"], dtype=float)

            loaded_weights = (
                loaded_w1, loaded_b1,
                loaded_w2, loaded_b2,
                loaded_w3, loaded_b3,
            )

            # A weight saved from a run that diverged before gradient
            # clipping existed (or any other future bug) must not get a
            # second life just because it made it to disk. Persisted
            # state is trusted only if it's finite and within a sane
            # magnitude -- otherwise this is treated exactly like a
            # version mismatch: start a fresh network rather than
            # resume from known-bad weights.
            MAX_SANE_WEIGHT = 1.0e4

            sane = all(
                np.all(np.isfinite(w)) and np.all(np.abs(w) < MAX_SANE_WEIGHT)
                for w in loaded_weights
            )

            if not sane:
                self._initialize_network()
                self.training_steps = 0
                self.best_loss = float("inf")
                self.validation_loss = float("inf")
                return

            self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = loaded_weights

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

            self.best_reference_score = float(
                payload.get(
                    "best_reference_score",
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

            # Count successes, not attempts -- this is what persists
            # cleanly across a restart (self_program.json's "accepted"
            # field already does, with no extra bookkeeping needed) and
            # it's the more honest number anyway.
            if self.last_evolution_result.get("accepted"):
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

