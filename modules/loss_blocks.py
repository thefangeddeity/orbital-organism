from __future__ import annotations

"""
Block-based loss composer with per-primitive derivatives.

A loss here is not one of a handful of hand-written functions. It is a
composition:

    loss = mean( W(target) * C(prediction - target) )

where C is a *core penalty* primitive applied elementwise to the
residual, and W is a *weighting* primitive computed from the target.
Every core primitive ships its own analytic derivative, so the
gradient is derived from the same block that produced the value.

Why that matters: the previous implementation computed the forward
loss through whichever variant was active, but always applied the MSE
gradient (2*e/n) in the backward pass. Switching to MAE or Huber made
the reported loss and the actually-applied gradient disagree -- the
network was being trained on a different objective than the one being
measured. Pairing value and grad in one block makes that class of bug
unrepresentable.

6 core penalties x 4 weightings = 24 composable losses, versus the 4
that were hardcoded.
"""

import numpy as np


# ======================================================================
# CORE PENALTIES -- elementwise in the residual e = prediction - target
#
# Each entry is (value_fn, grad_fn) where grad_fn is d/de of value_fn.
# ======================================================================

HUBER_DELTA = 0.5


def _square(e):
    return e * e


def _square_grad(e):
    return 2.0 * e


def _absolute(e):
    return np.abs(e)


def _absolute_grad(e):
    return np.sign(e)


def _huber(e):
    a = np.abs(e)
    return np.where(
        a <= HUBER_DELTA,
        0.5 * e * e,
        HUBER_DELTA * (a - 0.5 * HUBER_DELTA),
    )


def _huber_grad(e):
    return np.where(np.abs(e) <= HUBER_DELTA, e, HUBER_DELTA * np.sign(e))


def _logcosh(e):
    # log(cosh(e)) computed stably: |e| + log1p(exp(-2|e|)) - log(2).
    # The naive cosh(e) overflows for |e| > ~710.
    a = np.abs(e)
    return a + np.log1p(np.exp(-2.0 * a)) - np.log(2.0)


def _logcosh_grad(e):
    return np.tanh(e)


def _quartic(e):
    return e ** 4


def _quartic_grad(e):
    return 4.0 * e ** 3


def _pseudo_huber(e):
    d2 = HUBER_DELTA * HUBER_DELTA
    return d2 * (np.sqrt(1.0 + (e / HUBER_DELTA) ** 2) - 1.0)


def _pseudo_huber_grad(e):
    return e / np.sqrt(1.0 + (e / HUBER_DELTA) ** 2)


CORE_PENALTIES = {
    "square": (_square, _square_grad),
    "absolute": (_absolute, _absolute_grad),
    "huber": (_huber, _huber_grad),
    "logcosh": (_logcosh, _logcosh_grad),
    "quartic": (_quartic, _quartic_grad),
    "pseudohuber": (_pseudo_huber, _pseudo_huber_grad),
}


# ======================================================================
# WEIGHTINGS -- functions of the target only
#
# The target is a constant with respect to the prediction, so these
# contribute no derivative term of their own; they just scale it.
# Shaped (n, 1) so they broadcast across output dimensions.
# ======================================================================

def _row_magnitude(target):
    return np.sqrt(np.sum(target * target, axis=1, keepdims=True))


def _w_uniform(target):
    return np.ones((target.shape[0], 1), dtype=float)


def _w_magnitude(target):
    # Emphasizes the outer bodies, whose coordinates are large.
    return 1.0 + _row_magnitude(target)


def _w_inverse_magnitude(target):
    # Emphasizes the inner bodies, which a magnitude-weighted or even
    # unweighted loss lets the outer planets dominate.
    return 1.0 / (1.0 + _row_magnitude(target))


def _w_log_magnitude(target):
    # Between the two: compresses the 0.4-to-30 AU spread.
    return 1.0 + np.log1p(_row_magnitude(target))


WEIGHTINGS = {
    "uniform": _w_uniform,
    "magnitude": _w_magnitude,
    "inverse": _w_inverse_magnitude,
    "logmagnitude": _w_log_magnitude,
}


# ======================================================================
# COMPOSITION
# ======================================================================

class ComposedLoss:
    """
    One composed loss: a core penalty and a weighting.

    value() and grad() are computed from the same pair of blocks, so
    they cannot describe different objectives.
    """

    def __init__(self, core: str = "square", weighting: str = "uniform"):
        if core not in CORE_PENALTIES:
            core = "square"
        if weighting not in WEIGHTINGS:
            weighting = "uniform"

        self.core = core
        self.weighting = weighting

    @property
    def name(self) -> str:
        return f"{self.core}/{self.weighting}"

    def value(self, prediction, target) -> float:
        error = prediction - target
        penalty, _ = CORE_PENALTIES[self.core]
        weights = WEIGHTINGS[self.weighting](target)
        return float(np.mean(weights * penalty(error)))

    def grad(self, prediction, target):
        """
        d(value)/d(prediction), same shape as prediction.

        The mean in value() divides by the full element count, so the
        same 1/N appears here.
        """
        error = prediction - target
        _, penalty_grad = CORE_PENALTIES[self.core]
        weights = WEIGHTINGS[self.weighting](target)
        n = float(prediction.size) or 1.0
        return weights * penalty_grad(error) / n


# The fixed yardstick evolution is judged against, deliberately NOT the
# loss being trained on.
#
# Candidate mutations are accepted or rejected by comparing validation
# loss before and after. If the mutation switches the loss function,
# those two numbers are in different units -- absolute-error values run
# systematically smaller than squared-error values for residuals under
# 1, so a mere unit change reads as a large improvement. Every
# candidate is therefore scored on this one stable metric regardless of
# what it trains on.
REFERENCE_METRIC = ComposedLoss("square", "uniform")


# Old single-name variants map onto composed pairs, so persisted
# self_program.json histories written before the composer existed still
# replay to the same behavior.
LEGACY_ALIASES = {
    "mse": ("square", "uniform"),
    "mae": ("absolute", "uniform"),
    "huber": ("huber", "uniform"),
    "weighted_mse": ("square", "magnitude"),
}


def from_name(name: str) -> ComposedLoss:
    """Accepts 'core/weighting' or a legacy single name."""
    if name in LEGACY_ALIASES:
        core, weighting = LEGACY_ALIASES[name]
        return ComposedLoss(core, weighting)

    if "/" in name:
        core, _, weighting = name.partition("/")
        return ComposedLoss(core, weighting)

    return ComposedLoss()


def check_gradients(seed: int = 0, tolerance: float = 1e-5) -> list[str]:
    """
    Finite-difference every core primitive's analytic derivative.

    A composer is only worth having if the derivatives are right, so
    this is checkable rather than asserted. Returns a list of failure
    descriptions; empty means every primitive agrees with its own
    numerical gradient.
    """
    rng = np.random.default_rng(seed)
    failures = []

    # Residuals spanning both sides of the Huber/pseudo-Huber knee, and
    # deliberately avoiding exact zero where |e| and sign(e) are not
    # differentiable.
    prediction = rng.uniform(-2.0, 2.0, size=(24, 2))
    target = rng.uniform(-2.0, 2.0, size=(24, 2))
    prediction = np.where(
        np.abs(prediction - target) < 1e-3, prediction + 0.05, prediction
    )

    step = 1e-6

    for core in CORE_PENALTIES:
        for weighting in WEIGHTINGS:
            loss = ComposedLoss(core, weighting)
            analytic = loss.grad(prediction, target)

            numeric = np.zeros_like(prediction)
            for index in np.ndindex(prediction.shape):
                up = prediction.copy()
                down = prediction.copy()
                up[index] += step
                down[index] -= step
                numeric[index] = (
                    loss.value(up, target) - loss.value(down, target)
                ) / (2.0 * step)

            worst = float(np.max(np.abs(analytic - numeric)))
            scale = max(1.0, float(np.max(np.abs(numeric))))

            if worst / scale > tolerance:
                failures.append(
                    f"{loss.name}: max |analytic - numeric| = {worst:.3e} "
                    f"(scaled {worst / scale:.3e})"
                )

    return failures


if __name__ == "__main__":
    problems = check_gradients()
    if problems:
        print("GRADIENT CHECK FAILED")
        for problem in problems:
            print(" ", problem)
        raise SystemExit(1)

    print(
        f"gradient check passed: "
        f"{len(CORE_PENALTIES)} cores x {len(WEIGHTINGS)} weightings = "
        f"{len(CORE_PENALTIES) * len(WEIGHTINGS)} composed losses"
    )
