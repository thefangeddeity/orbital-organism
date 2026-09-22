from __future__ import annotations

import math
from typing import Any


# ======================================================================
# FIDELITY LEVELS
#
# What the organism is "allowed" to render, from cheapest to most
# expensive. Each level's compute_factor is relative to L0 (=1.0), used
# to project cost before granting an upgrade. "implemented" is honest
# bookkeeping: consider_upgrade() will never grant a level whose render
# path doesn't actually exist yet.
# ======================================================================

FIDELITY_LEVELS = {
    0: {
        "name": "point",
        "description": "Bare dot markers, no physical extent.",
        "memory_per_body_kb": 1,
        "compute_factor": 1.0,
        "implemented": True,
    },
    1: {
        "name": "sphere",
        "description": "True-radius 3D spheres, mass-scaled.",
        "memory_per_body_kb": 48,
        "compute_factor": 2.5,
        "implemented": True,
    },
    2: {
        "name": "textured",
        "description": "UV-mapped surface texture on spheres.",
        "memory_per_body_kb": 512,
        "compute_factor": 5.0,
        "implemented": False,
    },
    3: {
        "name": "cratered",
        "description": "Procedural surface relief (craters, terrain).",
        "memory_per_body_kb": 2048,
        "compute_factor": 9.0,
        "implemented": False,
    },
}


class OrganicBudget:
    """
    Tracks the organism's metabolic resource budget ("ATP") and decides
    whether it has earned the right to grow into a higher simulation
    fidelity level.

    This class does not render anything and does not run arbitrary
    code. It only gates an integer level number that the renderer
    reads. Growth is proposed periodically; it is granted only if:

        1. the render path for the next level actually exists,
        2. recent learning progress cleared the improvement threshold,
        3. projected CPU/memory cost stays inside the configured caps.

    Denials are cheap and expected -- most checks will say no. That is
    the point: the organism grows only when it has earned it.
    """

    def __init__(self, config: dict[str, Any], body_count: int):
        self.body_count = max(1, body_count)

        self.fidelity_level = int(config.get("fidelity_level", 0))

        self.max_cpu_ms_per_frame = float(config.get("budget_cpu_ms_per_frame", 30.0))
        self.max_memory_mb = float(config.get("budget_memory_mb", 256.0))
        self.max_fidelity_level = int(config.get("max_fidelity_level", 3))

        self.upgrade_improvement_threshold = float(
            config.get("upgrade_if_improvement_pct", 0.05)
        )
        self.upgrade_check_every_n = max(
            1,
            int(config.get("try_upgrade_every_n_generations", 10)),
        )

        self.last_cpu_ms = 0.0
        self.last_memory_mb = self._estimate_memory_mb()
        self.upgrades_granted = 0
        self.upgrades_denied = 0
        self.last_report: dict[str, Any] = {}

        self._loss_at_last_check: float | None = None

    # ------------------------------------------------------------------
    # Frame-cost tracking
    # ------------------------------------------------------------------

    def record_frame_cost(self, cpu_ms: float) -> None:
        self.last_cpu_ms = cpu_ms
        self.last_memory_mb = self._estimate_memory_mb()

    def _estimate_memory_mb(self) -> float:
        level = FIDELITY_LEVELS[self.fidelity_level]
        kb = level["memory_per_body_kb"] * self.body_count
        return kb / 1024.0

    def cpu_headroom_pct(self) -> float:
        if self.max_cpu_ms_per_frame <= 0:
            return 0.0
        return 1.0 - (self.last_cpu_ms / self.max_cpu_ms_per_frame)

    def memory_headroom_pct(self) -> float:
        if self.max_memory_mb <= 0:
            return 0.0
        return 1.0 - (self.last_memory_mb / self.max_memory_mb)

    # ------------------------------------------------------------------
    # Growth decision
    # ------------------------------------------------------------------

    def _projected_cost(self, level: int) -> tuple[float, float]:
        current = FIDELITY_LEVELS[self.fidelity_level]
        target = FIDELITY_LEVELS[level]

        compute_ratio = target["compute_factor"] / current["compute_factor"]
        baseline_cpu = max(self.last_cpu_ms, 1.0)
        projected_cpu = baseline_cpu * compute_ratio

        projected_mem_kb = target["memory_per_body_kb"] * self.body_count
        projected_mem = projected_mem_kb / 1024.0

        return projected_cpu, projected_mem

    def consider_upgrade(self, generation: int, best_loss: float) -> dict[str, Any]:
        """
        Called periodically (by whoever coordinates evolution) to check
        whether a fidelity upgrade has been earned this generation.

        Returns a report dict describing what happened. Never raises --
        a malformed loss value or an unreachable level is just another
        reason to say no.
        """
        report: dict[str, Any] = {
            "checked": False,
            "granted": False,
            "reason": "not due",
            "level": self.fidelity_level,
        }

        if generation <= 0 or generation % self.upgrade_check_every_n != 0:
            return report

        report["checked"] = True

        if self.fidelity_level >= self.max_fidelity_level:
            report["reason"] = "already at configured max fidelity"
            self.last_report = report
            return report

        next_level = self.fidelity_level + 1

        if next_level not in FIDELITY_LEVELS:
            report["reason"] = f"no such level L{next_level}"
            self.last_report = report
            return report

        if not FIDELITY_LEVELS[next_level]["implemented"]:
            report["reason"] = f"L{next_level} render path not implemented yet"
            self.last_report = report
            return report

        if not math.isfinite(best_loss):
            report["reason"] = "best_loss not finite yet"
            self.last_report = report
            return report

        if self._loss_at_last_check is None:
            self._loss_at_last_check = best_loss
            report["reason"] = "establishing baseline"
            self.last_report = report
            return report

        if self._loss_at_last_check <= 0:
            improvement = 0.0
        else:
            improvement = (
                (self._loss_at_last_check - best_loss)
                / self._loss_at_last_check
            )

        self._loss_at_last_check = best_loss

        if improvement < self.upgrade_improvement_threshold:
            report["reason"] = (
                f"improvement {improvement:.1%} below "
                f"{self.upgrade_improvement_threshold:.1%} threshold"
            )
            self.last_report = report
            return report

        projected_cpu, projected_mem = self._projected_cost(next_level)

        if projected_cpu > self.max_cpu_ms_per_frame:
            self.upgrades_denied += 1
            report["reason"] = (
                f"CPU budget insufficient "
                f"({projected_cpu:.1f}ms projected > "
                f"{self.max_cpu_ms_per_frame:.1f}ms cap)"
            )
            self.last_report = report
            return report

        if projected_mem > self.max_memory_mb:
            self.upgrades_denied += 1
            report["reason"] = (
                f"memory budget insufficient "
                f"({projected_mem:.1f}MB projected > "
                f"{self.max_memory_mb:.1f}MB cap)"
            )
            self.last_report = report
            return report

        # Earned it.
        self.fidelity_level = next_level
        self.upgrades_granted += 1
        report["granted"] = True
        report["level"] = self.fidelity_level
        report["reason"] = (
            f"improvement {improvement:.1%} cleared threshold, "
            f"budget ok -> grew to L{self.fidelity_level} "
            f"({FIDELITY_LEVELS[self.fidelity_level]['name']})"
        )
        self.last_report = report
        return report

    def state(self) -> dict[str, Any]:
        level = FIDELITY_LEVELS[self.fidelity_level]
        return {
            "fidelity_level": self.fidelity_level,
            "fidelity_name": level["name"],
            "cpu_ms": self.last_cpu_ms,
            "cpu_cap_ms": self.max_cpu_ms_per_frame,
            "memory_mb": self.last_memory_mb,
            "memory_cap_mb": self.max_memory_mb,
            "upgrades_granted": self.upgrades_granted,
            "upgrades_denied": self.upgrades_denied,
        }
