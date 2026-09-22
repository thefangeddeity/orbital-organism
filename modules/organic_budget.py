from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None  # idle-headroom scaling degrades to the flat floor cap


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

        # budget_cpu_ms_per_frame is the FLOOR, not a hard ceiling -- the
        # number this process is guaranteed to have even when the rest
        # of the machine is busy (it's coordinator+renderer for the
        # whole fleet; see providers.local_cpu in organism.json, which
        # is why this stays conservative by default rather than 0).
        # When the rest of the machine is idle, _idle_headroom_multiplier()
        # scales it up, toward idle_cpu_multiplier_cap -- "use up to n-1
        # cores' worth when idle" -- rather than sitting at one fixed
        # number regardless of what else is happening on the box.
        self.max_cpu_ms_per_frame = float(config.get("budget_cpu_ms_per_frame", 30.0))
        self.max_memory_mb = float(config.get("budget_memory_mb", 256.0))
        self.max_fidelity_level = int(config.get("max_fidelity_level", 3))

        self.idle_aware = bool(config.get("idle_aware_cpu_budget", True))
        self.idle_cpu_multiplier_cap = float(
            config.get(
                "idle_cpu_multiplier_cap",
                max(1.0, float((os.cpu_count() or 2) - 1)),
            )
        )

        # Same floor-not-ceiling treatment as CPU, mirrored for memory --
        # but memory's real-time signal is "how much RAM is ACTUALLY
        # free right now" (psutil.virtual_memory().available), not a
        # static machine property like core count. A fixed "always
        # allow 1GB" ceiling would be unsafe on a machine that's
        # genuinely low on RAM, not just generous -- memory_safety_
        # fraction caps this process to a conservative slice of
        # whatever's currently free, same reasoning as never claiming
        # the whole machine's idle CPU either.
        self.idle_memory_multiplier_cap = float(
            config.get("idle_memory_multiplier_cap", 4.0)
        )
        self.memory_safety_fraction = float(
            config.get("memory_safety_fraction", 0.5)
        )

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

        self.state_path: Path | None = None

    # ------------------------------------------------------------------
    # Persistence
    #
    # Fidelity is earned growth, not a preference -- closing the window
    # must not silently reset it back to whatever organism.json's static
    # default says. Mirrors NeuralLearner's own state file: same
    # directory, same atomic tmp-then-replace write.
    # ------------------------------------------------------------------

    def attach_state_path(self, state_path: Path) -> None:
        self.state_path = state_path
        self._load_state()

    def _load_state(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return

        try:
            payload = json.loads(
                self.state_path.read_text(encoding="utf-8-sig")
            )

            loaded_level = int(payload.get("fidelity_level", 0))

            if 0 <= loaded_level <= self.max_fidelity_level:
                self.fidelity_level = loaded_level

            self.upgrades_granted = int(payload.get("upgrades_granted", 0))
            self.upgrades_denied = int(payload.get("upgrades_denied", 0))

            loaded_loss = payload.get("loss_at_last_check")
            if loaded_loss is not None and math.isfinite(loaded_loss):
                self._loss_at_last_check = float(loaded_loss)

        except (OSError, ValueError, KeyError, TypeError):
            # A damaged budget state must not block boot -- resume from
            # whatever organism.json's default says, same fallback
            # NeuralLearner uses for its own corrupted state.
            pass

    def save_state(self) -> None:
        if self.state_path is None:
            return

        payload = {
            "fidelity_level": self.fidelity_level,
            "upgrades_granted": self.upgrades_granted,
            "upgrades_denied": self.upgrades_denied,
            "loss_at_last_check": self._loss_at_last_check,
        }

        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

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

    def _idle_headroom_multiplier(self) -> float:
        """
        How far above the floor cap this process may currently spend,
        based on how idle the REST of the machine is right now.

        psutil.cpu_percent(interval=None) is non-blocking and reports
        the delta since the last call -- exactly the usage pattern for
        a value sampled once per frame in a tight loop, not something
        needing its own timer. system_busy_pct already includes this
        process's own usage, so idle_fraction reflects what everything
        ELSE on the machine is doing: at 0% other usage, the multiplier
        approaches idle_cpu_multiplier_cap (n-1 cores' worth); at 100%,
        it collapses back to 1.0x, the floor.
        """
        if not self.idle_aware or psutil is None:
            return 1.0

        try:
            system_busy_pct = psutil.cpu_percent(interval=None)
        except Exception:
            return 1.0

        idle_fraction = max(0.0, min(1.0, 1.0 - system_busy_pct / 100.0))
        multiplier = 1.0 + idle_fraction * (self.idle_cpu_multiplier_cap - 1.0)

        return max(1.0, min(multiplier, self.idle_cpu_multiplier_cap))

    def effective_cpu_cap_ms(self) -> float:
        return self.max_cpu_ms_per_frame * self._idle_headroom_multiplier()

    def cpu_headroom_pct(self) -> float:
        cap = self.effective_cpu_cap_ms()
        if cap <= 0:
            return 0.0
        return 1.0 - (self.last_cpu_ms / cap)

    def _idle_memory_multiplier(self) -> float:
        """
        How far above the memory floor this process may currently
        claim, based on real system RAM availability right now --
        psutil.virtual_memory().available, not a fixed ceiling. On the
        machine this was developed on (7.7GB total, observed at 89.6%
        used / 0.8GB available), a flat "always allow 1GB" cap would
        have been unsafe, not just generous; this scales down with it
        automatically the same way the CPU multiplier already does
        when the machine gets busy.
        """
        if not self.idle_aware or psutil is None or self.max_memory_mb <= 0:
            return 1.0

        try:
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
        except Exception:
            return 1.0

        safe_available_mb = available_mb * self.memory_safety_fraction
        multiplier = safe_available_mb / self.max_memory_mb

        return max(1.0, min(multiplier, self.idle_memory_multiplier_cap))

    def effective_memory_cap_mb(self) -> float:
        return self.max_memory_mb * self._idle_memory_multiplier()

    def memory_headroom_pct(self) -> float:
        cap = self.effective_memory_cap_mb()
        if cap <= 0:
            return 0.0
        return 1.0 - (self.last_memory_mb / cap)

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
        effective_cap = self.effective_cpu_cap_ms()

        if projected_cpu > effective_cap:
            self.upgrades_denied += 1
            report["reason"] = (
                f"CPU budget insufficient "
                f"({projected_cpu:.1f}ms projected > "
                f"{effective_cap:.1f}ms cap)"
            )
            self.last_report = report
            return report

        effective_memory_cap = self.effective_memory_cap_mb()

        if projected_mem > effective_memory_cap:
            self.upgrades_denied += 1
            report["reason"] = (
                f"memory budget insufficient "
                f"({projected_mem:.1f}MB projected > "
                f"{effective_memory_cap:.1f}MB cap)"
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
        multiplier = self._idle_headroom_multiplier()
        return {
            "fidelity_level": self.fidelity_level,
            "fidelity_name": level["name"],
            "cpu_ms": self.last_cpu_ms,
            "cpu_cap_ms": self.max_cpu_ms_per_frame * multiplier,
            "cpu_cap_floor_ms": self.max_cpu_ms_per_frame,
            "cpu_idle_multiplier": multiplier,
            "memory_mb": self.last_memory_mb,
            "memory_cap_mb": self.effective_memory_cap_mb(),
            "memory_cap_floor_mb": self.max_memory_mb,
            "memory_idle_multiplier": self._idle_memory_multiplier(),
            "upgrades_granted": self.upgrades_granted,
            "upgrades_denied": self.upgrades_denied,
        }
