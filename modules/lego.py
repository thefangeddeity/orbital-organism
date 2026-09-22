from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResourceBudget:
    cpu_seconds: float = 1.0
    memory_mb: float = 128.0
    wall_seconds: float = 1.0


@dataclass
class ModuleResult:
    observations: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)


class Lego(ABC):
    """
    Base interface for every organism capability.

    A Lego is deliberately small and replaceable.
    The organism talks to Legos through this interface
    instead of knowing their implementation details.
    """

    name: str = "unnamed"
    version: str = "0.1"
    description: str = ""

    adjustable: dict[str, Any] = {}
    budget: ResourceBudget = ResourceBudget()

    @abstractmethod
    def configure(self, parameters: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def observe(self, world: Any) -> ModuleResult:
        pass

    def capabilities(self) -> list[str]:
        return []

    def state(self) -> dict[str, Any]:
        return {}

    def shutdown(self) -> None:
        pass
