from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class ComputeProvider(ABC):
    """
    Replaceable execution provider.

    The organism only knows this interface.

    Today:
        local CPU

    Future:
        Tanzania
        Tina
        another workstation
        a local inference server
        a neural-network accelerator
    """

    name = "unnamed"

    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def execute(
        self,
        function: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError


class LocalCPUProvider(ComputeProvider):

    name = "local-cpu"

    def available(self) -> bool:
        return True

    def execute(
        self,
        function,
        *args,
        **kwargs,
    ):
        return function(
            *args,
            **kwargs,
        )


class RemoteProvider(ComputeProvider):
    """
    Placeholder for future remote workers.

    No network protocol is assumed yet.

    A future Lego can implement execute() using SSH,
    a local HTTP service, a Unix socket, or another protocol
    without changing the learning engine.
    """

    def __init__(
        self,
        name: str,
        address: str | None = None,
    ):
        self.name = name
        self.address = address

    def available(self) -> bool:
        return False

    def execute(
        self,
        function,
        *args,
        **kwargs,
    ):
        raise RuntimeError(
            f"Remote provider '{self.name}' is not enabled."
        )
