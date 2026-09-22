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
    Placeholder for future remote workers (Tanzania, Tina, Ariana).

    No network protocol is assumed yet.

    A future Lego can implement execute() using SSH,
    a local HTTP service, a Unix socket, or another protocol
    without changing the learning engine.

    available() deliberately returns False until a real reachability
    check (ping, SSH handshake, HTTP health check) is implemented.
    Until then, dispatch() always falls through to local-cpu -- the
    organism runs correctly with zero remote workers online.
    """

    def __init__(
        self,
        name: str,
        address: str | None = None,
        role: str = "",
        intermittent: bool = False,
        enabled_in_config: bool = False,
    ):
        self.name = name
        self.address = address
        self.role = role
        self.intermittent = intermittent
        self.enabled_in_config = enabled_in_config

    def available(self) -> bool:
        # TODO: real reachability check once a transport is chosen.
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


def build_providers(config: dict) -> dict[str, ComputeProvider]:
    """
    Construct every configured provider from organism.json's
    "providers" section. Always includes local-cpu.
    """
    providers: dict[str, ComputeProvider] = {
        "local-cpu": LocalCPUProvider(),
    }

    provider_config = config.get("providers", {})

    for name in ("tanzania", "tina", "ariana"):
        entry = provider_config.get(name, {})

        providers[name] = RemoteProvider(
            name=name,
            address=entry.get("address"),
            role=entry.get("role", ""),
            intermittent=bool(entry.get("intermittent", False)),
            enabled_in_config=bool(entry.get("enabled", False)),
        )

    return providers


def dispatch(
    providers: dict[str, ComputeProvider],
    order: list[str],
    function,
    *args,
    **kwargs,
):
    """
    Try providers in priority order, falling back to local-cpu.

    Returns (result, provider_name_used). A remote provider that
    raises mid-execution is treated the same as one that was never
    available -- the call simply falls through to the next candidate,
    ending at local-cpu, which cannot fail this way.
    """
    for name in order:
        provider = providers.get(name)

        if provider is None or not provider.available():
            continue

        try:
            return provider.execute(function, *args, **kwargs), name
        except Exception:
            continue

    return providers["local-cpu"].execute(function, *args, **kwargs), "local-cpu"
