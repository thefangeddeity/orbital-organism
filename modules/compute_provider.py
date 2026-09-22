from __future__ import annotations

import subprocess
import threading
import time
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
    Remote worker reachable over SSH (Tanzania, Tina, Ariana -- host
    aliases already configured in ~/.ssh/config, key-based, no
    password).

    available() runs a real reachability check ('ssh <name> echo ok'),
    but never inline: this is called from organism.py's dashboard
    render every frame, and a blocking SSH round-trip there would
    freeze the renderer. Instead the check runs in a background daemon
    thread on a cooldown, and available() always returns instantly
    from a cached result -- correct-eventually, never blocking.

    execute() deliberately stays unimplemented for the live path.
    Real dispatch happens through tools/dispatch_tanzania.py, run
    out-of-band from the render loop entirely (SSH round-trips are
    seconds, not milliseconds -- there's no way to fit that inside a
    frame budget). This class's job is only to answer "is it up" for
    the dashboard and for dispatch()'s fallback-ordering logic, not to
    carry live work itself.
    """

    CHECK_INTERVAL_SECONDS = 20.0
    CHECK_TIMEOUT_SECONDS = 6.0

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

        self._cached_available = False
        self._check_in_progress = False
        self._last_check_time = 0.0
        self._lock = threading.Lock()

    def available(self) -> bool:
        if not self.enabled_in_config:
            # Not a declared fleet member -- never probed, never up.
            return False

        now = time.monotonic()

        with self._lock:
            due = (now - self._last_check_time) >= self.CHECK_INTERVAL_SECONDS
            in_progress = self._check_in_progress

            if due and not in_progress:
                self._check_in_progress = True
                should_start = True
            else:
                should_start = False

        if should_start:
            threading.Thread(
                target=self._check_reachability,
                daemon=True,
            ).start()

        return self._cached_available

    def _check_reachability(self) -> None:
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o", f"ConnectTimeout={int(self.CHECK_TIMEOUT_SECONDS)}",
                    "-o", "BatchMode=yes",
                    self.name,
                    "echo ok",
                ],
                capture_output=True,
                text=True,
                timeout=self.CHECK_TIMEOUT_SECONDS + 2.0,
            )
            reachable = result.returncode == 0 and "ok" in result.stdout
        except Exception:
            reachable = False

        with self._lock:
            self._cached_available = reachable
            self._last_check_time = time.monotonic()
            self._check_in_progress = False

    def execute(
        self,
        function,
        *args,
        **kwargs,
    ):
        raise RuntimeError(
            f"Remote provider '{self.name}' has no live execute() path -- "
            f"use tools/dispatch_tanzania.py for real dispatch, out-of-band "
            f"from the render loop."
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
