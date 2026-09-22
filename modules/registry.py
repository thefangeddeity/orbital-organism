from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Type

from lego import Lego


class ModuleRegistry:
    """
    Dynamic Lego registry.

    Modules are discovered from the modules directory.
    The registry does not need to know their names in advance.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.types: dict[str, Type[Lego]] = {}
        self.instances: dict[str, Lego] = {}

    def discover(self) -> list[str]:
        self.types.clear()

        for path in sorted(self.directory.glob("*.py")):

            if path.name.startswith("_"):
                continue

            if path.name in {
                "lego.py",
                "registry.py",
            }:
                continue

            module_name = f"organism_dynamic_{path.stem}"

            spec = importlib.util.spec_from_file_location(
                module_name,
                path,
            )

            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)

            spec.loader.exec_module(module)

            for value in vars(module).values():

                if (
                    isinstance(value, type)
                    and issubclass(value, Lego)
                    and value is not Lego
                ):
                    self.types[value.name] = value

        return sorted(self.types)

    def activate(
        self,
        name: str,
        parameters: dict | None = None,
    ) -> Lego:

        if name not in self.types:
            raise KeyError(
                f"Unknown Lego: {name}"
            )

        lego = self.types[name]()

        lego.configure(parameters or {})

        self.instances[name] = lego

        return lego

    def deactivate(self, name: str) -> None:

        lego = self.instances.pop(name, None)

        if lego is not None:
            lego.shutdown()

    def active(self) -> dict[str, Lego]:
        return dict(self.instances)

