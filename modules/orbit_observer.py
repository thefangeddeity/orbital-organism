from __future__ import annotations

import math
from typing import Any

from lego import Lego, ModuleResult


class OrbitObserver(Lego):

    name = "orbit_observer"
    version = "0.1"
    description = (
        "Observes the orbital simulator and exposes "
        "positions, velocities, energy and geometry."
    )

    adjustable = {
        "simulation_days_per_second": 2.0,
        "trail_length": 1000,
        "display": True,
    }

    def __init__(self):
        self.sim = None
        self.time_seconds = 0.0
        self.parameters = dict(self.adjustable)
        self.history: dict[str, list[tuple[float, float]]] = {}

    def configure(self, parameters: dict[str, Any]) -> None:

        self.parameters.update(parameters)

    def attach_simulator(self, simulator) -> None:

        self.sim = simulator

        for body in simulator.PLANETS:
            self.history.setdefault(
                body.name,
                [],
            )

    def observe(self, world) -> ModuleResult:

        if self.sim is None:
            raise RuntimeError(
                "OrbitObserver has no simulator."
            )

        states = {}

        for body in self.sim.PLANETS:


            orbit = self.sim.planet_orbit(body)

            position, velocity = (
                orbit.state_at_time(
                    self.time_seconds
                )
            )

            radius_km = math.hypot(
                position[0],
                position[1],
            )

            speed = math.hypot(
                velocity[0],
                velocity[1],
            )

            energy = (
                0.5 * speed * speed
                - self.sim.SUN.gm / radius_km
            )

            x = position[0] / self.sim.AU_KM
            y = position[1] / self.sim.AU_KM

            states[body.name] = {
                "x_au": x,
                "y_au": y,
                "radius_au": radius_km / self.sim.AU_KM,
                "speed_km_s": speed,
                "energy": energy,
            }

            history = self.history[body.name]

            history.append((x, y))

            limit = int(
                self.parameters["trail_length"]
            )

            if len(history) > limit:
                del history[:-limit]

        return ModuleResult(
            observations={
                "simulation_time_seconds":
                    self.time_seconds,
                "bodies": states,
            },

            metrics={
                "body_count": float(len(states)),
            },
        )

    def advance(self, seconds: float) -> None:

        self.time_seconds += seconds

    def capabilities(self) -> list[str]:

        return [
            "observe_orbital_state",
            "observe_position",
            "observe_velocity",
            "observe_energy",
            "maintain_trajectory_history",
        ]

    def state(self) -> dict[str, Any]:

        return {
            "simulation_time_seconds":
                self.time_seconds,

            "parameters":
                dict(self.parameters),
        }

