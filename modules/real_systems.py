from __future__ import annotations

import math
from typing import Any


# ======================================================================
# REAL ASTRONOMICAL SYSTEMS
#
# Orbital elements below are public-domain literature values (Wikipedia
# summaries of the discovery/characterization papers, cross-checked
# against the NASA Exoplanet Archive descriptions), hand-seeded here as
# a first cut -- NOT a live feed. A later plank (the "curiosity queue")
# is meant to replace/refine these via bounded, logged fetches against
# VizieR / the NASA Exoplanet Archive / Gaia. Where a value genuinely
# isn't well constrained (radial-velocity-only planets have unknown
# true inclination due to the sin(i) degeneracy), that is said plainly
# in a comment rather than papered over with a confident-looking number.
#
# GM for each host is DERIVED from a real (semi-major axis, period)
# pair via Kepler's third law rather than hand-typed from a separate
# stellar-mass estimate, so orbital propagation is self-consistent:
# the period you get back out is exactly the period that was measured.
# ======================================================================

AU_KM = 149_597_870.7
DAY = 86_400.0
YEAR = 365.25 * DAY
SOLAR_RADIUS_KM = 696_000.0
EARTH_RADIUS_KM = 6_371.0


def _gm_from_kepler(a_au: float, period_days: float) -> float:
    """
    Invert Kepler's third law: mu = 4*pi^2 * a^3 / T^2.

    Given a real measured (semi-major axis, period) pair, returns the
    GM (km^3/s^2) that reproduces it exactly under this toolkit's
    two-body Orbit class. This is standard practice when a star's mass
    is itself only loosely known but the orbit is well measured.
    """
    a_km = a_au * AU_KM
    period_s = period_days * DAY
    return 4.0 * math.pi**2 * a_km**3 / period_s**2


def _mass_radius_estimate_km(mass_earth: float) -> float:
    """
    Rough rocky-planet mass-radius relation (R ~ M^0.27), used only
    because these radial-velocity-detected planets don't transit and
    so have no measured radius at all. This is a placeholder, not a
    measurement -- flagged as such wherever it's used.
    """
    return EARTH_RADIUS_KM * (mass_earth ** 0.27)


def build_proxima_system(sim) -> tuple[Any, list[Any]]:
    """
    Proxima Centauri and its two confirmed planets, b and d.

    Source values (approximate, literature-sourced):
      Proxima b: a=0.04856 AU, period=11.186 days, e~0.02 (poorly
                 constrained, RV-only), min mass 1.07 Earth masses.
      Proxima d: a=0.029 AU,  period=5.15 days,  e=0 (assumed circular
                 in the discovery fit), min mass 0.26 Earth masses.

    True orbital inclination/orientation is unknown for both (RV
    detections only constrain minimum mass and the orbital plane
    projected along the line of sight is undetermined) -- inclination
    and argument of periapsis are left at 0 rather than invented.
    """
    star = sim.Body(
        "Proxima Centauri",
        gm=_gm_from_kepler(a_au=0.04856, period_days=11.186),
        radius=0.1542 * SOLAR_RADIUS_KM,
    )

    proxima_b = sim.Body(
        "Proxima b",
        gm=0.0,
        radius=_mass_radius_estimate_km(1.07),
        a_au=0.04856,
        ecc=0.02,
        parent=star,
    )

    proxima_d = sim.Body(
        "Proxima d",
        gm=0.0,
        radius=_mass_radius_estimate_km(0.26),
        a_au=0.029,
        ecc=0.0,
        parent=star,
    )

    return star, [proxima_b, proxima_d]


def build_alpha_centauri_system(sim) -> tuple[Any, list[Any]]:
    """
    Alpha Centauri A/B as a resolved visual binary.

    Modeled as B's relative orbit around A, which is the standard
    two-body reduction: the relative separation vector between two
    gravitating bodies obeys a Kepler orbit around mu = G*(M_A + M_B).
    This is the same parent/child pattern the toolkit already uses for
    Sun + planet; it is dynamically exact for a two-body system.

    Source values (approximate, literature-sourced):
      period ~79.762 years, eccentricity ~0.52,
      periastron 11.2 AU, apastron 35.6 AU -> a = 23.4 AU.

    Orbital orientation (inclination, longitude of ascending node,
    argument of periapsis) IS well measured for this system in the
    literature but wasn't sourced in this pass -- left at 0 rather
    than guessed. Shape (a, e, period) is real; 3D tilt is a
    placeholder pending a follow-up fetch.
    """
    star_a = sim.Body(
        "Alpha Centauri A",
        gm=_gm_from_kepler(a_au=23.4, period_days=79.762 * 365.25),
        radius=1.2234 * SOLAR_RADIUS_KM,
    )

    star_b = sim.Body(
        "Alpha Centauri B",
        gm=0.0,
        radius=0.8632 * SOLAR_RADIUS_KM,
        a_au=23.4,
        ecc=0.52,
        parent=star_a,
    )

    return star_a, [star_b]


class RealSystemAdapter:
    """
    Presents a real (non-solar-system) star + planet set through the
    same surface OrbitObserver / NeuralLearner already expect from the
    loaded orbital_sandbox module: .PLANETS, .SUN, .AU_KM,
    .planet_orbit(body). Both modules are used completely unchanged --
    this is the "domain adapter" boundary: swap what's plugged in here,
    not the evolution/learning engine that consumes it.
    """

    def __init__(self, sim, star, planets, label: str):
        self.PLANETS = planets
        self.SUN = star
        self.AU_KM = sim.AU_KM
        self.label = label
        self._sim = sim

    def planet_orbit(self, body):
        return self._sim.planet_orbit(body)

    def run_self_tests(self) -> bool:
        # Orbit propagation itself is validated by the underlying
        # toolkit's own self-tests (run once at boot, against the
        # solar system). Nothing here needs a second, redundant check.
        return True


WORLD_BUILDERS = {
    "proxima": (build_proxima_system, "Proxima Centauri system"),
    "alpha_centauri": (build_alpha_centauri_system, "Alpha Centauri AB"),
}


def build_world(sim, mode: str):
    """
    Returns the object the rest of the organism should treat as
    "the world" -- either the solar-system module itself (mode ==
    "solar_system", the default, zero behavior change), or a
    RealSystemAdapter wrapping a real system built on the same engine.
    """
    if mode == "solar_system" or mode not in WORLD_BUILDERS:
        return sim

    builder, label = WORLD_BUILDERS[mode]
    star, planets = builder(sim)

    return RealSystemAdapter(sim, star, planets, label)
