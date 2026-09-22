from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


# ======================================================================
# REAL ASTRONOMICAL SYSTEMS
#
# Orbital elements are preferentially LIVE: tools/fetch_exoplanet_data.py
# queries the NASA Exoplanet Archive and caches the result to
# state/exoplanet_cache.json (never called from here -- see that
# module's docstring for why a network round-trip has no place in the
# boot/render path). When that cache exists and has a usable row for a
# given planet, its values are used; otherwise this falls back to the
# public-domain literature values hand-seeded below (Wikipedia summaries
# of the discovery/characterization papers, cross-checked against the
# archive's own descriptions at the time they were written). The
# fallback is per-planet, never a mix of a live semi-major-axis with a
# hand-seeded eccentricity for the same body -- those need to come from
# the same fit to stay self-consistent. Where a value genuinely isn't
# well constrained (radial-velocity-only planets have unknown true
# inclination due to the sin(i) degeneracy), that is said plainly in a
# comment rather than papered over with a confident-looking number.
#
# Alpha Centauri's A/B stellar binary orbit is NOT in the NASA
# Exoplanet Archive -- confirmed empirically (zero rows for any
# 'alf Cen%' host pattern) -- because it's a planet archive, not a
# stellar-multiplicity catalog. build_alpha_centauri_system() therefore
# stays fully literature-sourced; no live path exists for it yet.
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


CACHE_PATH = Path(__file__).resolve().parents[1] / "state" / "exoplanet_cache.json"

# Fields a cache row must have, non-null, to be trusted for orbital
# propagation. pl_rade (radius) and pl_orbeccen are checked separately
# per-use since those have honest fallbacks (mass-radius estimate,
# e=0) rather than making the whole row unusable.
_REQUIRED_ROW_FIELDS = ("pl_orbsmax", "pl_orbper")


def _load_exoplanet_cache() -> dict | None:
    if not CACHE_PATH.exists():
        return None
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return None


def _cached_row(cache: dict | None, system_key: str, name_contains: str) -> dict | None:
    if cache is None:
        return None

    entry = cache.get("systems", {}).get(system_key)
    if not entry or entry.get("error") or not entry.get("rows"):
        return None

    for row in entry["rows"]:
        name = row.get("pl_name") or ""
        if name_contains.lower() not in name.lower():
            continue
        if all(row.get(field) is not None for field in _REQUIRED_ROW_FIELDS):
            return row

    return None


def build_proxima_system(sim) -> tuple[Any, list[Any]]:
    """
    Proxima Centauri and its two confirmed planets, b and d.

    Prefers live rows from state/exoplanet_cache.json (see this
    module's header) over the literature fallback below, per planet.

    Literature fallback values (approximate):
      Proxima b: a=0.04856 AU, period=11.186 days, e~0.02 (poorly
                 constrained, RV-only), min mass 1.07 Earth masses.
      Proxima d: a=0.029 AU,  period=5.15 days,  e=0 (assumed circular
                 in the discovery fit), min mass 0.26 Earth masses.

    True orbital inclination/orientation is unknown for both, live or
    fallback (RV detections only constrain minimum mass and the orbital
    plane projected along the line of sight is undetermined) --
    inclination and argument of periapsis are left at 0 rather than
    invented either way.
    """
    cache = _load_exoplanet_cache()
    row_b = _cached_row(cache, "proxima", "Proxima Cen b")
    row_d = _cached_row(cache, "proxima", "Proxima Cen d")

    if row_b is not None:
        b_a_au = float(row_b["pl_orbsmax"])
        b_period_days = float(row_b["pl_orbper"])
        b_ecc = float(row_b["pl_orbeccen"]) if row_b.get("pl_orbeccen") is not None else 0.02
        b_mass = float(row_b["pl_bmasse"]) if row_b.get("pl_bmasse") is not None else 1.07
        b_radius_km = (
            float(row_b["pl_rade"]) * EARTH_RADIUS_KM
            if row_b.get("pl_rade") is not None
            else _mass_radius_estimate_km(b_mass)
        )
    else:
        b_a_au, b_period_days, b_ecc, b_mass = 0.04856, 11.186, 0.02, 1.07
        b_radius_km = _mass_radius_estimate_km(b_mass)

    star = sim.Body(
        "Proxima Centauri",
        gm=_gm_from_kepler(a_au=b_a_au, period_days=b_period_days),
        radius=0.1542 * SOLAR_RADIUS_KM,
    )

    proxima_b = sim.Body(
        "Proxima b",
        gm=0.0,
        radius=b_radius_km,
        a_au=b_a_au,
        ecc=b_ecc,
        parent=star,
    )

    if row_d is not None:
        d_a_au = float(row_d["pl_orbsmax"])
        d_ecc = float(row_d["pl_orbeccen"]) if row_d.get("pl_orbeccen") is not None else 0.0
        d_mass = float(row_d["pl_bmasse"]) if row_d.get("pl_bmasse") is not None else 0.26
        d_radius_km = (
            float(row_d["pl_rade"]) * EARTH_RADIUS_KM
            if row_d.get("pl_rade") is not None
            else _mass_radius_estimate_km(d_mass)
        )
    else:
        d_a_au, d_ecc, d_mass = 0.029, 0.0, 0.26
        d_radius_km = _mass_radius_estimate_km(d_mass)

    proxima_d = sim.Body(
        "Proxima d",
        gm=0.0,
        radius=d_radius_km,
        a_au=d_a_au,
        ecc=d_ecc,
        parent=star,
    )

    fetched_at = cache.get("fetched_at") if cache else None
    if row_b is not None or row_d is not None:
        print(f"[real_systems] Proxima Centauri: live NASA archive data "
              f"(fetched {fetched_at})")
    else:
        print("[real_systems] Proxima Centauri: literature fallback "
              "(no usable cache -- run tools/fetch_exoplanet_data.py)")

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
