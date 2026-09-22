"""Gun geometry, charge, elevation, and the interior-ballistics launch solver.

The reference gun is a Gribeauval-pattern 12-pounder firing solid cast-iron
shot.

While the shot is in the bore it is constrained to the barrel axis: the gas
force acts along that axis, and gravity contributes only its axial
component. The transverse component is carried by the bore wall and does not
move the shot.

Barrel transit is about 5 ms, so the bore phase is integrated at a fixed
`BORE_DT_S` = 1e-6 s rather than on any frame clock. This runs once per shot
and its cost is irrelevant.

GUN GEOMETRY, AND A DEVIATION FROM THE BRIEF
--------------------------------------------
The brief puts the origin at "the muzzle-end ground point". Taken literally
that fixes the muzzle in space and swings the breech, which drives the
breech below ground at any real elevation: a 2.00 m bore with its muzzle
0.95 m up puts the breech face 0.46 m underground at 45 degrees. The piece
cannot be drawn, because the geometry is not physical.

So the pivot is modelled where the real one is - the trunnions - and the
origin is the ground point below them:

    trunnion  = (0, TRUNNION_HEIGHT_M)
    muzzle    = trunnion + axis * TRUNNION_TO_MUZZLE_M

`TRUNNION_HEIGHT_M` = 0.95 m is the bore-axis height of a Gribeauval
12-pounder on its field carriage, so the muzzle stands 0.95 m up at zero
elevation, which is the figure the Tech Lead specified. It is an approximate
number, not a measured one: period drawings put the bore axis a little under
a metre above ground, over an axle carried on roughly 1.5 m wheels. Good to
about +/- 0.1 m.

`TRUNNION_TO_MUZZLE_M` = 1.15 m places the trunnions about 43% of the bore
length back from the muzzle, which is roughly where they sit on the real
piece and, not coincidentally, is what keeps the breech face above ground
across the full 0-85 degree elevation range (0.10 m clearance at 85
degrees). It is chosen, not measured.

The launch point therefore moves by up to 1.15 m as the piece is elevated.
That is a real effect, it is correctly signed, and it is negligible against
kilometre ranges - but it means the shot does not start exactly at the
origin, and downrange distance is measured from below the trunnions rather
than from below the muzzle.
"""

import math
from dataclasses import dataclass

from .ballistics import (
    CHARGE_REFERENCE_KG,
    P0_REFERENCE_PA,
    bore_force_n,
    bore_pressure_pa,
    chamber_pressure_pa,
    chamber_volume_m3,
)
from .frames import surface_velocity_eci
from .physics import ShotState, Vec2
from .planet import (
    OMEGA_EARTH_RAD_S,
    gravitational_acceleration,
    local_up,
    surface_origin,
)

#: Fixed integration step through the bore, s.
BORE_DT_S = 1e-6

#: Calibration target for the reference charge, m/s.
TARGET_MUZZLE_VELOCITY_MS = 440.0

#: Bore-axis height at the trunnions on a field carriage, m. Approximate;
#: see the module docstring. Equals muzzle height at zero elevation.
TRUNNION_HEIGHT_M = 0.95

#: Distance from the trunnion axis forward to the muzzle face, m. Chosen;
#: see the module docstring.
TRUNNION_TO_MUZZLE_M = 1.15


@dataclass(frozen=True, slots=True)
class Gun:
    name: str
    bore_length_m: float
    bore_diameter_m: float
    trunnion_height_m: float = TRUNNION_HEIGHT_M
    trunnion_to_muzzle_m: float = TRUNNION_TO_MUZZLE_M
    p0_reference_pa: float = P0_REFERENCE_PA

    @property
    def bore_area_m2(self):
        return 0.25 * math.pi * self.bore_diameter_m * self.bore_diameter_m

    @property
    def trunnion_m(self):
        """Pivot point, in PLANET-CENTRED coordinates.

        The gun stands on the surface at (0, R_EARTH + trunnion height), not
        at the origin - the origin is now the Earth's centre. Under the
        flat-earth fixture `surface_origin` returns (0, 0) and this reduces
        to the pre-batch-8 geometry exactly.
        """
        return surface_origin() + Vec2(0.0, self.trunnion_height_m)

    def axis(self, elevation_rad):
        """Unit vector along the bore, in the LOCAL FRAME AT THE GUN.

        Built from the local up at the trunnions and the local horizontal
        perpendicular to it, NOT from the world axes. Those two coincide
        only because this gun happens to stand at downrange zero, where
        local up is (0, 1) - so the world-axis form was right by accident
        and would break silently the moment a piece was emplaced anywhere
        else, or the frame re-anchored. Elevation means angle above the
        local horizontal at the muzzle, and this is where that is made true
        rather than assumed.
        """
        up = local_up(self.trunnion_m)
        downrange = up.perp() * -1.0
        return downrange * math.cos(elevation_rad) + up * math.sin(elevation_rad)

    def muzzle_m(self, elevation_rad):
        """Muzzle face position at a given elevation."""
        return self.trunnion_m + self.axis(elevation_rad) * self.trunnion_to_muzzle_m

    def breech_m(self, elevation_rad):
        """Breech face of the bore, i.e. the muzzle less one bore length."""
        return self.muzzle_m(elevation_rad) - self.axis(elevation_rad) * self.bore_length_m


@dataclass(frozen=True, slots=True)
class Shot:
    name: str
    mass_kg: float
    diameter_m: float

    @property
    def radius_m(self):
        return 0.5 * self.diameter_m


@dataclass(frozen=True, slots=True)
class BoreResult:
    """What happened between ignition and the muzzle."""

    muzzle_speed_ms: float
    transit_time_s: float
    peak_pressure_pa: float
    muzzle_pressure_pa: float


GRIBEAUVAL_12PDR = Gun(
    name="Gribeauval 12-pounder",
    bore_length_m=2.00,
    bore_diameter_m=0.115,
)

IRON_SHOT_12PDR = Shot(
    name="12 lb cast-iron solid shot",
    mass_kg=5.44,
    diameter_m=0.115,
)


def solve_bore(gun, shot, charge_kg, elevation_rad):
    """Integrate the shot from ignition to the muzzle. Returns a BoreResult.

    One-dimensional along the barrel axis, fixed-step RK4 at BORE_DT_S, with
    the final step shortened so the shot stops exactly at the muzzle rather
    than overshooting it.
    """
    area_m2 = gun.bore_area_m2
    length_m = gun.bore_length_m
    volume_0_m3 = chamber_volume_m3(charge_kg)
    pressure_0_pa = chamber_pressure_pa(charge_kg, gun.p0_reference_pa)

    # Local gravity at the gun, not the flat-earth constant: mu/R^2 is
    # 9.82025 against standard gravity's 9.80665, a 0.14% difference. It is
    # worth about 0.01 J against 527 kJ of muzzle energy here, so it changes
    # nothing measurable - but leaving a flat constant in the bore solver is
    # how a flat-earth assumption survives a frame change.
    gravity_axial_ms2 = -gravitational_acceleration(
        gun.trunnion_m
    ).length() * math.sin(elevation_rad)
    inverse_mass = 1.0 / shot.mass_kg

    def accel_ms2(travel_m):
        gas_n = bore_force_n(travel_m, area_m2, volume_0_m3, pressure_0_pa)
        return gas_n * inverse_mass + gravity_axial_ms2

    travel_m = 0.0
    speed_ms = 0.0
    time_s = 0.0

    while travel_m < length_m:
        new_travel_m, new_speed_ms = _rk4_axial(
            travel_m, speed_ms, BORE_DT_S, accel_ms2
        )
        if new_travel_m >= length_m:
            exit_dt_s = _bisect_bore_exit(
                travel_m, speed_ms, BORE_DT_S, accel_ms2, length_m
            )
            travel_m, speed_ms = _rk4_axial(
                travel_m, speed_ms, exit_dt_s, accel_ms2
            )
            time_s += exit_dt_s
            break
        travel_m, speed_ms = new_travel_m, new_speed_ms
        time_s += BORE_DT_S

        if speed_ms <= 0.0 and travel_m <= 0.0:
            raise RuntimeError(
                f"charge of {charge_kg} kg cannot drive the shot out of the "
                f"bore at {math.degrees(elevation_rad):.1f} degrees elevation"
            )

    return BoreResult(
        muzzle_speed_ms=speed_ms,
        transit_time_s=time_s,
        # Pressure is monotonically decreasing in travel, so the peak is at
        # ignition and the minimum is at the muzzle.
        peak_pressure_pa=pressure_0_pa,
        muzzle_pressure_pa=bore_pressure_pa(
            length_m, area_m2, volume_0_m3, pressure_0_pa
        ),
    )


def fire(
    gun,
    shot,
    charge_kg=CHARGE_REFERENCE_KG,
    elevation_rad=math.radians(45.0),
    spin_rads=0.0,
):
    """Fire the gun. Returns (initial free-flight ShotState, BoreResult).

    THE MUZZLE VELOCITY IS RELATIVE TO THE GUN, and the gun is fixed to the
    ground, so the shot's inertial velocity is the muzzle velocity plus the
    ground's own motion. That is the first of the two ways rotation reaches a
    trajectory - the other being drag against the co-rotating air. Neither is a
    fictitious force; see `frames`.

    At OMEGA_EARTH_RAD_S = 0 the added term is exactly zero, so this is
    bit-identical to the muzzle velocity alone.
    """
    bore = solve_bore(gun, shot, charge_kg, elevation_rad)
    muzzle_m = gun.muzzle_m(elevation_rad)
    return (
        ShotState(
            pos=muzzle_m,
            vel=(
                gun.axis(elevation_rad) * bore.muzzle_speed_ms
                + surface_velocity_eci(muzzle_m, OMEGA_EARTH_RAD_S)
            ),
            spin=spin_rads,
            mass=shot.mass_kg,
            radius=shot.radius_m,
        ),
        bore,
    )


def _rk4_axial(x_m, v_ms, dt_s, accel_ms2):
    """One RK4 step of the 1D bore problem. Acceleration depends only on x."""
    half_s = 0.5 * dt_s

    a1 = accel_ms2(x_m)
    x2 = x_m + v_ms * half_s
    v2 = v_ms + a1 * half_s

    a2 = accel_ms2(x2)
    x3 = x_m + v2 * half_s
    v3 = v_ms + a2 * half_s

    a3 = accel_ms2(x3)
    x4 = x_m + v3 * dt_s
    v4 = v_ms + a3 * dt_s

    a4 = accel_ms2(x4)

    sixth = dt_s / 6.0
    return (
        x_m + (v_ms + 2.0 * v2 + 2.0 * v3 + v4) * sixth,
        v_ms + (a1 + 2.0 * a2 + 2.0 * a3 + a4) * sixth,
    )


def _bisect_bore_exit(x_m, v_ms, dt_s, accel_ms2, length_m, iterations=60):
    """Find the sub-step duration that lands the shot exactly at the muzzle."""
    low_s = 0.0
    high_s = dt_s
    for _ in range(iterations):
        mid_s = 0.5 * (low_s + high_s)
        mid_x_m, _ = _rk4_axial(x_m, v_ms, mid_s, accel_ms2)
        if mid_x_m >= length_m:
            high_s = mid_s
        else:
            low_s = mid_s
    return 0.5 * (low_s + high_s)
