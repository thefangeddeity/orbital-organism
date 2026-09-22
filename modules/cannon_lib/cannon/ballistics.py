"""Interior and exterior ballistics models. Pure scalar physics, SI units.

This module holds no state and imports nothing from the rest of the package.
Everything here is a function of its arguments.

Interior ballistics
-------------------
The propellant is modelled as an ideal gas that burns instantaneously at
ignition and then expands adiabatically behind the shot:

    V(x) = V0 + A*x
    p(x) = p0 * (V0 / V(x))**GAMMA
    F(x) = p(x) * A                for 0 <= x < L

`P0_REFERENCE_PA` IS A FITTED LUMP PARAMETER, NOT A DERIVED ONE.
It was obtained by solving for the chamber pressure that produces the
historically observed 440 m/s muzzle velocity with the reference charge,
using `calibrate_p0` in this module. It is absorbing, in a single number:
the finite burn rate, heat loss to the barrel wall, gas leakage past the
windage gap, and the Lagrange pressure gradient between the breech face and
the shot base. It is not a chamber pressure that anyone measured.

Do not replace it with a first-principles Noble-Abel instantaneous-burn
figure and then trust the result: with black-powder impetus f ~ 300 kJ/kg,
that calculation overpredicts muzzle velocity by roughly a factor of two,
because real black powder delivers only ~10% of its chemical energy to shot
kinetic energy. The fitted p0 encodes that loss. A Vieille burn-rate model
with a grain form function is the stage-6 replacement for this whole
section, at which point p0 stops being fitted and starts being computed.

Charge scaling is linear in both chamber volume and peak pressure. This is
crude, but it is monotonic and correctly signed, which is all stage 1 needs.

Exterior ballistics
-------------------
US Standard Atmosphere 1976, layered to 86 km with an exponential tail above
it, and a Mach-dependent drag coefficient for a sphere.

The atmosphere is a function of ALTITUDE, which the caller computes; this
module does not know about the planet frame. That is why it can still be
imported by anything without pulling in a coordinate system.

KNOWN UNCERTAINTY IN Cd: at the Reynolds numbers this shot flies at
(Re ~ 3.5e6), a *smooth* sphere has passed through the drag crisis and sits
near Cd = 0.20. The value used here for the incompressible limit,
`CD_SUBSONIC` = 0.47, is the pre-crisis / tripped-boundary-layer figure. The
justification is that a cast-iron ball carries a mould seam and a rough
as-cast surface, which trips the boundary layer, and that historical range
tables are consistent with the higher number. This is a defensible choice,
not a measured one. `CD_SUBSONIC` is a named constant so it can be swept.
"""

import math
from bisect import bisect_right

# --------------------------------------------------------------------------
# Interior ballistics
# --------------------------------------------------------------------------

GAMMA = 1.2  # ratio of specific heats for black-powder combustion gas

#: Chamber volume for the reference charge, m^3.
#: 1.8 kg of black powder at a loaded density of ~900 kg/m^3.
CHAMBER_VOLUME_REFERENCE_M3 = 2.0e-3

#: Charge mass the chamber volume and p0 above are quoted for, kg.
CHARGE_REFERENCE_KG = 1.8

#: FITTED. See module docstring. Solved by `calibrate_p0` for a 440 m/s
#: muzzle velocity with the reference charge, a 5.44 kg shot, a 0.01039 m^2
#: bore area and a 2.00 m bore.
P0_REFERENCE_PA = 1.366986e8


def chamber_volume_m3(charge_kg):
    """Initial gas volume behind the shot for a given charge mass, m^3."""
    return CHAMBER_VOLUME_REFERENCE_M3 * (charge_kg / CHARGE_REFERENCE_KG)


def chamber_pressure_pa(charge_kg, p0_reference_pa=P0_REFERENCE_PA):
    """Ignition pressure for a given charge mass, Pa. Linear in charge."""
    return p0_reference_pa * (charge_kg / CHARGE_REFERENCE_KG)


def bore_pressure_pa(travel_m, bore_area_m2, volume_0_m3, pressure_0_pa):
    """Gas pressure behind the shot after `travel_m` of bore travel, Pa."""
    volume_m3 = volume_0_m3 + bore_area_m2 * travel_m
    return pressure_0_pa * (volume_0_m3 / volume_m3) ** GAMMA


def bore_force_n(travel_m, bore_area_m2, volume_0_m3, pressure_0_pa):
    """Axial force the gas exerts on the shot base, N."""
    return bore_area_m2 * bore_pressure_pa(
        travel_m, bore_area_m2, volume_0_m3, pressure_0_pa
    )


def bore_work_j(bore_length_m, bore_area_m2, volume_0_m3, pressure_0_pa):
    """Closed-form work done by the gas over the full bore, J.

    Analytic antiderivative of `bore_force_n`. Used by `calibrate_p0` to get
    a first guess; the substepped integration in `gun.fire` is what actually
    decides muzzle velocity.
    """
    volume_muzzle_m3 = volume_0_m3 + bore_area_m2 * bore_length_m
    exponent = 1.0 - GAMMA
    return (
        pressure_0_pa
        * volume_0_m3**GAMMA
        / exponent
        * (volume_muzzle_m3**exponent - volume_0_m3**exponent)
    )


def calibrate_p0(
    shot_mass_kg,
    bore_length_m,
    bore_area_m2,
    volume_0_m3,
    target_muzzle_velocity_ms,
):
    """Solve for the ignition pressure giving a target muzzle velocity, Pa.

    Exact rather than iterative: the work integral is linear in p0, so the
    required pressure follows directly from the drag-free, gravity-free
    work-energy balance 0.5*m*v^2 = W(p0).
    """
    target_energy_j = 0.5 * shot_mass_kg * target_muzzle_velocity_ms**2
    work_per_pascal = bore_work_j(bore_length_m, bore_area_m2, volume_0_m3, 1.0)
    return target_energy_j / work_per_pascal


# --------------------------------------------------------------------------
# Atmosphere: US Standard 1976, layered to 86 km, exponential above
# --------------------------------------------------------------------------

#: US Standard Atmosphere 1976. Replaces the ISA troposphere model, whose
#: 11 km ceiling was insufficient: the rocket rungs spend most of their
#: ascent above it, and drag being silently wrong up there is exactly the
#: failure that would not be noticed until it mattered.
#:
#: Layers are defined on GEOPOTENTIAL altitude, which is what the standard
#: specifies - geopotential folds the variation of g with height into the
#: altitude coordinate so the hydrostatic integration can assume g is
#: constant. Callers pass GEOMETRIC altitude (|r| - R_EARTH) and the
#: conversion happens here; at 86 km the two differ by about 1.1 km, so
#: treating them as the same would be a real error rather than a rounding
#: one.
SEA_LEVEL_TEMPERATURE_K = 288.15
SEA_LEVEL_PRESSURE_PA = 101325.0
SEA_LEVEL_DENSITY_KGM3 = 1.225

#: Standard gravity used for the hydrostatic integration, m/s^2. This is the
#: standard's own constant and is deliberately NOT mu/R^2 - the layer base
#: pressures below were derived with it, so substituting a different g would
#: make the table internally inconsistent.
_G0_MS2 = 9.80665

#: Effective Earth radius for the geopotential conversion, m. The standard's
#: value, not R_EARTH.
_R0_GEOPOTENTIAL_M = 6356766.0

#: Molar mass of air (kg/mol) and the universal gas constant (J/(mol K)),
#: both as the standard defines them.
_MOLAR_MASS_AIR = 0.0289644
_UNIVERSAL_GAS_R = 8.31432

#: Specific gas constant for air, J/(kg K).
R_SPECIFIC_AIR = _UNIVERSAL_GAS_R / _MOLAR_MASS_AIR

#: Ratio of specific heats for air. Used only for the speed of sound; the
#: black-powder gamma in the interior model is a different gas.
GAMMA_AIR = 1.4

#: Geopotential base altitude (m), base temperature (K), lapse rate (K/m).
#: Seven layers to 84.852 km geopotential, which is 86 km geometric.
_LAYER_BASE_H_M = (0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0)
_LAYER_BASE_T_K = (288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65)
_LAYER_LAPSE_K_PER_M = (-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002)

#: Top of the layered model, geopotential m.
_MODEL_TOP_H_M = 84852.0

#: Lowest altitude the model is evaluated at, m. Below this - which the shot
#: reaches only briefly while penetrating the ground before contact
#: resolves - values are CLAMPED to sea level rather than extrapolated, so a
#: transient sub-surface position cannot produce a nonsense density.
_ALTITUDE_FLOOR_M = 0.0


def _geopotential_m(geometric_altitude_m):
    return (
        _R0_GEOPOTENTIAL_M
        * geometric_altitude_m
        / (_R0_GEOPOTENTIAL_M + geometric_altitude_m)
    )


def _layer_base_pressures_pa():
    """Base pressure of each layer, integrated upward from sea level."""
    pressures = [SEA_LEVEL_PRESSURE_PA]
    for index in range(len(_LAYER_BASE_H_M) - 1):
        pressures.append(
            _pressure_within_layer(index, _LAYER_BASE_H_M[index + 1], pressures[index])
        )
    return tuple(pressures)


def _pressure_within_layer(index, geopotential_m, base_pressure_pa):
    base_h = _LAYER_BASE_H_M[index]
    base_t = _LAYER_BASE_T_K[index]
    lapse = _LAYER_LAPSE_K_PER_M[index]
    if lapse == 0.0:
        return base_pressure_pa * math.exp(
            -_G0_MS2 * (geopotential_m - base_h) / (R_SPECIFIC_AIR * base_t)
        )
    temperature = base_t + lapse * (geopotential_m - base_h)
    return base_pressure_pa * (base_t / temperature) ** (
        _G0_MS2 / (R_SPECIFIC_AIR * lapse)
    )


_LAYER_BASE_P_PA = _layer_base_pressures_pa()

#: Conditions at the top of the layered model, for the exponential tail.
_TOP_T_K = _LAYER_BASE_T_K[-1] + _LAYER_LAPSE_K_PER_M[-1] * (
    _MODEL_TOP_H_M - _LAYER_BASE_H_M[-1]
)
_TOP_P_PA = _pressure_within_layer(
    len(_LAYER_BASE_H_M) - 1, _MODEL_TOP_H_M, _LAYER_BASE_P_PA[-1]
)

#: Scale height for the exponential tail above the model top, m. Taken from
#: the conditions there, so density is continuous across the join. Drag is
#: negligible long before this altitude; what matters is that the function
#: stays finite, positive and continuous rather than being accurate.
_TAIL_SCALE_HEIGHT_M = R_SPECIFIC_AIR * _TOP_T_K / _G0_MS2


def _layer_index(geopotential_m):
    index = 0
    for candidate in range(len(_LAYER_BASE_H_M)):
        if geopotential_m >= _LAYER_BASE_H_M[candidate]:
            index = candidate
    return index


def temperature_k(altitude_m):
    """Air temperature, K. US Standard Atmosphere 1976."""
    geopotential_m = _geopotential_m(max(altitude_m, _ALTITUDE_FLOOR_M))
    if geopotential_m >= _MODEL_TOP_H_M:
        return _TOP_T_K
    index = _layer_index(geopotential_m)
    return _LAYER_BASE_T_K[index] + _LAYER_LAPSE_K_PER_M[index] * (
        geopotential_m - _LAYER_BASE_H_M[index]
    )


def pressure_pa(altitude_m):
    """Air pressure, Pa."""
    geopotential_m = _geopotential_m(max(altitude_m, _ALTITUDE_FLOOR_M))
    if geopotential_m >= _MODEL_TOP_H_M:
        return _TOP_P_PA * math.exp(
            -(geopotential_m - _MODEL_TOP_H_M) / _TAIL_SCALE_HEIGHT_M
        )
    index = _layer_index(geopotential_m)
    return _pressure_within_layer(index, geopotential_m, _LAYER_BASE_P_PA[index])


def density_kgm3(altitude_m):
    """Air density, kg/m^3. Always finite and strictly positive."""
    return pressure_pa(altitude_m) / (R_SPECIFIC_AIR * temperature_k(altitude_m))


def speed_of_sound_ms(altitude_m):
    """Speed of sound, m/s.

    Above the model top this is constant, because the tail holds temperature
    fixed. Mach ceases to mean much for drag purposes up there anyway, and
    `drag_coefficient` already clamps to its supersonic tail rather than
    letting M run away.
    """
    return math.sqrt(GAMMA_AIR * R_SPECIFIC_AIR * temperature_k(altitude_m))


# --------------------------------------------------------------------------
# Drag
# --------------------------------------------------------------------------

#: Incompressible-limit drag coefficient. See module docstring: this is the
#: rough-sphere / tripped-boundary-layer value, not the smooth-sphere
#: post-drag-crisis value of ~0.20. Sweep this to test that choice.
CD_SUBSONIC = 0.47

#: Mach number breakpoints for the Cd table.
_CD_MACH = (0.0, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0)

#: Drag coefficient at each breakpoint. The first entry is CD_SUBSONIC.
_CD_VALUES = (
    CD_SUBSONIC, 0.48, 0.55, 0.72, 0.90, 0.98, 1.00, 0.98, 0.92, 0.88, 0.85,
)


def drag_coefficient(mach):
    """Sphere drag coefficient by linear interpolation over the Cd table.

    Clamped to the table endpoints below M=0 and above M=3.
    """
    if mach <= _CD_MACH[0]:
        return _CD_VALUES[0]
    if mach >= _CD_MACH[-1]:
        return _CD_VALUES[-1]
    i = bisect_right(_CD_MACH, mach) - 1
    span = _CD_MACH[i + 1] - _CD_MACH[i]
    t = (mach - _CD_MACH[i]) / span
    return _CD_VALUES[i] + t * (_CD_VALUES[i + 1] - _CD_VALUES[i])


def drag_force_magnitude_n(speed_ms, altitude_m, frontal_area_m2):
    """Magnitude of aerodynamic drag, N. Acts along -v_hat.

        F_d = 0.5 * rho(h) * v^2 * Cd(M) * A
    """
    if speed_ms <= 0.0:
        return 0.0
    mach = speed_ms / speed_of_sound_ms(altitude_m)
    return (
        0.5
        * density_kgm3(altitude_m)
        * speed_ms
        * speed_ms
        * drag_coefficient(mach)
        * frontal_area_m2
    )
