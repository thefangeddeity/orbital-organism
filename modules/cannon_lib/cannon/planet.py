"""Planet-centred frame: gravity, the ground, and the flat-earth fixture.

COORDINATE FRAME: planet-centred, non-rotating, 2D, in the equatorial plane.
The origin is the Earth's CENTRE, not the gun. The gun sits at
(0, R_EARTH_M + trunnion height). Positive x is downrange.

This replaces the frame every other module was written against. The cannon's
890 m arc was never a parabola - it is a segment of an ellipse with the
Earth's centre at the far focus, and the parabola was an approximation that
happened to be good enough. This makes that exactly true rather than
approximately true.

PRECISION HAZARD, READ BEFORE DOING CONTACT GEOMETRY HERE. Positions are now
~6.4e6 m while block placement needs sub-metre accuracy: 1e-7 relative.
float64 carries ~1e-16 so the state vector is fine, but contact mathematics
is NOT - swept-sphere-vs-OBB in planet-centred coordinates subtracts large
nearly-equal numbers and loses most of its significant digits. Translate to
a block-local frame before any contact geometry and translate back
afterwards. A precision loss here shows up as jitter, not as a failed
assertion, so the CCD gates will not necessarily catch it.
"""

import math
from contextlib import contextmanager

from .vec import Vec2  # re-exported: gates build states in this frame

#: Standard gravitational parameter, m^3/s^2. USED DIRECTLY, never computed
#: as G*M: mu is known to far more significant digits than either factor, so
#: multiplying them out would throw away precision for nothing.
MU_EARTH_M3_S2 = 3.986004418e14

#: Mean radius, m. Spherical model - no oblateness, see J2 below.
R_EARTH_M = 6371000.0

#: Earth rotation rate, rad/s. OFF, deliberately.
#:
#: Present as a named constant and wired through the acceleration function so
#: it can be switched on without restructuring. At cannon scale Coriolis
#: deflection over 890 m is sub-millimetre, so rotation buys nothing yet. It
#: starts mattering at the rocket rungs, where the launch site's ~465 m/s
#: eastward velocity is free delta-v and launch azimuth becomes a real
#: choice. Turning it on also forces a display-frame split - physics in
#: inertial, rendering in a frame rotating with the ground - which is real
#: work with no payoff now.
OMEGA_EARTH_RAD_S = 0.0

#: Standard gravity for the flat-earth fixture only, m/s^2.
FLAT_GRAVITY_MS2 = 9.80665

_flat_earth = False


@contextmanager
def flat_earth():
    """Swap in constant gravity and a y = 0 ground. TEST FIXTURE ONLY.

    This exists to keep gate 1's exact v^2/g identity alive as an integrator
    regression, and for nothing else. IT MUST NOT APPEAR IN SETTINGS - it is
    not a game mode, it is a way of asking whether the integrator still
    works on a problem with a closed-form answer.
    """
    global _flat_earth
    previous = _flat_earth
    _flat_earth = True
    try:
        yield
    finally:
        _flat_earth = previous


def flat_earth_active():
    """True when the fixture is engaged. Gates assert on this."""
    return _flat_earth


def gravitational_acceleration(position_m):
    """Acceleration due to gravity at a position, m/s^2.

    Structured so perturbation terms can be added without any caller
    changing. J2 (Earth's oblateness) is ONE ADDITIVE TERM and slots in
    below, which is exactly why it can wait: it is invisible for a cannon
    and only matters once something is in orbit for multiple revolutions.
    No stub function that returns zero - the comment is the placeholder.
    """
    if _flat_earth:
        return Vec2(0.0, -FLAT_GRAVITY_MS2)

    distance_m = position_m.length()
    acceleration = position_m * (-MU_EARTH_M3_S2 / (distance_m**3))
    # acceleration += j2_term(position_m)   # not implemented
    return acceleration


def altitude_m(position_m):
    """Height above the ground: |r| - R_EARTH, or y under the fixture.

    Signed, so it doubles as the ground-crossing function the impact
    bisection roots on.
    """
    if _flat_earth:
        return position_m.y
    return position_m.length() - R_EARTH_M


def local_up(position_m):
    """Unit outward normal of the ground under a position.

    r-hat, not (0, 1). The block's "up" is the local normal AT ITS OWN
    POSITION, which is not parallel to the gun's - 0.008 degrees apart at
    890 m. Negligible in magnitude, but contact code must be written in
    terms of this rather than a global up, or every later rung inherits a
    flat-earth assumption buried in the corner solver.
    """
    if _flat_earth:
        return Vec2(0.0, 1.0)
    return position_m.normalized()


def surface_origin():
    """The ground point directly below the gun."""
    return Vec2(0.0, 0.0 if _flat_earth else R_EARTH_M)


def surface_point_below(position_m):
    """The point on the ground directly below a position."""
    if _flat_earth:
        return Vec2(position_m.x, 0.0)
    return position_m.normalized() * R_EARTH_M


def downrange_m(position_m):
    """Arc length along the surface from the gun, m.

    Under round-earth this is R * theta, not the Cartesian x. The two agree
    to about a part in 10^7 at cannon range, but they are different
    quantities and the distinction only grows.
    """
    if _flat_earth:
        return position_m.x
    return R_EARTH_M * math.atan2(position_m.x, position_m.y)


def position_at(downrange_m_value, altitude_m_value):
    """World position a given arc distance downrange and height up."""
    if _flat_earth:
        return Vec2(downrange_m_value, altitude_m_value)
    angle_rad = downrange_m_value / R_EARTH_M
    radius = R_EARTH_M + altitude_m_value
    return Vec2(math.sin(angle_rad) * radius, math.cos(angle_rad) * radius)


def surface_tilt_rad(downrange_m_value):
    """Rotation that lays a body flat on the ground at this arc distance.

    A box resting on a curved surface is tilted relative to the world axes
    by the angle its own position subtends. Zero under the fixture, so a
    flat-earth block is axis-aligned exactly as before.
    """
    if _flat_earth:
        return 0.0
    return -downrange_m_value / R_EARTH_M


def flight_path_angle_rad(position_m, velocity_m_s):
    """Angle between velocity and the LOCAL horizontal at the shot, rad.

    Distinct from launch elevation, which is fixed at the muzzle. With
    radial gravity the local vertical rotates as the shot travels, so these
    are different quantities that happen to nearly coincide at cannon scale.
    """
    up = local_up(position_m)
    vertical = velocity_m_s.dot(up)
    horizontal = velocity_m_s.dot(up.perp())
    return math.atan2(vertical, -horizontal if horizontal < 0 else horizontal)
