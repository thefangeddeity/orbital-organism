"""The two reference frames, and how to convert between them.

Batch 9 SS5, in the form batch 14 SS2 corrected it to.

    ECI     Earth-Centred Inertial. Origin at the planet's centre, axes fixed
            against the stars. THE INTEGRATION FRAME: every trajectory in this
            simulator is integrated here, and nothing else is.

    ECEF    Earth-Centred, Earth-Fixed. Same origin, axes rotating with the
            planet at OMEGA_EARTH_RAD_S. THE EMPLACEMENT FRAME: the gun, the
            targets and the atmosphere are all fixed in this one.

THERE ARE NO FICTITIOUS FORCES IN THIS MODULE AND THERE MUST NEVER BE ANY.
Batch 9 SS5 originally asked for Coriolis and centrifugal terms; batch 14 SS2
retracted that, and the retraction is the important part of this file. A
trajectory integrated in an inertial frame has no fictitious forces by
definition - they are what appears when you write Newton's second law in a
rotating frame, and we do not. Adding them to an ECI integration would
double-count rotation and produce a trajectory that is wrong in a way that
looks physical.

Rotation therefore enters in exactly two places, both of them real:

    1.  INITIAL CONDITIONS. A gun bolted to the ground is not at rest in ECI;
        it is moving at omega x r. `surface_velocity_eci` is that velocity, and
        it is what the muzzle velocity is added to.

    2.  RELATIVE WIND. The atmosphere co-rotates, so drag acts on the velocity
        relative to the air, not on the ECI velocity. `air_relative_velocity`
        is that difference.

The omega x r term inside `velocity_eci_from_ecef` is NOT a fictitious force
either. It is the derivative of a rotating coordinate transformation - the
kinematic fact that a fixed point in a turning frame is moving. Converting a
velocity between frames requires it; integrating a trajectory does not.

OMEGA_EARTH_RAD_S IS CURRENTLY ZERO, so every function here is the identity and
every existing gate is bit-identical with them wired in. That is deliberate:
the plumbing is proved correct while it costs nothing, and batch 14 changes one
constant rather than every call site. The conversions take omega as an ARGUMENT
so they can be gated at the real rate today - a function only ever exercised at
omega = 0 would be a guess.

WHAT IS NOT MODELLED, stated because the gate deliberately does not assert it:
on a rotating perfect sphere a body "at rest" on the ground is not in
equilibrium - the ground has to supply the centripetal force, and on a sphere
the surface normal cannot. The real Earth solves this by being an ellipsoid,
its equatorial bulge exactly the shape that makes local plumb-vertical normal
to the surface. We model neither the bulge nor J2, so at non-zero omega a
resting body would slowly slide poleward. Gate 31 asserts rest at omega = 0,
where it is true, and asserts the CONVERSIONS at the real rate, where they are
true. It does not assert rest at the real rate, because that would be false.
"""

def rotation_angle_rad(time_s, omega_rad_s):
    """How far ECEF has turned from ECI at `time_s`.

    The two frames coincide at t = 0 by definition. There is no epoch and no
    sidereal offset: nothing in this simulator refers to a real date, and a
    zero point invented here would be a number with no meaning attached.
    """
    return omega_rad_s * time_s


def position_eci_from_ecef(position_ecef, time_s, omega_rad_s):
    """A planet-fixed position, expressed in the inertial frame."""
    return position_ecef.rotated(rotation_angle_rad(time_s, omega_rad_s))


def position_ecef_from_eci(position_eci, time_s, omega_rad_s):
    """An inertial position, expressed in the planet-fixed frame."""
    return position_eci.rotated(-rotation_angle_rad(time_s, omega_rad_s))


def surface_velocity_eci(position_eci, omega_rad_s):
    """Inertial velocity of a point FIXED TO THE GROUND at `position_eci`.

    omega x r, which in the plane is omega * perp(r). This is how rotation
    reaches a trajectory: it is added to the muzzle velocity, not to the
    acceleration.
    """
    return position_eci.perp() * omega_rad_s


def velocity_eci_from_ecef(velocity_ecef, position_ecef, time_s, omega_rad_s):
    """A planet-fixed velocity, expressed in the inertial frame.

    v_eci = R(theta) * v_ecef + omega x r_eci. The second term is kinematics,
    not a force - see the module docstring.
    """
    angle_rad = rotation_angle_rad(time_s, omega_rad_s)
    position_eci = position_ecef.rotated(angle_rad)
    return (
        velocity_ecef.rotated(angle_rad)
        + surface_velocity_eci(position_eci, omega_rad_s)
    )


def velocity_ecef_from_eci(velocity_eci, position_eci, time_s, omega_rad_s):
    """An inertial velocity, expressed in the planet-fixed frame.

    The exact inverse of `velocity_eci_from_ecef`: subtract the frame's own
    motion in ECI, then rotate. Doing it in the other order subtracts a vector
    expressed in the wrong frame, which is a mistake that stays small enough to
    look plausible - gate 31 round-trips to catch it.
    """
    angle_rad = rotation_angle_rad(time_s, omega_rad_s)
    return (
        velocity_eci - surface_velocity_eci(position_eci, omega_rad_s)
    ).rotated(-angle_rad)


def air_relative_velocity(velocity_eci, position_eci, omega_rad_s):
    """Velocity relative to the CO-ROTATING ATMOSPHERE, in ECI components.

    The air is fixed in ECEF, so its inertial velocity at a point is omega x r
    and the shot's velocity through it is the difference. Returned in ECI
    components rather than rotated into ECEF because the drag force is applied
    to an ECI-integrated state and rotating twice would be waste.

    Wind proper - anything the atmosphere does other than co-rotate - is
    parked. This is the co-rotation term only.
    """
    if omega_rad_s == 0.0:
        # Exactly the ECI velocity, and returning it unchanged keeps the drag
        # path bit-identical at the shipping constant instead of merely close.
        return velocity_eci
    return velocity_eci - surface_velocity_eci(position_eci, omega_rad_s)


def frames_coincide(omega_rad_s):
    """True when ECI and ECEF are the same frame, so conversions are identity.

    Named rather than written as `omega_rad_s == 0.0` at each call site, so the
    assumption is greppable when batch 14 switches rotation on. `gun.fire` and
    the drag path are the two places that ask.
    """
    return omega_rad_s == 0.0
