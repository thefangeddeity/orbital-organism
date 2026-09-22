"""State vector, RK4 integrator, and free-flight trajectory integration.

SI throughout. Coordinates are PLANET-CENTRED - the origin is the Earth's
centre, not the gun. See `planet`. This module imports the ballistics and
planet models; it imports nothing from the render layer, and never will.

The integrator is fixed-step RK4 at `FREE_FLIGHT_DT_S`, deliberately
decoupled from any render tick. Physics is never integrated on the frame
clock.
"""

import math
from dataclasses import dataclass, replace

from .ballistics import drag_force_magnitude_n, speed_of_sound_ms
from .frames import air_relative_velocity
from .planet import (
    OMEGA_EARTH_RAD_S,
    altitude_m,
    downrange_m,
    gravitational_acceleration,
)
from .vec import Vec2

#: Fixed integration step for free flight, s.
FREE_FLIGHT_DT_S = 1e-3


@dataclass(frozen=True, slots=True)
class ShotState:
    """Complete state of the shot. Nothing else belongs here."""

    pos: Vec2  # m
    vel: Vec2  # m/s
    spin: float  # rad/s, positive counter-clockwise
    mass: float  # kg
    radius: float  # m

    @property
    def frontal_area_m2(self):
        return math.pi * self.radius * self.radius

    @property
    def inertia(self):
        """Moment of inertia of a solid sphere, kg m^2. Derived, not stored.

        Needed so that a tangential contact impulse can spin the shot up
        without leaking angular momentum.
        """
        return 0.4 * self.mass * self.radius * self.radius

    @property
    def speed_ms(self):
        return self.vel.length()

    @property
    def mach(self):
        return self.speed_ms / speed_of_sound_ms(altitude_m(self.pos))


@dataclass(frozen=True, slots=True)
class FlightResult:
    """Outcome of one free-flight integration."""

    impact: ShotState | None  # None if the flight timed out before ground
    flight_time_s: float
    apex_height_m: float
    apex_range_m: float
    max_mach: float
    samples: tuple  # (time_s, ShotState) pairs, for the render layer


def free_flight_acceleration(pos, vel, mass_kg, frontal_area_m2, drag_enabled):
    """Acceleration on a free-flying shot, m/s^2.

    Gravity plus aerodynamic drag. Magnus is stage 5 and is not here yet.

    Gravity is RADIAL, from `planet.gravitational_acceleration` - it is no
    longer a constant (0, -g) and no longer points the same way at every
    point of the trajectory. Drag reads altitude as |r| - R_EARTH rather
    than as y.

    DRAG ACTS ON THE AIR-RELATIVE VELOCITY, not on the inertial velocity. The
    atmosphere co-rotates with the planet, so at non-zero rotation the two
    differ by omega x r. At OMEGA_EARTH_RAD_S = 0 they are the same object and
    this is bit-identical to reading `vel` directly - which is the point:
    batch 14 switches one constant instead of editing the hot path.

    NO FICTITIOUS FORCES. This is an inertial-frame integration; there is no
    Coriolis and no centrifugal term here and there must never be one. See
    `frames` for why.
    """
    gravity = gravitational_acceleration(pos)
    ax = gravity.x
    ay = gravity.y
    if drag_enabled:
        relative = air_relative_velocity(vel, pos, OMEGA_EARTH_RAD_S)
        speed_ms = relative.length()
        if speed_ms > 0.0:
            force_n = drag_force_magnitude_n(
                speed_ms, altitude_m(pos), frontal_area_m2
            )
            # -F * v_hat / m, expanded to avoid building an extra Vec2. Applied
            # along the RELATIVE velocity, which is the direction drag acts in.
            k = -force_n / (mass_kg * speed_ms)
            ax += k * relative.x
            ay += k * relative.y
    return Vec2(ax, ay)


def rk4_step(pos, vel, dt, accel):
    """One classical RK4 step of d(pos)/dt = vel, d(vel)/dt = accel(pos, vel).

    `accel` takes (pos, vel) and returns a Vec2 acceleration.
    """
    half = dt * 0.5

    a1 = accel(pos, vel)
    p2 = pos + vel * half
    v2 = vel + a1 * half

    a2 = accel(p2, v2)
    p3 = pos + v2 * half
    v3 = vel + a2 * half

    a3 = accel(p3, v3)
    p4 = pos + v3 * dt
    v4 = vel + a3 * dt

    a4 = accel(p4, v4)

    sixth = dt / 6.0
    new_pos = pos + (vel + 2.0 * v2 + 2.0 * v3 + v4) * sixth
    new_vel = vel + (a1 + 2.0 * a2 + 2.0 * a3 + a4) * sixth
    return new_pos, new_vel


def _acceleration_for(state, drag_enabled):
    """Bind a state's constant properties into an accel(pos, vel) callable."""
    area_m2 = state.frontal_area_m2
    mass_kg = state.mass

    def accel(pos, vel):
        return free_flight_acceleration(pos, vel, mass_kg, area_m2, drag_enabled)

    return accel


def step_free_flight(
    state, dt=FREE_FLIGHT_DT_S, *, drag_enabled=True, ground_enabled=True
):
    """Advance the shot by at most `dt`. Returns (state, hit_ground, elapsed_s).

    THE GROUND IS A CIRCLE OF RADIUS R_EARTH, not the line y = 0. The test
    is `altitude_m(pos) < 0`, which is signed and doubles as the function
    the crossing bisection roots on, so the same code serves both the round
    model and the flat-earth fixture.

    If the step would cross the ground it is shortened by bisection so the
    returned state sits exactly on it, and `elapsed_s` is the real time that
    took. This is the single stepping primitive: the batch integrator and
    the interactive loop both go through it, so a shot fired on screen
    follows exactly the same trajectory as one fired headless.

    Pass `ground_enabled=False` to fly through the ground, which is what the
    conservation gates need.

    Note this is a ground *stop*, not a collision response.

    Tail parameters are KEYWORD-ONLY on purpose. They are flags and
    configuration, and `0.0`, `None` and `False` are all falsy or valid, so a
    positional call that binds to the wrong slot after a signature change is
    both easy to write and completely silent. The bare `*` turns that into a
    TypeError at the call site. Batch 8 shipped exactly that bug twice on
    adjacent lines: one crashed at impact, the other quietly disabled the
    ground and flew the shot through the Earth.
    """
    accel = _acceleration_for(state, drag_enabled)
    new_pos, new_vel = rk4_step(state.pos, state.vel, dt, accel)

    if ground_enabled and altitude_m(new_pos) < 0.0 <= altitude_m(state.pos):
        hit_dt = _bisect_ground_crossing(state.pos, state.vel, dt, accel)
        new_pos, new_vel = rk4_step(state.pos, state.vel, hit_dt, accel)
        return replace(state, pos=new_pos, vel=new_vel), True, hit_dt

    return replace(state, pos=new_pos, vel=new_vel), False, dt


def integrate_flight(
    state,
    dt=FREE_FLIGHT_DT_S,
    *,
    drag_enabled=True,
    ground_enabled=True,
    max_time_s=120.0,
    sample_interval_s=0.02,
):
    """Integrate free flight until the shot reaches the ground or times out.

    Pass `ground_enabled=False` to disable ground termination, which is what
    the conservation gates need.

    The ground crossing is located by bisecting the final step, so impact
    time and range do not inherit the step size as an error.
    """
    t_s = 0.0
    current = state

    apex_height_m = altitude_m(current.pos)
    apex_range_m = downrange_m(current.pos)
    max_mach = current.mach

    samples = [(0.0, state)]
    next_sample_s = sample_interval_s

    impact = None
    while t_s < max_time_s:
        current, hit_ground, elapsed_s = step_free_flight(
            current,
            dt,
            drag_enabled=drag_enabled,
            ground_enabled=ground_enabled,
        )
        t_s += elapsed_s

        if hit_ground:
            impact = current
            samples.append((t_s, impact))
            break

        if altitude_m(current.pos) > apex_height_m:
            apex_height_m = altitude_m(current.pos)
            apex_range_m = downrange_m(current.pos)

        max_mach = max(max_mach, current.mach)

        if t_s >= next_sample_s:
            samples.append((t_s, current))
            next_sample_s += sample_interval_s

    return FlightResult(
        impact=impact,
        flight_time_s=t_s,
        apex_height_m=apex_height_m,
        apex_range_m=apex_range_m,
        max_mach=max_mach,
        samples=tuple(samples),
    )


def _bisect_ground_crossing(pos, vel, dt, accel, iterations=60):
    """Find the sub-step duration at which the trajectory reaches the ground.

    Roots on signed altitude, so it is identical under both the round model
    and the flat-earth fixture.
    """
    low_s = 0.0
    high_s = dt
    for _ in range(iterations):
        mid_s = 0.5 * (low_s + high_s)
        mid_pos, _ = rk4_step(pos, vel, mid_s, accel)
        if altitude_m(mid_pos) < 0.0:
            high_s = mid_s
        else:
            low_s = mid_s
    return 0.5 * (low_s + high_s)
