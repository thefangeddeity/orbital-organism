"""The target block, continuous collision detection, and contact resolution.

SI throughout. Imports from `physics` and `ballistics`; imports nothing from
the render layer.

NO MATERIAL TABLE EXISTED BEFORE STAGE 4. The stage-4 brief refers to "the
existing material table"; there wasn't one, so it was created here. Every
number in every entry is chosen to be plausible, not sourced - see
`ContactMaterial`.

THE LARGEST OMISSION IN THIS MODULE IS THAT THE TARGET CANNOT BREAK.
The target is diabase. Stone is brittle, and historically a cannonball
against masonry broke the masonry rather than displacing it. So for a stone
target, fracture is not a refinement on top of the response - it IS the
response, and what this module models instead (a block that slides, rocks
and tips as a rigid body) is the secondary effect standing in for the
primary one.

This is a bigger gap than the restitution one. The restitution law is at
least the right kind of model applied slightly out of regime; here the right
kind of model is absent altogether. Diabase makes the gap comfortable to
live with rather than closing it: it is dense enough to resist being moved
and tough enough to resist being broken, which makes it a good instrument
for exercising the contact solver and a dull thing to shoot at. Anything
softer - timber, masonry, earthwork - would make the omission obvious
immediately. Fracture is parked, and the parked target-assortment work is
coupled to it rather than independent of it.

Collision detection is a swept sphere against an oriented box. At 88 m/s and
dt = 1e-3 s the shot moves 88 mm per step against a 500 mm block, which
discrete overlap testing would probably survive at the default charge and
would certainly not survive at 440 m/s.

Contact resolution carries the `r x j` coupling on both bodies, so where the
shot lands on the block changes the outcome. Gate 7 (angular momentum
conservation) is what proves that coupling is right.
"""

import math
from dataclasses import dataclass, replace

from .physics import FREE_FLIGHT_DT_S, Vec2, step_free_flight
from .planet import (
    altitude_m,
    downrange_m,
    gravitational_acceleration,
    local_up,
    position_at,
    surface_tilt_rad,
)

# --------------------------------------------------------------------------
# Material table
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContactMaterial:
    """Quasi-static restitution, Coulomb friction and yield velocity.

    EVERY VALUE IN EVERY ENTRY IS CHOSEN, NOT MEASURED. This caveat applies
    to each entry individually and does NOT carry over from one to the next
    just because the table already existed - a new entry is a new set of
    invented numbers. They are order-of-magnitude plausible and their
    ORDERING against each other is meant to be defensible; the absolute
    figures are not.

    `restitution` is the QUASI-STATIC coefficient - the value below yield.
    It is not what gets used at speed: see `restitution_at`.

    `yield_velocity_ms` is a property of the CONTACT PAIR, not a global. The
    speed at which a contact stops being elastic is set by the softer of the
    two materials, so cast iron against stone leaves the elastic regime at a
    different speed from cast iron against steel. It was a module constant
    when only one pair existed; keeping it that way would have been wrong
    the moment a second pair was added.

    Adding a material is meant to be DATA, NOT CODE - a new entry here and
    nothing else - so that the parked target-assortment work does not need
    to touch the solver.
    """

    name: str
    restitution: float
    friction: float
    yield_velocity_ms: float


#: Exponent of the elastic-plastic restitution scaling. Johnson, *Contact
#: Mechanics* (CUP, 1985): above the yield velocity, elastic-plastic impact
#: gives e proportional to v^(-1/4).
RESTITUTION_EXPONENT = 0.25


def restitution_at(normal_closing_ms, material):
    """Velocity-dependent coefficient of restitution for a contact pair.

        e(v) = min(E0, E0 * (V_yield / |v|) ** 0.25)

    Johnson, *Contact Mechanics* (1985). Below the pair's yield velocity
    this returns the quasi-static coefficient unchanged; above it,
    restitution falls as the contact goes elastic-plastic and impact energy
    is spent on permanent deformation instead of being returned.

    STILL MORE ELASTIC THAN REALITY, and for a stone target that gap is now
    the model's largest single omission - see the module docstring. The
    scaling narrows the error; it does not close it.
    """
    speed_ms = abs(normal_closing_ms)
    if speed_ms <= material.yield_velocity_ms:
        return material.restitution
    return min(
        material.restitution,
        material.restitution
        * (material.yield_velocity_ms / speed_ms) ** RESTITUTION_EXPONENT,
    )


#: The pair the e(v) law was first fitted against, and now the shot-versus-
#: slab pair of batch 16 SS4's target.
CAST_IRON_ON_STEEL = ContactMaterial("cast iron on steel", 0.42, 0.18, 1.0)

#: Slab against the ground. CHOSEN, on the same terms as the pair above.
STEEL_ON_SOIL = ContactMaterial("steel on soil", 0.18, 0.55, 0.5)

#: Shot against the ground, now that the ground is a MATERIAL rather than a
#: stop - batch 16 Part II SS13. Restitution and friction are CHOSEN, on the
#: same terms as every other entry in this table - but not ungrounded: at
#: this pair's yield velocity, restitution at 28 m/s works out to about 0.2,
#: matching the ~85-99% kinetic-energy loss Kim & Choi measured for steel
#: spheres ricocheting off sand at comparable speed (see
#: `critical_grazing_angle_deg` below for the full citation). The angle that
#: decides ricochet from burial is sourced, not chosen; these two numbers are
#: not, and are checked against that same source only for plausibility.
CAST_IRON_ON_SOIL = ContactMaterial("cast iron on soil", 0.35, 0.45, 3.0)

#: CAST_IRON_ON_DIABASE and DIABASE_ON_SOIL are DELETED with the stone box -
#: batch 16 SS4. Nothing in the shipped scenario is stone any more, and a
#: material pair with no body left to use it is dead data.
#:
#: Every pair in the table. Gate 10 sweeps this, so a new entry is covered
#: by the restitution gate automatically rather than needing a test edit.
MATERIAL_TABLE = (CAST_IRON_ON_STEEL, STEEL_ON_SOIL, CAST_IRON_ON_SOIL)

#: Critical grazing angle for a sphere ricocheting off sand or soil, degrees,
#: as a function of impact speed - SOURCED, not chosen, from two anchors:
#:
#:   Daneshi, G.H. & Johnson, W. "The ricochet of spherical projectiles off
#:   sand." Int. J. Mech. Sci., 1977, 19(8), 491-497: a 12.7 mm steel ball
#:   ricochets off sand below about 20 deg at 100 m/s, falling to about 18
#:   deg at 375 m/s - the critical angle falls as speed rises.
#:
#:   Kim, Y.K. & Choi, W.C. "Ricochet of Spheres on Sand of Various
#:   Temperature." Defence Science Journal, 2018, 68(2), 150-158: 7-10 mm
#:   steel spheres at 24-32 m/s ricocheted at 15 deg and 20 deg but never at
#:   25 deg - "the critical angles of the spheres are about 25 deg" - which
#:   is exactly the 1977 trend continued to lower speed.
#:
#: NEITHER PAPER TESTED A 12-POUNDER SHOT (121 mm, 5.9 kg) OR SOIL RATHER
#: THAN SAND. They are the closest sourced analogue available - a shot on
#: real earthwork would differ - not a measurement of this exact body; that
#: caveat is inherited from every other figure in this module, not new to
#: this one. The interpolation between the three anchors (linear in log
#: speed) and the flat extrapolation past 375 m/s are CHOSEN, not sourced;
#: only the three (speed, angle) pairs themselves are.
_RICOCHET_SPEED_ANCHORS_MS = (28.0, 100.0, 375.0)
_RICOCHET_ANGLE_ANCHORS_DEG = (25.0, 20.0, 18.0)


def critical_grazing_angle_deg(speed_ms):
    """Grazing angle, degrees, above which a shot buries rather than skips."""
    speeds = _RICOCHET_SPEED_ANCHORS_MS
    angles = _RICOCHET_ANGLE_ANCHORS_DEG
    if speed_ms <= speeds[0]:
        return angles[0]
    if speed_ms >= speeds[-1]:
        return angles[-1]
    for i in range(len(speeds) - 1):
        if speeds[i] <= speed_ms <= speeds[i + 1]:
            t = (
                (math.log(speed_ms) - math.log(speeds[i]))
                / (math.log(speeds[i + 1]) - math.log(speeds[i]))
            )
            return angles[i] + t * (angles[i + 1] - angles[i])
    return angles[-1]

#: Dimensionless rolling-resistance coefficient for the block on soil.
#: Lumped: it damps residual rocking once the block is down. Sliding is
#: arrested by the Coulomb friction in DIABASE_ON_SOIL, not by this.
SOIL_ROLLING_RESISTANCE = 0.25

#: Below this normal approach speed, ground contacts are treated as fully
#: inelastic. Without it a resting block trades gravity's per-step velocity
#: increment against restitution forever and never settles. Standard
#: practice; the threshold is ~2x the per-step increment at dt = 1e-3 s.
REST_VELOCITY_THRESHOLD_MS = 0.05

#: Relaxation passes over the corner contacts per step.
#:
#: The four corners are solved sequentially, so each pass sees velocities the
#: previous corner already changed and the two supporting corners' lever-arm
#: torques do not cancel. One pass leaves a residual that integrates into a
#: visible sideways CREEP of a block that is supposed to be at rest -
#: measured at 5.5 mm/s, which is 126 mm over a 30 s flight, easily enough to
#: slide the block out from under an incoming shot. Iterating drives the
#: residual down by roughly an order of magnitude per pass.
GROUND_RELAXATION_PASSES = 4

#: A block in ground contact slower than this in BOTH senses is snapped to
#: rest. This is a numerical stabiliser, not physics: iteration reduces the
#: creep residual but cannot reach zero, and any nonzero residual integrates
#: without bound over a 30 s flight. Thresholds sit above gravity's per-step
#: velocity increment (g*dt = 9.8 mm/s at dt = 1e-3 s) so they actually
#: engage. The visible cost is that genuine sliding below 5 cm/s is
#: truncated to a stop rather than decaying asymptotically.
SLEEP_SPEED_MS = 0.05
SLEEP_OMEGA_RADS = 0.05


# --------------------------------------------------------------------------
# The block
# --------------------------------------------------------------------------

#: THE STONE BOX AND THE STEEL CRATE ARE DELETED - batch 16 SS4. Both were
#: built against a Tech Lead spec that contradicted a PM decision from the
#: batch 6 era: the target was always meant to be a standing steel plate that
#: topples, not a box that slides. The error was in the brief, not in the
#: build; see the batch 16 v2 report.
#:
#: The target is now ONE vertical steel slab. Multi-target support in
#: `step_shot_and_blocks` stays - the earliest-contact sweep is correct and
#: worth keeping - but nothing in this module constructs more than one body by
#: default. Multiple targets return with batch 10's block-against-block
#: contact, where a second body means something.

#: Slab height, m. Vertical, the "3.0 m tall" the brief specifies directly.
SLAB_HEIGHT_M = 3.0

#: Slab face width, m. The CROSS-RANGE dimension - perpendicular to the plane
#: this simulator models, so it never enters the 2D collision geometry. It
#: enters mass and inertia only, exactly as it would in the real object.
SLAB_WIDTH_M = 1.5

#: Slab thickness, m. The dimension ALONG THE LINE OF FIRE - this is what
#: makes it a "plate" rather than a block, and it is what the round actually
#: has to reach: half of it, 12.5 mm, is `half_width` below.
SLAB_THICKNESS_M = 0.025

#: Structural steel, kg/m^3. Mild steel; unchanged from the crate this
#: replaces, which used the same density for the same reason.
STEEL_DENSITY_KGM3 = 7850.0

#: Slab mass, kg. DERIVED from all three real dimensions: rho * H * W * T.
#: Reported in the batch report against the brief's own arithmetic rather than
#: trusted from it, per instruction.
SLAB_MASS_KG = STEEL_DENSITY_KGM3 * SLAB_HEIGHT_M * SLAB_WIDTH_M * SLAB_THICKNESS_M

#: Half-extents used by the 2D contact geometry, m.
#:
#: THE THIRD DIMENSION (SLAB_WIDTH_M) IS NOT ONE OF THEM. This simulator
#: models one vertical plane - downrange and altitude - so a plate standing
#: broadside to the gun projects into that plane as HEIGHT (vertical) by
#: THICKNESS (downrange); its face width is perpendicular to the plane and
#: invisible to it, exactly as it is invisible to a side-on photograph.
SLAB_HALF_HEIGHT_M = 0.5 * SLAB_HEIGHT_M
SLAB_HALF_WIDTH_M = 0.5 * SLAB_THICKNESS_M

#: Slab inertia, kg m^2, about the centroidal axis the 2D sim rotates around -
#: the CROSS-RANGE axis, perpendicular to the modelled plane. That is the
#: standard rectangular result I = m(a^2+b^2)/12 with a, b the two dimensions
#: PERPENDICULAR TO the rotation axis: thickness and height. Width does not
#: appear, for the same reason it does not appear in half_width/half_height -
#: rotation about the cross-range axis does not move mass along it.
SLAB_INERTIA_KGM2 = (
    SLAB_MASS_KG * (SLAB_THICKNESS_M**2 + SLAB_HEIGHT_M**2) / 12.0
)

#: Cap on collision resolutions within a single physics step.
MAX_IMPACTS_PER_STEP = 4

_CORNER_SIGNS = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))


@dataclass(frozen=True, slots=True)
class BlockState:
    """Rigid body free to translate and rotate in the plane.

    THE BODY OWNS ITS MATERIALS. They used to be parameters of
    `step_shot_and_block`, which was fine while there was exactly one target
    and stopped being fine the moment there were two of different substances:
    the caller would have had to hold a parallel list of materials in the same
    order as the blocks, and any reordering would have silently swapped stone
    for steel. A material carried on the body cannot be paired with the wrong
    body.
    """

    pos: Vec2  # centre of mass, m
    vel: Vec2  # m/s
    theta: float  # rad, positive counter-clockwise
    omega: float  # rad/s
    mass: float  # kg
    inertia: float  # kg m^2
    #: Half-extents in the block's OWN local frame at theta=0: half_width along
    #: local x (the downrange axis when theta=0), half_height along local y
    #: (the vertical axis when theta=0). SEPARATE FIELDS, not one square value -
    #: batch 16 SS4's slab is 3.0 m tall and 25 mm thick, and a single
    #: `half_extent` cannot represent that. Every square body (nothing left in
    #: this file, kept for future bodies) is simply the case half_width ==
    #: half_height.
    half_width: float  # m
    half_height: float  # m
    #: Shot-against-this-body pair, and this-body-against-the-ground pair.
    #: Defaulted to the slab's own materials, the only pair left in the table.
    material: ContactMaterial = CAST_IRON_ON_STEEL
    ground_material: ContactMaterial = STEEL_ON_SOIL
    #: Shown in the HUD and on the focus ring.
    label: str = "target"

    def corners(self):
        """The four corners in world coordinates."""
        return tuple(
            self.pos
            + Vec2(sx * self.half_width, sy * self.half_height).rotated(self.theta)
            for sx, sy in _CORNER_SIGNS
        )


def make_slab(downrange_m_value):
    """A standing steel plate at rest on the ground, `downrange_m_value` along it.

    THE ONE TARGET - batch 16 SS4. Stands broadside to the gun: its 3.0 m
    height and 25 mm thickness are what the 2D contact geometry sees;
    `SLAB_WIDTH_M`, the cross-range face width, is not a dimension this
    simulator models and enters only the mass and inertia above it.

    The argument is ARC DISTANCE along the surface, not a Cartesian x. `pos`
    is placed at radius R_EARTH + half_HEIGHT, so the plate's BASE touches the
    ground - not half_width, which for this body is a different, much smaller
    number. TILTED to match the local horizon, so it stands plumb on the
    curve rather than leaning at a fixed world angle; under the flat-earth
    fixture the tilt is zero.
    """
    return BlockState(
        pos=position_at(downrange_m_value, SLAB_HALF_HEIGHT_M),
        vel=Vec2(0.0, 0.0),
        theta=surface_tilt_rad(downrange_m_value),
        omega=0.0,
        mass=SLAB_MASS_KG,
        inertia=SLAB_INERTIA_KGM2,
        half_width=SLAB_HALF_WIDTH_M,
        half_height=SLAB_HALF_HEIGHT_M,
        material=CAST_IRON_ON_STEEL,
        ground_material=STEEL_ON_SOIL,
        label="steel slab",
    )


# --------------------------------------------------------------------------
# Continuous collision detection: swept sphere vs oriented box
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Contact:
    """Earliest contact found over a swept step."""

    time_fraction: float  # where in [0, 1] of the sweep it happened
    point: Vec2  # world contact point, on both surfaces
    normal: Vec2  # unit, outward from the block toward the shot
    feature: str  # "face" or "corner"


def sweep_sphere_vs_block(start_m, end_m, radius_m, block):
    """Earliest contact of a sphere swept from `start_m` to `end_m`, or None.

    The sweep is treated as a straight segment. Over one 1e-3 s step the
    trajectory's departure from a straight line is gravity's sag, about
    5e-6 m, which is four orders of magnitude below the block.

    The block is treated as stationary over the step: its speed is three
    orders of magnitude below the shot's. If a later stage makes the block
    fast, this needs revisiting.

    Method: work in the block's local frame, where the box is axis-aligned.
    Expand the box by the sphere radius, which gives a rounded rectangle
    (the Minkowski sum), and find the earliest intersection of the ray with
    it. In 2D that rectangle has exactly two kinds of feature - four flat
    sides and four quarter-circle corners - so a hit is classified "face" or
    "corner". There is no separate edge case: the faces of the 3D cube are
    what appear as the sides of this cross-section.
    """
    half_width = block.half_width
    half_height = block.half_height
    local_start = (start_m - block.pos).rotated(-block.theta)
    local_end = (end_m - block.pos).rotated(-block.theta)
    delta = local_end - local_start

    overlap_normal = _overlap_normal(local_start, half_width, half_height, radius_m)
    if overlap_normal is not None:
        return _make_contact(block, 0.0, local_start, overlap_normal, radius_m)

    best_time = math.inf
    best_normal = None
    best_feature = ""

    # Flat sides of the expanded rectangle. Each face's OWN half-extent (along
    # its normal) sets where the sphere surface touches the expanded plane;
    # the OTHER axis's half-extent bounds where that face ends and the
    # rounded corner begins. A square block has these equal; a rectangular
    # one - the slab - does not, and conflating them was the bug this
    # generalisation exists to avoid.
    for axis, half_along, half_across in (
        (Vec2(1.0, 0.0), half_width, half_height),
        (Vec2(0.0, 1.0), half_height, half_width),
    ):
        for sign in (-1.0, 1.0):
            normal = axis * sign
            along = delta.dot(normal)
            if abs(along) < 1e-15:
                continue
            time = ((half_along + radius_m) - local_start.dot(normal)) / along
            if not 0.0 <= time <= 1.0 or time >= best_time:
                continue
            centre = local_start + delta * time
            if abs(centre.dot(normal.perp())) <= half_across:
                best_time, best_normal, best_feature = time, normal, "face"

    # Quarter-circle corners of the expanded rectangle.
    length_sq = delta.dot(delta)
    if length_sq > 1e-30:
        for sx, sy in _CORNER_SIGNS:
            corner = Vec2(sx * half_width, sy * half_height)
            offset = local_start - corner
            b = 2.0 * offset.dot(delta)
            c = offset.dot(offset) - radius_m * radius_m
            discriminant = b * b - 4.0 * length_sq * c
            if discriminant < 0.0:
                continue
            time = (-b - math.sqrt(discriminant)) / (2.0 * length_sq)
            if not 0.0 <= time <= 1.0 or time >= best_time:
                continue
            centre = local_start + delta * time
            best_time = time
            best_normal = (centre - corner).normalized()
            best_feature = "corner"

    if best_normal is None:
        return None

    centre = local_start + delta * best_time
    return _make_contact(block, best_time, centre, best_normal, radius_m, best_feature)


def _overlap_normal(local_point, half_width, half_height, radius_m):
    """Outward normal if the sphere already overlaps the box, else None."""
    outside_x = abs(local_point.x) - half_width
    outside_y = abs(local_point.y) - half_height

    if outside_x > 0.0 and outside_y > 0.0:  # corner region
        corner = Vec2(
            math.copysign(half_width, local_point.x),
            math.copysign(half_height, local_point.y),
        )
        offset = local_point - corner
        if offset.length() >= radius_m:
            return None
        return offset.normalized()

    if max(outside_x, outside_y) >= radius_m:
        return None
    if outside_x > outside_y:
        return Vec2(math.copysign(1.0, local_point.x), 0.0)
    return Vec2(0.0, math.copysign(1.0, local_point.y))


def _make_contact(block, time_fraction, local_centre, local_normal, radius_m,
                  feature="corner"):
    local_point = local_centre - local_normal * radius_m
    return Contact(
        time_fraction=time_fraction,
        point=block.pos + local_point.rotated(block.theta),
        normal=local_normal.rotated(block.theta),
        feature=feature,
    )


# --------------------------------------------------------------------------
# Contact resolution
# --------------------------------------------------------------------------


def resolve_shot_block_contact(shot, block, contact, *, material):
    """Apply normal and tangential impulses. Returns (shot, block).

    Equal and opposite impulses at a single shared contact point, so linear
    and angular momentum are conserved to machine precision by construction.
    Both the `r x j` torque on the block and the one on the shot are carried:
    the shot's is what a glancing hit spins it up with, and dropping it would
    leak angular momentum even though the shot is a sphere.
    """
    normal = contact.normal
    lever_shot = contact.point - shot.pos
    lever_block = contact.point - block.pos

    velocity_shot = shot.vel + lever_shot.perp() * shot.spin
    velocity_block = block.vel + lever_block.perp() * block.omega
    relative = velocity_shot - velocity_block

    approach_ms = relative.dot(normal)
    if approach_ms >= 0.0:
        return shot, block  # already separating

    normal_denominator = _impulse_denominator(shot, block, lever_shot, lever_block, normal)
    restitution = restitution_at(approach_ms, material)
    normal_impulse = -(1.0 + restitution) * approach_ms / normal_denominator
    shot, block = _apply_impulse(
        shot, block, normal * normal_impulse, lever_shot, lever_block
    )

    tangent_velocity = relative - normal * approach_ms
    tangent_speed_ms = tangent_velocity.length()
    if tangent_speed_ms > 1e-12:
        tangent = tangent_velocity * (1.0 / tangent_speed_ms)
        tangent_denominator = _impulse_denominator(
            shot, block, lever_shot, lever_block, tangent
        )
        # Impulse that would exactly kill tangential sliding, then Coulomb.
        tangent_impulse = max(
            -tangent_speed_ms / tangent_denominator,
            -material.friction * normal_impulse,
        )
        shot, block = _apply_impulse(
            shot, block, tangent * tangent_impulse, lever_shot, lever_block
        )

    return shot, block


def _impulse_denominator(shot, block, lever_shot, lever_block, direction):
    shot_term = lever_shot.cross(direction)
    block_term = lever_block.cross(direction)
    return (
        1.0 / shot.mass
        + 1.0 / block.mass
        + shot_term * shot_term / shot.inertia
        + block_term * block_term / block.inertia
    )


def _apply_impulse(shot, block, impulse, lever_shot, lever_block):
    return (
        replace(
            shot,
            vel=shot.vel + impulse * (1.0 / shot.mass),
            spin=shot.spin + lever_shot.cross(impulse) / shot.inertia,
        ),
        replace(
            block,
            vel=block.vel - impulse * (1.0 / block.mass),
            omega=block.omega - lever_block.cross(impulse) / block.inertia,
        ),
    )


# --------------------------------------------------------------------------
# Block against the ground
# --------------------------------------------------------------------------


def step_block(block, dt):
    """Integrate the block one step and resolve its ground contacts.

    THE GROUND MATERIAL COMES OFF THE BLOCK, not off a parameter. It was a
    parameter defaulted to DIABASE_ON_SOIL, which no caller ever overrode; with
    two targets of different substances that default would have quietly given
    the steel crate diabase friction against the soil.

    Semi-implicit Euler, not RK4: the block's motion is dominated by
    discontinuous contact impulses, which a higher-order integrator cannot
    help with and which would make its intermediate stages meaningless.

    Ground contact is resolved per corner rather than at a single point.
    A single-point ground stop is what would stop the block ever tipping or
    rocking correctly: two corners in contact is what produces the flat
    resting state, and one corner in contact is what produces the rocking.
    """
    if _is_settled(block):
        return block

    velocity = block.vel + gravitational_acceleration(block.pos) * dt
    block = replace(
        block,
        vel=velocity,
        pos=block.pos + velocity * dt,
        theta=block.theta + block.omega * dt,
    )
    return _resolve_ground_contacts(block, block.ground_material)


#: How close to the surface a motionless block counts as settled, m.
#: Comfortably above the ~1e-9 m float noise of a 6.4e6 m coordinate and
#: far below anything visible.
SETTLED_TOLERANCE_M = 1e-6


def _is_settled(block):
    """A motionless block resting on the ground leaves the integration set.

    THIS IS A PRECISION REQUIREMENT UNDER THE PLANET-CENTRED FRAME, not an
    optimisation. Positional correction adds a ~1e-9 m quantity to a ~6.4e6 m
    coordinate; the low bits are lost, and repeating that every step for
    30,000 steps integrated into 1.19e-8 m of drift in a block that was
    supposed to be motionless. Gate 15 caught it.

    Reconstructing a canonical pose instead does not work either: the
    arc-distance round trip goes through atan2 and sin/cos and is not
    bit-exact, which left 6.5e-10 m per step, and it also flattened the
    orientation of a block that had toppled onto a different face.

    Returning a settled block UNTOUCHED is exactly idempotent by
    construction - the only version of this that gives exactly zero drift
    rather than a small one. It is the same treatment spent shot already
    get: a body that has stopped is no longer integrated.
    """
    if block.vel.x or block.vel.y or block.omega:
        return False
    deepest_m = min(altitude_m(corner) for corner in block.corners())
    return abs(deepest_m) <= SETTLED_TOLERANCE_M


def _resolve_ground_contacts(block, material):
    touching = False
    pass_normal_impulse = 0.0

    for _ in range(GROUND_RELAXATION_PASSES):
        # Reset each pass: the relaxation passes are re-solving the SAME
        # contact, not applying successive separate ones, so only the final
        # converged pass represents the impulse the ground actually applied
        # over this step.
        pass_normal_impulse = 0.0
        for corner in block.corners():
            if altitude_m(corner) >= 0.0:
                continue
            touching = True

            # The ground normal is the local up AT THIS CORNER, not a global
            # (0, 1) and not even the same as at the block's centre.
            normal = local_up(corner)
            lever = corner - block.pos
            velocity = block.vel + lever.perp() * block.omega
            approach_ms = velocity.dot(normal)
            if approach_ms >= 0.0:
                continue

            restitution = (
                0.0
                if abs(approach_ms) < REST_VELOCITY_THRESHOLD_MS
                else material.restitution
            )
            lever_term = lever.cross(normal)
            denominator = 1.0 / block.mass + lever_term * lever_term / block.inertia
            normal_impulse = -(1.0 + restitution) * approach_ms / denominator
            block = _apply_ground_impulse(block, normal * normal_impulse, lever)

            tangent_velocity = velocity - normal * approach_ms
            tangent_speed_ms = tangent_velocity.length()
            if tangent_speed_ms > 1e-12:
                tangent = tangent_velocity * (1.0 / tangent_speed_ms)
                tangent_term = lever.cross(tangent)
                tangent_denominator = (
                    1.0 / block.mass + tangent_term * tangent_term / block.inertia
                )
                tangent_impulse = max(
                    -tangent_speed_ms / tangent_denominator,
                    -material.friction * normal_impulse,
                )
                block = _apply_ground_impulse(block, tangent * tangent_impulse, lever)

            pass_normal_impulse += normal_impulse

    # Rolling resistance is applied ONCE PER STEP, not once per corner per
    # relaxation pass. It is a dissipative term, not a contact constraint:
    # accumulating it inside the relaxation loop multiplied the angular
    # damping by up to (passes x corners) = 16 and stopped the block ever
    # toppling, even when given four times the energy needed to go over.
    block = _apply_rolling_resistance(block, pass_normal_impulse)

    if touching and (
        block.vel.length() < SLEEP_SPEED_MS
        and abs(block.omega) < SLEEP_OMEGA_RADS
    ):
        block = replace(block, vel=Vec2(0.0, 0.0), omega=0.0)

    # Positional correction: lift the block clear of any residual
    # penetration, along the local up rather than along world +y.
    deepest = min(block.corners(), key=altitude_m)
    penetration_m = altitude_m(deepest)
    if penetration_m < 0.0:
        block = replace(
            block, pos=block.pos + local_up(deepest) * (-penetration_m)
        )
    return block


def _apply_ground_impulse(block, impulse, lever):
    """The ground is an infinite-mass body, so only the block responds."""
    return replace(
        block,
        vel=block.vel + impulse * (1.0 / block.mass),
        omega=block.omega + lever.cross(impulse) / block.inertia,
    )


def _apply_rolling_resistance(block, normal_impulse):
    """Angular impulse opposing rotation, bounded by the normal load.

    Lumped soil term. Bounded so it can only ever remove angular velocity,
    never reverse it and never inject energy.

    THE LEVER ARM IS half_width, NOT half_height. Rolling resistance resists
    rotation about the ground contact, and the horizontal offset from the
    centre of mass to that contact point is the block's HALF-WIDTH - its
    footprint on the ground - regardless of how tall the block stands. A
    square block has these equal, so this distinction was invisible until a
    body with half_width != half_height existed.
    """
    if block.omega == 0.0:
        return block
    resisting = SOIL_ROLLING_RESISTANCE * abs(normal_impulse) * block.half_width
    change = min(abs(block.omega) * block.inertia, resisting) / block.inertia
    return replace(block, omega=block.omega - math.copysign(change, block.omega))


# --------------------------------------------------------------------------
# Shot against the ground
# --------------------------------------------------------------------------


#: Below this outgoing normal speed, a "bounce" cannot be told apart from
#: resting-and-sliding at this step size - the classic bouncing-ball problem:
#: with restitution < 1, resolving bounce by bounce takes infinitely many
#: bounces (in finite total time) to truly stop, and a fixed-dt integrator
#: cannot resolve a bounce shorter than one step. Below this threshold the
#: contact is treated as GRINDING rather than skipping - see below. CHOSEN:
#: about 20x gravity's per-step velocity increment (g * FREE_FLIGHT_DT_S =
#: 9.8 mm/s), so a genuine skip still clears several steps of airborne time
#: and only the decaying tail of a spent ricochet is caught by this.
MIN_SKIP_SPEED_MS = 0.2


def resolve_ground_contact(shot, material, dt=FREE_FLIGHT_DT_S):
    """The shot's contact with the ground, as a MATERIAL response.

    batch 16 Part II SS13. THE GROUND USED TO BE A STOP: whatever crossed
    altitude 0 halted there, forever, regardless of speed or angle. This
    replaces that with the same kind of impulse response the block already
    gets against its own ground contact - selected by the impact conditions,
    not by a flag. See the module-level citations at
    `critical_grazing_angle_deg` for where the angle comes from.

    `dt` is how much time the CALLER intends this contact to account for
    (the remaining budget of its own step, not `step_free_flight`'s bisected
    sub-step to the crossing) - used only by the grinding case below, which
    has no collision impulse to derive a friction bound from and needs a
    real duration to derive one from instead. Passing the bisected sub-step
    here was the original bug: once pinned to the surface, that sub-step
    itself shrinks toward zero (the shot starts each step already AT the
    surface, so gravity re-crosses it almost immediately), so friction
    computed from it vanishes right along with it and NOTHING converges.

    Returns (shot, settled, grinding). Grazing above the critical angle
    buries - `settled=True` immediately, regardless of speed. At or below
    it, the outcome depends on how big the bounce would be:

      - Big enough to clear several steps of airborne time: a genuine skip.
        Restitution and Coulomb friction from `material`, `settled=False`,
        `grinding=False`.
      - Too small to resolve as airborne time at all: GRINDING, not skipping
        - the tail every decaying ricochet has, where the elastic bounce
          model degenerates into the same step re-triggering itself forever
          (verified: this hung gate 26 for tens of thousands of steps
          without converging, before `dt` was fixed to mean this). Pinned to
          the surface, Coulomb friction against the WEIGHT-supported normal
          force over `dt` instead of against this contact's own vanishing
          impulse - the same physical basis the block's own sliding friction
          already uses, just against gravity directly rather than a
          relaxation-solved normal load. `grinding=True` tells the caller
          this consumed the whole of `dt`, not a bisected fraction of it.

    This module does not model burial depth, so "buries" and "ground out a
    grind" are both the same zero-velocity result; the physically meaningful
    fork this simulator can actually show is skip vs. no skip, and that is
    the one the angle test makes.
    """
    # A ROUND AT REST STAYS AT REST - checked before anything else, and as an
    # exact fixed point (re-zeroing, not merely leaving `shot` alone) so
    # calling this again on an already-settled shot is idempotent by
    # construction rather than by coincidence of the branches below.
    if shot.vel.length() < REST_VELOCITY_THRESHOLD_MS:
        return replace(shot, vel=Vec2(0.0, 0.0)), True, False

    normal = local_up(shot.pos)
    approach_ms = shot.vel.dot(normal)
    if approach_ms >= 0.0:
        return shot, False, False  # already separating - nothing to resolve

    speed_ms = shot.vel.length()
    grazing_deg = math.degrees(
        math.asin(min(1.0, max(-1.0, -approach_ms / speed_ms)))
    )
    if grazing_deg > critical_grazing_angle_deg(speed_ms):
        return replace(shot, vel=Vec2(0.0, 0.0)), True, False  # too steep to skip

    tangent_velocity = shot.vel - normal * approach_ms
    tangent_speed_ms = tangent_velocity.length()
    tangent = (
        tangent_velocity * (1.0 / tangent_speed_ms)
        if tangent_speed_ms > 1e-12 else Vec2(0.0, 0.0)
    )

    restitution = (
        0.0 if abs(approach_ms) < REST_VELOCITY_THRESHOLD_MS
        else restitution_at(approach_ms, material)
    )
    rebound_ms = restitution * -approach_ms  # outgoing normal speed, >= 0

    if rebound_ms < MIN_SKIP_SPEED_MS:
        weight_impulse_ms = gravitational_acceleration(shot.pos).length() * dt
        removed_ms = min(tangent_speed_ms, material.friction * weight_impulse_ms)
        new_vel = tangent * (tangent_speed_ms - removed_ms)
        if new_vel.length() < REST_VELOCITY_THRESHOLD_MS:
            return replace(shot, vel=Vec2(0.0, 0.0)), True, True  # ground out
        return (
            replace(shot, pos=shot.pos + normal * 1e-9, vel=new_vel),
            False,
            True,
        )

    delta_normal_ms = -(1.0 + restitution) * approach_ms
    new_vel = shot.vel + normal * delta_normal_ms

    new_tangent_velocity = new_vel - normal * new_vel.dot(normal)
    new_tangent_speed_ms = new_tangent_velocity.length()
    if new_tangent_speed_ms > 1e-12:
        new_tangent = new_tangent_velocity * (1.0 / new_tangent_speed_ms)
        friction_ms = material.friction * delta_normal_ms
        removed_ms = min(new_tangent_speed_ms, friction_ms)
        new_vel = new_vel - new_tangent * removed_ms

    if new_vel.length() < REST_VELOCITY_THRESHOLD_MS:
        return replace(shot, vel=Vec2(0.0, 0.0)), True, False  # rolled out

    # Nudged off the surface along the ground normal so the NEXT step's RK4
    # integration starts from strictly positive altitude - otherwise a bounce
    # whose reflected velocity is nearly tangential could re-cross on the very
    # next step from float noise alone and be resolved twice.
    return replace(shot, pos=shot.pos + normal * 1e-9, vel=new_vel), False, False


# --------------------------------------------------------------------------
# Stepping both bodies together
# --------------------------------------------------------------------------


def step_shot_and_blocks(
    shot,
    blocks,
    dt,
    *,
    drag_enabled=True,
    ground_enabled=True,
    ground_material=CAST_IRON_ON_SOIL,
):
    """Advance the shot and every block by `dt`, with CCD against all of them.

    THE EARLIEST CONTACT WINS. With more than one target the sweep can report a
    hit on two blocks in the same step, and resolving them in list order would
    let the shot strike the far block first whenever the near one happened to
    be later in the list. The whole set is swept, the smallest time fraction is
    taken, and only that block is resolved; the rest of the step then re-sweeps
    from the new state, so a genuine double hit still resolves both - in the
    order they actually occur.

    BLOCK-AGAINST-BLOCK CONTACT IS NOT MODELLED. Two blocks pushed into each
    other pass through. That is batch 10's universal contact and is deliberately
    absent here rather than approximated: the two targets are emplaced apart,
    and a wrong answer would be worse than a missing one.

    Returns (shot, blocks, hit_ground, settled, elapsed_s, impacts) with
    `blocks` a tuple in the order given.

    HIT_GROUND AND SETTLED ARE DIFFERENT CLAIMS - batch 16 Part II SS13.
    `hit_ground` is TOUCHED: the shot crossed the ground surface at least once
    during this call, whether it then skipped onward or stopped - this is the
    same instant the old hard stop used to fire on, and it is what the
    predictor's "first contact" reading needs. `settled` is STOPPED: the shot
    is at rest and must be retired, never integrated again. A shallow, fast
    graze can report `hit_ground=True, settled=False` on one call and only
    settle several calls later.
    """
    blocks = tuple(blocks)
    remaining_s = dt
    elapsed_s = 0.0
    impacts = 0
    ground_events = 0
    touched_ground = False

    def advance_blocks(step_s):
        return tuple(step_block(block, step_s) for block in blocks)

    while (
        remaining_s > 1e-15
        and impacts < MAX_IMPACTS_PER_STEP
        and ground_events < MAX_IMPACTS_PER_STEP
    ):
        candidate, hit_ground, step_s = step_free_flight(
            shot, remaining_s, drag_enabled=drag_enabled,
            ground_enabled=ground_enabled
        )

        earliest_index, earliest = None, None
        for index, block in enumerate(blocks):
            contact = sweep_sphere_vs_block(
                shot.pos, candidate.pos, shot.radius, block
            )
            if contact is not None and (
                earliest is None or contact.time_fraction < earliest.time_fraction
            ):
                earliest_index, earliest = index, contact

        if earliest is None:
            if not hit_ground:
                blocks = advance_blocks(step_s)
                elapsed_s += step_s
                return candidate, blocks, touched_ground, False, elapsed_s, impacts
            touched_ground = True
            # `remaining_s` (BEFORE the bisected sub-step below is charged),
            # not `step_s` - see `resolve_ground_contact` for why the
            # bisected sub-step itself is not a usable duration here.
            shot, settled, grinding = resolve_ground_contact(
                candidate, ground_material, dt=remaining_s
            )
            if grinding:
                # Consumed the WHOLE remaining budget of this call sliding in
                # place, not the tiny bisected fraction step_s represents.
                blocks = advance_blocks(remaining_s)
                elapsed_s += remaining_s
                return shot, blocks, True, settled, elapsed_s, impacts
            blocks = advance_blocks(step_s)
            elapsed_s += step_s
            remaining_s -= step_s
            ground_events += 1
            if settled:
                return shot, blocks, True, True, elapsed_s, impacts
            continue

        impact_s = step_s * earliest.time_fraction
        if impact_s > 0.0:
            # Ground off: this advances to the time of impact with the BLOCK,
            # and a ground stop inside that sub-step would cut it short before
            # the contact is resolved.
            shot, _, _ = step_free_flight(
                shot, impact_s, drag_enabled=drag_enabled, ground_enabled=False
            )
            blocks = advance_blocks(impact_s)

        struck = blocks[earliest_index]
        shot, struck = resolve_shot_block_contact(
            shot, struck, earliest, material=struck.material
        )
        blocks = blocks[:earliest_index] + (struck,) + blocks[earliest_index + 1:]
        elapsed_s += impact_s
        remaining_s -= impact_s
        impacts += 1

    # Only reached if a cap (impacts or ground_events) stopped the loop with
    # time still on the clock - the same "one more step, no further sweeping"
    # tail the original single-target stepper had for the impacts cap alone.
    if remaining_s > 1e-15:
        pre_step_remaining_s = remaining_s
        shot, hit_ground, step_s = step_free_flight(
            shot, remaining_s, drag_enabled=drag_enabled,
            ground_enabled=ground_enabled
        )
        if not hit_ground:
            blocks = advance_blocks(step_s)
            elapsed_s += step_s
            return shot, blocks, touched_ground, False, elapsed_s, impacts
        touched_ground = True
        shot, settled, grinding = resolve_ground_contact(
            shot, ground_material, dt=pre_step_remaining_s
        )
        if grinding:
            blocks = advance_blocks(pre_step_remaining_s)
            elapsed_s += pre_step_remaining_s
        else:
            blocks = advance_blocks(step_s)
            elapsed_s += step_s
        return shot, blocks, True, settled, elapsed_s, impacts

    return shot, blocks, touched_ground, False, elapsed_s, impacts


def step_shot_and_block(shot, block, dt, *, drag_enabled=True,
                        ground_enabled=True, ground_material=CAST_IRON_ON_SOIL):
    """One shot against ONE block. Thin wrapper over `step_shot_and_blocks`.

    Kept because a dozen gates are written against the single-target case and
    reading them is easier without a one-element tuple in every line. It
    delegates rather than duplicating, so there is exactly one contact loop to
    be right.
    """
    shot, blocks, hit_ground, settled, elapsed_s, impacts = step_shot_and_blocks(
        shot, (block,), dt,
        drag_enabled=drag_enabled, ground_enabled=ground_enabled,
        ground_material=ground_material,
    )
    return shot, blocks[0], hit_ground, settled, elapsed_s, impacts
