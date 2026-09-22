"""Loop, input, and the physics accumulator.

    python -m cannon.main                     interactive, windowed
    python -m cannon.main --fullscreen        interactive, fullscreen
    python -m cannon.main --report            headless numbers, no pygame
    python -m cannon.main --report --no-drag --elevation-deg 20 --charge-kg 1.2

Physics runs at a fixed FREE_FLIGHT_DT_S step out of an accumulator. It is
never integrated on the frame clock: a dropped frame or a change of time
scale alters how much simulated time passes per frame, never the step size,
so a shot fired on screen follows exactly the trajectory the headless
harness computes.
"""

import argparse
import math
from dataclasses import replace

from . import focus
from .gun import GRIBEAUVAL_12PDR, IRON_SHOT_12PDR, fire
from .impact import (
    SLAB_HALF_HEIGHT_M,
    SLAB_HALF_WIDTH_M,
    make_slab,
    step_block,
    step_shot_and_blocks,
)
from .physics import FREE_FLIGHT_DT_S, Vec2, integrate_flight
from .planet import (
    MU_EARTH_M3_S2,
    altitude_m,
    downrange_m,
    flight_path_angle_rad,
    position_at,
    surface_tilt_rad,
)
from .settings import (
    ELEVATION_LIMITS_DEG,
    FIELDS,
    SETTINGS_CHARGE_LIMITS_KG,
    TIME_SCALES,
    adjust,
    factory_settings,
    load_settings,
    save_settings,
)

#: A block slower than this in both senses is treated as at rest.
BLOCK_REST_SPEED_MS = 1e-3
BLOCK_REST_OMEGA_RADS = 1e-3

#: ONE TARGET ON THE FIELD - batch 16 SS4. The stone box and steel crate, and
#: the offset that kept them apart, are deleted with the two-target scenario.
#: Multi-target support in `step_shot_and_blocks` stays; the default scenario
#: is a single slab, sighted onto the opening trajectory exactly as the box
#: was. Multiple targets return with batch 10's block-against-block contact,
#: where a second body means something.

#: Wheel zoom, multiplicative per notch. Additive steps are unusable across a
#: 13-decade band.
#:
#: 1.18, not 1.2, by PM judgement at the monitor. What is being judged is the
#: EXCESS OVER UNITY - the notch moves 18% instead of 20%, a tenth less
#: responsive - not the ratio itself, which would have been 1.08.
WHEEL_RATIO = 1.18

#: SHIFT+wheel, for crossing decades deliberately.
WHEEL_COARSE_RATIO = 2.0

#: Notches arriving within this of each other compound, so a deliberate spin
#: covers the band while a single notch stays fine.
WHEEL_ACCEL_WINDOW_S = 0.15

#: Ceiling on the compounded ratio, per notch. Beyond x2 a spin overshoots
#: faster than the eye can follow.
WHEEL_ACCEL_MAX_RATIO = 2.0

#: How far the cursor must move with the left button down before the gesture
#: counts as a drag rather than a click, px.
#:
#: IN PIXELS, NOT TIME. A time threshold makes a slow deliberate click into a
#: drag and a fast flick into a click, which is the wrong axis entirely - what
#: separates the two gestures is whether the cursor went anywhere. Four pixels
#: is about the hand tremor on a press.
DRAG_THRESHOLD_PX = 4.0

#: Nearest the gun a target may be dragged, m of arc.
#:
#: Batch 9 SS7. Below a few metres the target is inside the muzzle blast and
#: on top of the carriage geometry, and at zero it is inside the gun itself -
#: where the shot's first CCD sweep starts already overlapping it and the
#: contact normal is whichever face the rounding picks.
MIN_TARGET_DOWNRANGE_M = 5.0



def _block_is_moving(block):
    return (
        block.vel.length() > BLOCK_REST_SPEED_MS
        or abs(block.omega) > BLOCK_REST_OMEGA_RADS
    )


#: How far above the ground a permanently-static body is still "resting on
#: it", m - not zero, because a resting body's lowest point sits at exactly
#: altitude 0 only up to float precision at a ~6.371e6 m coordinate. Well
#: above that noise floor and far below anything a body could visibly float.
GROUNDED_TOLERANCE_M = 1e-3


def _assert_no_floating_static_bodies(spent_shots, blocks):
    """The gravity/solidity invariant - batch 16 Part II SS14.

    THE POINT OF THIS CHECK, NOT THE R KEY. R retiring a still-moving round
    into `spent_shots` was one way to produce this; the claim is general and
    does not care how a body got here. A body this program draws every frame
    but never integrates again is an implicit assertion that its velocity
    AND acceleration are permanently zero - and that is only physically true
    while the body is touching the ground, because gravity does not stop
    applying just because nothing is stepping the body. A permanently static
    body sitting above the ground, unsupported, is the impossible state the
    R-mid-flight freeze put on screen, and this is the general form of it.

    Plain `assert`, not a custom flag: stripped under `-O`, loud otherwise -
    exactly "assert, loudly, in development" as briefed. It runs in the sim
    loop itself, every frame, not only in tests.
    """
    for shot in spent_shots:
        assert altitude_m(shot.pos) <= GROUNDED_TOLERANCE_M, (
            f"IMPOSSIBLE STATE: a spent round at {downrange_m(shot.pos):.1f} m "
            f"downrange sits {altitude_m(shot.pos):.3f} m above the ground, "
            f"unsupported, and is never integrated again"
        )
    for block in blocks:
        if block.vel.length() == 0.0 and block.omega == 0.0:
            deepest_m = min(altitude_m(corner) for corner in block.corners())
            assert deepest_m <= GROUNDED_TOLERANCE_M, (
                f"IMPOSSIBLE STATE: {block.label} reads at rest but its "
                f"lowest corner is {deepest_m:.3f} m above the ground - "
                f"nothing is holding it up and nothing is integrating it"
            )


def target_home_x(settings, gun, shot, charge_kg, elevation_rad, *, drag_enabled):
    """Where the block is emplaced at startup. Returns (centre_x or None, path).

    Auto only. The explicit target-distance setting is gone: the block is
    sighted onto the opening trajectory here, and moved afterwards by
    DRAGGING IT, which is a direct manipulation rather than a number to
    maintain. That removes a setting rather than adding one.
    """
    if not settings.target_enabled:
        return None, "disabled"
    return (
        sighted_block_x(
            gun, shot, charge_kg, elevation_rad, drag_enabled=drag_enabled
        ),
        "auto (trajectory solve)",
    )


def predicted_impact_position(state, drag_enabled, blocks=()):
    """Where a shot in this state reaches its FIRST CONTACT. None if it never does.

    FIRST CONTACT, GROUND OR BLOCK, WHICHEVER COMES FIRST - SS11(a). This used to
    solve against the ground alone, through `integrate_flight`, while the live
    round went through `step_shot_and_blocks` and its CCD sweep. Two models of
    one question, which is the same failure as the two cameras and the two
    thick-line paths: aimed at the stone box the marker sat 0.73 m past the near
    face, inside the block's own footprint, while the round struck the face and
    rebounded.

    The process that computes the landing point can already see what is in the
    path, so it uses the same sweep the round does.

    PM decision, reversible: with a block in the path this marks first contact,
    not the resting place after rebound. First contact is what a gunner wants.

    THE MARKER IS A PURE FUNCTION OF ITS ARGUMENTS. Callers must pass a laying
    and the field, never a live shot's state - see the loop, where writing this
    from an in-flight round is what made the marker persist after the field was
    cleared.
    """
    blocks = tuple(blocks)
    if not blocks:
        flight = integrate_flight(
            state,
            dt=PREDICTION_DT_S,
            drag_enabled=drag_enabled,
            max_time_s=400.0,
            # One sample at each end: the caller wants the impact, not the
            # path, and collecting the path would allocate tens of thousands of
            # states ten times a second for nothing.
            sample_interval_s=1e9,
        )
        return flight.impact.pos if flight.impact else None

    # With something in the path, run the round's own sweep. Coarser step than
    # the live round for the same reason as above; the disagreement that buys is
    # one step of travel, which gate 115 bounds rather than assumes.
    for _ in range(int(400.0 / PREDICTION_DT_S)):
        state, blocks, hit_ground, _settled, _step_s, impacts = (
            step_shot_and_blocks(
                state, blocks, PREDICTION_DT_S, drag_enabled=drag_enabled
            )
        )
        if impacts or hit_ground:
            return state.pos
    return None


def prediction_key(elevation_deg, charge_kg, drag_enabled, blocks):
    """What the predicted-impact marker depends on, and nothing else.

    THE CACHE KEY IS THE PURITY STATEMENT. The old key was
    (elevation, charge, drag) and it did not include the field - so dragging a
    target left a stale marker - while a separate in-flight branch overwrote the
    marker from the LIVE SHOT STATE every 100 ms. Shot state is not in this
    tuple and must never be: that is what made the marker persist after the
    field was cleared and the gun reloaded.

    Block positions are quantised to a centimetre so a target settling after a
    hit does not retrigger the solve every frame. A centimetre is far below the
    marker's own drawn size at any usable zoom.
    """
    return (
        round(elevation_deg, 9),
        round(charge_kg, 9),
        bool(drag_enabled),
        tuple(round(downrange_m(block.pos), 2) for block in blocks),
    )


def compute_prediction(gun, shot, charge_kg, elevation_deg, drag_enabled,
                       blocks):
    """The marker, from the laying and the field. Never from a live round."""
    aimed, _ = fire(gun, shot, charge_kg, math.radians(elevation_deg))
    return predicted_impact_position(aimed, drag_enabled, blocks=blocks)


def pan_key_codes():
    """Name -> pygame keycode for the pan bindings.

    A FUNCTION, NOT A MODULE CONSTANT, because pygame is imported inside `run`
    and never at module scope - the headless --report path must not pull it in.
    """
    import pygame

    return {
        "K_i": pygame.K_i, "K_j": pygame.K_j,
        "K_k": pygame.K_k, "K_l": pygame.K_l,
    }


#: The four held-key pan bindings, as (name, dx, dy) in screen pixels.
PAN_KEYS = (
    ("K_j", -1.0, 0.0),
    ("K_l", 1.0, 0.0),
    ("K_i", 0.0, -1.0),
    ("K_k", 0.0, 1.0),
)


def apply_pan_keys(camera, held, pan_px, key_codes):
    """Apply one frame of held-key pan. Returns True if any key was down.

    EXTRACTED SO IT CAN BE GATED. The loop read `pygame.key.get_pressed()`
    inline, and posted KEYDOWN events do not affect OS key state - so no gate
    could reach this path, and the first one that tried (the keyboard case for
    Tranche 3 SS3) failed against correct code. `held` is any object indexable by
    keycode, which a gate can supply as a dict.

    IF YOU MOVE THE CAMERA YOURSELF, IT STOPS TRACKING - Tranche 3 SS3. The drop
    happens once, before any nudge, so a held key does not fight the follow
    logic frame by frame.
    """
    moving = [
        (dx, dy) for name, dx, dy in PAN_KEYS if held[key_codes[name]]
    ]
    if not moving:
        return False
    camera.drop_follow_in_place()
    for dx, dy in moving:
        camera.move_camera_by_pixels(dx * pan_px, dy * pan_px)
    return True


def focus_on_camera(camera, target):
    """Point the camera at a focus target. Returns the target unchanged.

    THE SEAM THE GATE DRIVES. The loop's own `focus_on` closure delegates here,
    so gate 107 exercises what actually runs rather than calling the camera
    itself - which would pass against a loop that never called it, and that is
    precisely the defect being fixed.

    FOLLOW IS DROPPED. The view is being pointed somewhere by hand; a follow
    camera would take it straight back on the next frame.
    """
    if target is None:
        return None
    camera.set_follow(False)
    camera.set_anchor(target.position_m, anchor_frac=target.anchor_frac)
    return target


def grab_block(rect_px, mouse_px):
    """Is the cursor on the block's DRAWN extent?

    Takes the rect rather than the camera so this module never has to import
    the render layer, which would drag pygame into the headless `--report`
    path. `render.block_screen_rect` supplies it.

    Grabbing works against what is on screen, not against true size. At low
    zoom the block is sub-pixel and is drawn at the 4 px minimum-size floor,
    so the floor is what has to be grabbable - otherwise the block becomes
    unclickable at exactly the zoom where you most need to place it.
    """
    left, top, width, height = rect_px
    return (
        left <= mouse_px[0] <= left + width
        and top <= mouse_px[1] <= top + height
    )


def block_dragged_to(block, camera, mouse_px):
    """The block repositioned under the cursor. Ground-bound, so x only.

    Returns a block at rest: dragging places it, it does not throw it.

    CLAMPED AT MIN_TARGET_DOWNRANGE_M. Dragging a target back onto the gun put
    it inside the muzzle and inside the carriage, where the first CCD sweep
    begins already overlapping the box and the contact normal is decided by
    rounding rather than by geometry. The clamp is on ARC DISTANCE, the same
    measure the placement uses, so it holds at any zoom and anywhere on the
    curve.
    """
    world = Vec2(
        camera.center_m.x
        + (mouse_px[0] - 0.5 * camera.viewport_px[0]) / camera.scale_px_per_m,
        camera.center_m.y
        - (mouse_px[1] - 0.5 * camera.viewport_px[1]) / camera.scale_px_per_m,
    )
    downrange = max(MIN_TARGET_DOWNRANGE_M, downrange_m(world))
    return replace(
        block,
        # HALF_HEIGHT, NOT HALF_WIDTH. `pos` sits at an altitude equal to the
        # block's VERTICAL half-extent, so its base touches the ground - for a
        # square body the two were interchangeable; for the slab only the
        # height governs where the base is.
        pos=position_at(downrange, block.half_height),
        theta=surface_tilt_rad(downrange),
        vel=Vec2(0.0, 0.0),
        omega=0.0,
    )


def _log_emplacement(path, enabled, block):
    """Diagnostic line printed every time the block is emplaced."""
    if block is None:
        print(f"emplace: enabled={enabled} path={path} block=None")
        return
    print(
        f"emplace: enabled={enabled} path={path} label={block.label!r} "
        f"pos=({block.pos.x:.6f}, {block.pos.y:.6f}) "
        f"half_width={block.half_width:.4f} half_height={block.half_height:.4f} "
        f"theta={block.theta:.6f} "
        f"mass={block.mass:.1f} kg inertia={block.inertia:.1f} kg m^2"
    )


def _elevation_step(settings, fine):
    return (
        settings.elevation_fine_step_deg if fine
        else settings.elevation_coarse_step_deg
    )


def sighted_block_x(gun, shot, charge_kg, elevation_rad, *, drag_enabled):
    """Where to emplace the block so the opening shot strikes it squarely.

    Returns the block centre's x, or None if the shot never gets there.

    Run ONCE, for the load the simulator starts on, so the first round
    connects without anyone having to range the gun first. It is NOT re-run
    when elevation or charge change: past the opening shot, laying the gun
    onto the target is the exercise, and a block that re-sighted itself
    would remove it.

    The block is placed by where the DESCENDING trajectory crosses the
    block's centre HEIGHT, offset by one shot radius plus the block's
    half-WIDTH, so the sphere's surface meets the middle of the near face.
    Placing it instead at the ground impact point - which is what the
    stage-4 brief asks for - gives a top-face hit at any useful elevation,
    because the shot arrives 71.6 degrees nose-down at 45 degrees.

    TWO DIFFERENT HALF-EXTENTS, DELIBERATELY. The vertical target the
    trajectory is aimed at is the block's half-HEIGHT - for a square body
    the two were numerically identical and this distinction was invisible.
    For the slab they are not: aiming at half-WIDTH (12.5 mm) would target a
    point almost on the ground, missing the plate's actual centre by nearly
    1.5 m.
    """
    predicted, _ = fire(gun, shot, charge_kg, elevation_rad)
    flight = integrate_flight(
        predicted, drag_enabled=drag_enabled, sample_interval_s=FREE_FLIGHT_DT_S
    )
    half_height_m = SLAB_HALF_HEIGHT_M
    half_width_m = SLAB_HALF_WIDTH_M

    previous = None
    for _, sample in flight.samples:
        previous_alt = altitude_m(previous.pos) if previous is not None else None
        sample_alt = altitude_m(sample.pos)
        if (
            previous is not None
            and sample_alt < previous_alt          # descending
            and sample_alt <= half_height_m <= previous_alt
        ):
            span_m = previous_alt - sample_alt
            fraction = (previous_alt - half_height_m) / span_m if span_m else 0.0
            crossing = previous.pos + (sample.pos - previous.pos) * fraction
            # Arc distance, not Cartesian x: the offsets are along the
            # surface, so they have to be added in the same measure.
            return downrange_m(crossing) + shot.radius_m + half_width_m
        previous = sample
    return None

WINDOW_PX = (1280, 720)
TARGET_FPS = 60

#: Ceiling on simulated time advanced per frame, s. Stops a stall from
#: turning into an unbounded catch-up loop.
MAX_SIM_PER_FRAME_S = 0.25

#: Runtime charge range, kg. UNIFIED with the Settings clamp - the two were
#: 0.2-4.0 and 0.4-3.6 respectively, so a user could set the default to the
#: clamp floor and then step below it at runtime, out of the band the clamp
#: exists to enforce.
#:
#: 0.4 kg IS ITSELF AT THE EDGE OF VALIDITY. The figure for where the
#: interior model stops meaning anything is ~0.5 kg - below that there is no
#: bore-friction term and no windage-leakage floor, so the gas drives a
#: frictionless shot down a sealed tube and reports a muzzle velocity it
#: cannot justify. The factory default now sits BELOW that figure. It stays
#: there until those terms exist, at which point this floor should be
#: revisited deliberately rather than inherited.
CHARGE_LIMITS_KG = SETTINGS_CHARGE_LIMITS_KG

#: Physics steps between trail samples. At dt = 1 ms this is a point every
#: 20 ms of flight, which is smooth at any zoom without unbounded growth.
TRAIL_STRIDE = 20

#: Step size for the predicted-landing integration, s. Coarser than the
#: 1e-3 s the shot actually flies at, deliberately: the prediction is redrawn
#: continuously and a full-fidelity re-integration every frame would cost
#: more than the whole rest of the loop. At 5e-3 s the predicted impact is
#: within a few centimetres of the flown one, which is far inside the
#: marker's own width on screen.
PREDICTION_DT_S = 5e-3


#: Keyboard pan rate, SCREEN pixels per second. Constant in screen space on
#: purpose: panning then feels identical at every zoom, where a rate in
#: metres per second would crawl at map scale and fly at the muzzle.
PAN_RATE_PX_S = 700.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Smoothbore ballistics simulator")
    parser.add_argument("--report", action="store_true",
                        help="print a headless trajectory summary and exit")
    # Default None so the saved settings can supply the value; --report
    # falls back to the factory figures.
    parser.add_argument("--elevation-deg", type=float, default=None)
    parser.add_argument("--charge-kg", type=float, default=None)
    parser.add_argument("--no-drag", action="store_true",
                        help="disable atmosphere and drag (vacuum trajectory)")
    parser.add_argument("--fullscreen", action="store_true",
                        help="start fullscreen instead of windowed; overrides "
                             "the saved setting")
    parser.add_argument("--settings", default=None,
                        help="path to the settings file (default: "
                             "%%APPDATA%%\\cannon\\settings.json)")
    return parser.parse_args(argv)


def report(args):
    """Stage 1 / stage 2 text output. Imports no view code."""
    gun = GRIBEAUVAL_12PDR
    shot = IRON_SHOT_12PDR
    defaults = factory_settings()
    elevation_deg = (
        args.elevation_deg if args.elevation_deg is not None else defaults.elevation_deg
    )
    charge_kg = args.charge_kg if args.charge_kg is not None else defaults.charge_kg
    state, bore = fire(gun, shot, charge_kg, math.radians(elevation_deg))
    flight = integrate_flight(state, drag_enabled=not args.no_drag)

    print(f"{gun.name}, {shot.name}")
    print(f"  charge {charge_kg:.2f} kg   elevation {elevation_deg:.1f} deg"
          f"   drag {'off' if args.no_drag else 'on'}")
    print("  interior ballistics")
    print(f"    ignition pressure   {bore.peak_pressure_pa / 1e6:9.2f} MPa (fitted p0)")
    print(f"    muzzle pressure     {bore.muzzle_pressure_pa / 1e6:9.2f} MPa")
    print(f"    barrel transit      {bore.transit_time_s * 1e3:9.3f} ms")
    print(f"    muzzle velocity     {bore.muzzle_speed_ms:9.2f} m/s")
    print(f"    muzzle energy       "
          f"{0.5 * shot.mass_kg * bore.muzzle_speed_ms ** 2 / 1e3:9.1f} kJ")
    print(f"  exterior ballistics (RK4, dt = {FREE_FLIGHT_DT_S * 1e3:.1f} ms)")
    print(f"    peak Mach           {flight.max_mach:9.3f}")
    print(f"    apex                {flight.apex_height_m:9.1f} m "
          f"at {flight.apex_range_m:.0f} m downrange")

    if flight.impact is None:
        print("    no impact within the integration window")
        return 1

    print(f"    range               {flight.impact.pos.x:9.1f} m")
    print(f"    flight time         {flight.flight_time_s:9.2f} s")
    print(f"    impact speed        {flight.impact.speed_ms:9.1f} m/s")
    print(f"    impact angle        "
          f"{math.degrees(math.atan2(-flight.impact.vel.y, flight.impact.vel.x)):9.1f}"
          f" deg below horizontal")
    return 0


def run(args):
    """Interactive loop. pygame is imported here and nowhere in the physics."""
    import pygame

    from .render import (
        Camera, Hud, SETTINGS_ACTIONS,
        block_screen_rect, draw_scene, draw_settings,
    )

    settings = load_settings(args.settings)

    pygame.init()
    pygame.display.set_caption("cannon - Gribeauval 12-pounder")
    # Windowed is the startup default; --fullscreen or the saved setting opts
    # in. Desktop-resolution fullscreen, so there is no mode switch and the
    # camera keeps its px/m scale.
    fullscreen = args.fullscreen or settings.start_fullscreen
    screen = pygame.display.set_mode(
        (0, 0) if fullscreen else WINDOW_PX,
        pygame.FULLSCREEN if fullscreen else pygame.RESIZABLE,
    )
    clock = pygame.time.Clock()
    fonts = {
        "mono": pygame.font.SysFont("consolas,dejavusansmono,couriernew", 17),
        "small": pygame.font.SysFont("consolas,dejavusansmono,couriernew", 13),
    }

    gun = GRIBEAUVAL_12PDR
    shot = IRON_SHOT_12PDR

    # CLI arguments override the saved defaults for this run only.
    elevation_deg = (
        args.elevation_deg if args.elevation_deg is not None else settings.elevation_deg
    )
    charge_kg = args.charge_kg if args.charge_kg is not None else settings.charge_kg
    drag_enabled = settings.drag_enabled and not args.no_drag
    time_scale_index = TIME_SCALES.index(settings.time_scale)

    # ONE CAMERA - batch 16 SS1. There were two, with two scales and two
    # disjoint bands, and M swapped which was looking; the snapshot/restore
    # dance that kept them consistent is gone with them.
    #
    # Anchored on the trunnions, not the muzzle: the trunnions are the pivot
    # and so do not move as the piece is elevated.
    camera = Camera(
        screen.get_size(), gun.trunnion_m,
        scale_px_per_m=settings.zoom_default_px_per_m,
        follow=settings.camera_follow,
    )
    # M IS AN OVERLAY TOGGLE, NOT A MODE. It shows or hides the osculating
    # conic, apsides and ground crossings. It does not touch the camera and it
    # does not switch representations - each object decides how to draw itself
    # from its own drawn size.
    overlay_conic = False
    # The camera anchor is a body REFERENCE. `focus_key` outlives the bodies it
    # names: `focus.find` falls back to the gun when the shot it pointed at has
    # been retired.
    focus_key = "gun"

    def focus_on(key):
        """Select a body AND reframe on it. Returns the target, or None.

        CHANGING FOCUS MOVES THE VIEW - SS2, reversing the batch 15 ruling that
        made focus a selection and HOME the movement. The evidence against that
        ruling was the PM's: TAB changed a variable and nothing happened on
        screen, so the display stayed welded to the cannon.

        The objection I raised then - that clicking a target while laying the
        gun must not throw the camera - is answered by separating the GESTURES,
        not by making focus inert. A click selects and reframes; a drag past the
        4 px threshold moves the target and never reframes. SS7 built that split.

        FOLLOW IS DROPPED, for the same reason a drag drops it: the view is being
        pointed somewhere by hand, and a follow camera would take it straight
        back on the next frame.
        """
        nonlocal focus_key
        focus_key = key
        return focus_on_camera(camera, focus.find(focus_targets(), focus_key))

    def focus_targets():
        """The focus ring for the world as it stands this frame.

        Rebuilt on demand rather than cached: bodies come and go - a shot
        appears on fire and becomes a spent round on impact - and a cached ring
        would name bodies that no longer exist. `focus.find` falls back to the
        gun when the focused body has gone.
        """
        return focus.targets(
            gun, blocks=blocks, shot=state, spent_shots=spent_shots
        )

    state = None
    bore = None
    # ONE TARGET, batch 16 SS4: the steel slab, on the opening trajectory.
    # Still held as a list - `step_shot_and_blocks` and the drag/focus code all
    # work on the plural set - so a second target is a data change, not a
    # structural one, when batch 10 makes multiple targets meaningful.
    # `home_downrange` holds where C returns each of them.
    blocks = []
    home_downrange = []
    # SPENT SHOT ARE INERT SCENERY. Once a round comes to rest it leaves the
    # dynamic set entirely: never integrated again, never contact-tested.
    # Cost per spent round is one draw call, so the field scales to any
    # number of rounds. This is the recommended default pending the PM's
    # decision, not a permanent choice - and it is a list of full ShotState
    # precisely so the collidable version is reachable without a rewrite.
    # That version would step this list alongside the block and needs
    # ball-on-ball contact, which does not exist yet and is a stage of its
    # own. Nothing below assumes the list stays static.
    spent_shots = []
    in_flight = False
    flight_time_s = 0.0
    accumulator_s = 0.0
    step_count = 0
    trail = []
    status = "ready"
    settings_open = False
    settings_index = 0
    settings_dirty = False
    #: Index into `blocks` of the target being dragged, or None. An index
    #: rather than a flag now that there is more than one target.
    dragging_index = None
    #: Where the left button went down, and what was under it. A left press is
    #: not yet a click or a drag - motion past DRAG_THRESHOLD_PX decides which.
    left_press_px = None
    left_press_dragged = False
    left_press_index = None
    #: The world point currently held under the cursor by a grab-pan, and when
    #: the last wheel notch arrived (for compounding).
    grab_world_m = None
    last_wheel_s = -1e9
    wheel_streak = 0
    predicted_landing = None
    #: What the marker was last computed for. See `prediction_key`.
    last_prediction_key = None

    def emplace_targets():
        """Sight the slab onto the opening trajectory.

        The solve runs ONCE, for the load the simulator starts on, so the first
        round connects without anyone ranging the gun first. It is not re-run
        when elevation or charge change - laying the gun onto the target is the
        exercise, and a target that re-sighted itself would remove it.
        """
        nonlocal blocks, home_downrange
        sighted_m, path = target_home_x(
            settings, gun, shot, charge_kg,
            math.radians(elevation_deg), drag_enabled=drag_enabled,
        )
        if sighted_m is None:
            blocks, home_downrange = [], []
            _log_emplacement(path, settings.target_enabled, None)
            return
        home_downrange = [sighted_m]
        blocks = [make_slab(home_downrange[0])]
        for block in blocks:
            _log_emplacement(path, settings.target_enabled, block)

    def apply_settings():
        """Adopt the saved defaults. Only ever called from clear_field()."""
        nonlocal elevation_deg, charge_kg, drag_enabled, time_scale_index
        elevation_deg = settings.elevation_deg
        charge_kg = settings.charge_kg
        drag_enabled = settings.drag_enabled
        time_scale_index = TIME_SCALES.index(settings.time_scale)
        camera.set_scale(settings.zoom_default_px_per_m)
        camera.set_follow(settings.camera_follow)

    def retire_shot():
        """Move a round that has genuinely come to rest onto the field.

        ONLY CALLED WHEN `settled` IS TRUE - batch 16 Part II SS14. A body
        added here has zero velocity and is resting on a surface, so it is
        safe as permanent, un-integrated scenery: nothing about it will ever
        need to move again. This is NOT the function for "the player wants
        the active round gone" - see `discard_shot`, which is the one that
        handles a round that has not actually stopped.
        """
        nonlocal state, in_flight
        if state is not None:
            spent_shots.append(state)
        state = None
        in_flight = False

    def discard_shot():
        """Drop the active round outright - it is gone, not parked.

        batch 16 Part II SS14. R used to route through `retire_shot`, which
        added whatever `state` was to the drawn-forever `spent_shots` list
        regardless of whether it had actually come to rest. Pressed mid-
        flight, that put a body with real velocity into a list nothing ever
        integrates again - drawn every frame at its R-press position and
        velocity, forever, which is the R-freeze the PM reported. A round
        this function discards was never resolved to be at rest, so it must
        not become scenery: `state` is simply cleared, `spent_shots` is
        untouched, and the round leaves the world instead of freezing in it.
        """
        nonlocal state, in_flight
        state = None
        in_flight = False

    def clear_round():
        """R. Discard the active round and clear the trail.

        Elevation, charge and target positions are all unchanged - this is not
        re-aiming and it is not a reset.

        R used to load as well, which is why loading was invisible. Batch 15 SS3
        makes the load physical: once the carriage recoils, running the piece
        back up to the firing position is what sets the rate of fire, and a key
        that silently re-loaded would hide the constraint the whole exercise is
        about. Loading is A, and only A.
        """
        nonlocal flight_time_s, accumulator_s, step_count, status
        discard_shot()
        flight_time_s = 0.0
        accumulator_s = 0.0
        step_count = 0
        trail.clear()
        camera.reframe()
        status = "round cleared"

    def clear_field():
        """C. Remove all spent shot and return the targets to where they lie.

        This is the reset analogue, so pending Settings changes land here.
        They cannot land on R: R clears the bore, and re-applying the default
        elevation and charge every time the bore was cleared would re-lay the
        gun for the user - exactly what stage 4 established must not happen.
        """
        nonlocal state, bore, blocks, in_flight, settings_dirty
        nonlocal flight_time_s, accumulator_s, step_count, status
        state = None
        bore = None
        in_flight = False
        flight_time_s = 0.0
        accumulator_s = 0.0
        step_count = 0
        trail.clear()
        spent_shots.clear()
        camera.reframe()
        if settings_dirty:
            apply_settings()
            settings_dirty = False
            emplace_targets()
        else:
            # Back to where it was last laid. Rebuilt from the factory rather
            # than from a stored copy, so a dragged slab comes back with its
            # true derived mass and inertia rather than a stale snapshot.
            blocks = [make_slab(downrange) for downrange in home_downrange]
        status = "field cleared"

    emplace_targets()

    def zoom_by(factor, anchor_px=None):
        """Zoom about a screen point, easing in over the next few frames.

        CURSOR-ANCHORED, not centre-anchored. Scaling about the screen centre
        slides whatever you were looking at away, so you chase it with the pan.

        WITH NOTHING UNDER THE CURSOR - a keyboard zoom, or a wheel event with
        the pointer outside the viewport - the anchor is the FOCUSED BODY's
        screen position, not the centre. Keyboard zoom should hold the thing you
        are looking at, and after batch 16 SS1 the focused body is what that
        means.
        """
        if anchor_px is None:
            target = focus.find(focus_targets(), focus_key)
            anchor_px = (
                camera.to_screen(target.position_m) if target is not None
                else (0.5 * screen.get_width(), 0.5 * screen.get_height())
            )
        camera.request_zoom(factor, anchor_px)

    def begin_grab(point_px):
        """Start a grab-pan at `point_px`. Returns the world point grabbed.

        DROPS FOLLOW. Under direct manipulation, grabbing the world while the
        camera tracks a shot must do something rather than nothing - the gesture
        wins, and follow does not silently resume. That reverses gate 80's
        inert-while-following case, deliberately.
        """
        camera.drop_follow_in_place()
        return camera.to_world(point_px)

    def close_settings():
        nonlocal settings_open
        settings_open = False
        save_settings(settings, args.settings)

    def handle_settings_key(event):
        """Modal key handling. Returns False if the screen just closed."""
        nonlocal settings, settings_index, settings_dirty
        rows = len(FIELDS) + len(SETTINGS_ACTIONS)

        # ESC ONLY. S opens this screen; batch 15 SS0 removed it as a second
        # way to close, because the footer can only honestly advertise one.
        if event.key == pygame.K_ESCAPE:
            close_settings()
            return False
        if event.key == pygame.K_UP:
            settings_index = (settings_index - 1) % rows
        elif event.key == pygame.K_DOWN:
            settings_index = (settings_index + 1) % rows
        elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
            if settings_index < len(FIELDS):
                direction = 1 if event.key == pygame.K_RIGHT else -1
                settings = adjust(
                    settings, FIELDS[settings_index], direction, fine
                )
                settings_dirty = True
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            action = settings_index - len(FIELDS)
            if action == 0:  # reset to factory defaults
                settings = factory_settings()
                settings_dirty = True
            elif action == 1:  # close and save
                close_settings()
                return False
        return True

    def toggle_fullscreen():
        nonlocal screen, fullscreen
        fullscreen = not fullscreen
        if fullscreen:
            # Desktop-resolution fullscreen: no mode switch, so the camera
            # keeps its px/m scale and only the framing widens.
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            screen = pygame.display.set_mode(WINDOW_PX, pygame.RESIZABLE)
        camera.set_viewport(screen.get_size())

    panning_with_mouse = False

    PAN_KEY_CODES = pan_key_codes()

    running = True
    while running:
        frame_dt_s = clock.tick(TARGET_FPS) / 1000.0
        mods = pygame.key.get_mods()
        fine = bool(mods & pygame.KMOD_SHIFT)

        # Held-key pan, read from the keyboard state rather than from
        # KEYDOWN events so it repeats smoothly instead of stuttering on the
        # OS key-repeat delay.
        if not settings_open:
            held = pygame.key.get_pressed()
            pan_px = PAN_RATE_PX_S * frame_dt_s
            # IJKL MOVE THE CAMERA; content slides opposite. J is camera
            # left, so content slides right - and a rightward mouse drag
            # slides content right too. The two models agree on screen.
            #
            # I and K are INERT IN SURFACE MODE: it is a horizontal view
            # along a surface, and there is nothing above or below to reach.
            # Map mode is fully 2D and uses all four.
            # I AND K ARE ACTIVE EVERYWHERE - batch 16 SS1. Batch 13 SS0 made
            # them inert in the surface view, reasoning that a horizontal view
            # has nothing above or below to reach. That was wrong when written
            # (the shot arcs) and is meaningless now there is one band and no
            # surface mode for a control to be inert in.
            apply_pan_keys(camera, held, pan_px, PAN_KEY_CODES)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE and not fullscreen:
                screen = pygame.display.set_mode(
                    (event.w, event.h), pygame.RESIZABLE
                )
                camera.set_viewport(screen.get_size())
            elif event.type == pygame.MOUSEWHEEL and not settings_open:
                # MULTIPLICATIVE, WITH ACCELERATION ON A RAPID SPIN. At a flat
                # x1.2 the 13.1-decade band takes about 165 notches; compounding
                # inside a 150 ms window lets a deliberate spin cross it while a
                # single notch is still a single notch.
                now_s = pygame.time.get_ticks() / 1000.0
                if now_s - last_wheel_s < WHEEL_ACCEL_WINDOW_S:
                    wheel_streak += 1
                else:
                    wheel_streak = 0
                last_wheel_s = now_s
                if fine:                      # SHIFT: the coarse decade step
                    ratio = WHEEL_COARSE_RATIO
                else:
                    ratio = min(
                        WHEEL_RATIO * (1.0 + 0.35 * wheel_streak),
                        WHEEL_ACCEL_MAX_RATIO,
                    )
                zoom_by(ratio ** event.y, pygame.mouse.get_pos())

            # ---- MOUSE ARBITRATION, batch 16 tranche 2 SS3 ----
            #
            # ONE RULE, NOT A PER-HANDLER DECISION, and crucially not an elif
            # chain where exactly one branch runs per event.
            #
            # A PRESS is routed by button: right starts a pan, left starts a
            # potential select-or-drag. A RELEASE is routed by WHAT IS LATCHED,
            # not by which button arrived, because one release may have to
            # finish more than one thing. The old chain put the button-3 release
            # above the drag release, so releasing right during a left-drag left
            # the target latched to the cursor forever - measured in Report 1 at
            # -1.600 m of travel with no button held.
            #
            # That is the same shape as batch 8's positional bind: two copies of
            # one concept - "which gesture is active" - held in the branch
            # structure instead of in state.
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and not settings_open
            ):
                if event.button == 3:
                    # RIGHT-DRAG ALWAYS PANS and never hit-tests anything.
                    panning_with_mouse = True
                    grab_world_m = begin_grab(event.pos)
                elif event.button == 1:
                    # Left press: remember where, and what was under it. Whether
                    # this becomes a click or a drag is decided on motion.
                    left_press_px = event.pos
                    left_press_dragged = False
                    left_press_index = None
                    if not in_flight:
                        for index, block in enumerate(blocks):
                            if grab_block(
                                block_screen_rect(camera, block), event.pos
                            ):
                                left_press_index = index
                                break
                    # LEFT-DRAG ON EMPTY SPACE PANS - the addendum's SS5, and
                    # SFS's background drag. The common gesture belongs under
                    # the dominant finger. A press ON a target still moves the
                    # target; the pixel threshold below still separates either
                    # drag from a click.
                    if left_press_index is None:
                        grab_world_m = begin_grab(event.pos)
            elif event.type == pygame.MOUSEMOTION:
                # GRAB-PAN: the world point taken on button-down stays under the
                # cursor. THE SIGN IS NO LONGER A DECISION - it falls out of the
                # transform, so it cannot be inverted again the way batch 13 SS0
                # inverted it, and it is 1:1 at every scale with no gain
                # constant because it is expressed in world coordinates.
                #
                # NEVER SMOOTHED. Direct manipulation has to be instantaneous or
                # it feels like ice; the glide belongs to zoom alone.
                dragging_view = panning_with_mouse or (
                    left_press_dragged and left_press_index is None
                )
                if dragging_view and grab_world_m is not None:
                    camera.hold(grab_world_m, event.pos)
                if left_press_px is not None and not left_press_dragged:
                    # A DRAG IS A PRESS PLUS MOTION PAST A THRESHOLD, in pixels
                    # and not in time. Below it the gesture is still a click, so
                    # a grab never fires a select and a select never nudges the
                    # target by a pixel.
                    if math.hypot(
                        event.pos[0] - left_press_px[0],
                        event.pos[1] - left_press_px[1],
                    ) > DRAG_THRESHOLD_PX:
                        left_press_dragged = True
                        dragging_index = left_press_index
                        if left_press_index is None and grab_world_m is not None:
                            # The drag began on empty space: re-take the grab at
                            # the press point, so the pan starts from where the
                            # button went down rather than from the threshold.
                            camera.hold(grab_world_m, event.pos)
                if dragging_index is not None:
                    blocks[dragging_index] = block_dragged_to(
                        blocks[dragging_index], camera, event.pos
                    )
            elif event.type == pygame.MOUSEBUTTONUP:
                # ROUTED BY WHAT IS LATCHED. Every latched gesture ends here,
                # whichever button arrived.
                panning_with_mouse = False
                grab_world_m = None
                if dragging_index is not None:
                    dropped = blocks[dragging_index]
                    # Dropping counts as re-emplacement: C returns the target
                    # here, not to the Auto position. Clearing the field is a
                    # frequent action and re-aiming is not, so a clear that
                    # undid the placement every time would be the wrong default.
                    home_downrange[dragging_index] = downrange_m(dropped.pos)
                    status = (
                        f"{dropped.label} moved to "
                        f"{home_downrange[dragging_index]:.0f} m"
                    )
                    dragging_index = None
                elif left_press_px is not None and not left_press_dragged:
                    # A CLICK, not a drag: it selects.
                    #
                    # THE LANDING MARKER NEVER HIT-TESTS. It is not in the focus
                    # ring and has no grab rect, so a press on its pixels falls
                    # through to whatever body is beneath - which in the default
                    # scenario is the target, because the target is sighted onto
                    # the predicted impact point and the two genuinely share
                    # pixels. Report 1's trace confirmed that already happens.
                    picked = focus.pick(
                        focus_targets(), camera.to_screen, left_press_px
                    )
                    if picked is not None:
                        found = focus_on(picked)
                        status = f"selected the {found.label}" if found else status
                left_press_px = None
                left_press_dragged = False
                left_press_index = None
            elif event.type in (pygame.WINDOWFOCUSLOST, pygame.ACTIVEEVENT):
                # ANY DRAG ENDS ON FOCUS LOSS, UNCONDITIONALLY. The button-up
                # that would have ended it is delivered to whatever took focus,
                # so without this the pan flag stays set for the rest of the
                # session and the view follows the cursor with no button held.
                # That is the erratic drag Report 1 reproduced.
                panning_with_mouse = False
                grab_world_m = None
                dragging_index = None
                left_press_px = None
                left_press_dragged = False
                left_press_index = None
            elif event.type == pygame.KEYDOWN and settings_open:
                handle_settings_key(event)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.unicode.lower() == "s":
                    settings_open = True
                    settings_index = 0
                elif event.unicode.lower() == "m":
                    # AN OVERLAY TOGGLE, NOTHING MORE - batch 16 SS1. It used to
                    # swap between two cameras, snapshotting one and clamping
                    # the other's zoom into a different band. The camera is not
                    # touched here at all now, which is what makes gate 70's
                    # bit-identity claim trivially true rather than delicate.
                    overlay_conic = not overlay_conic
                    status = (
                        "conic overlay on" if overlay_conic
                        else "conic overlay off"
                    )
                elif event.unicode.lower() == "z":
                    # Z FITS THE FOCUSED BODY, batch 16 SS1. IMFD's auto-zoom.
                    target = focus.find(focus_targets(), focus_key)
                    if target is not None:
                        camera.set_anchor(
                            target.position_m, anchor_frac=target.anchor_frac
                        )
                        camera.fit(target.extent_m)
                        status = (
                            f"framed the {target.label} "
                            f"({camera.scale_px_per_m:.3g} px/m)"
                        )
                elif event.key == pygame.K_TAB:
                    target = focus_on(focus.cycle(
                        focus_targets(), focus_key,
                        -1 if fine else 1,      # SHIFT+TAB steps backwards
                    ))
                    status = f"focus: {target.label}" if target else "focus: none"
                elif event.key == pygame.K_HOME:
                    # HOME RETURNS FOCUS TO THE GUN and reframes - SS2. It used
                    # to recentre on whatever was already focused, which made it
                    # a second way to do what focusing now does by itself.
                    target = focus_on("gun")
                    status = (
                        f"view returned to the {target.label}" if target
                        else "focus: none"
                    )
                elif event.unicode.lower() == "r":
                    clear_round()
                elif event.unicode.lower() == "c":
                    clear_field()
                elif event.key == pygame.K_SPACE and not in_flight:
                    # SPACE FIRES. SS12 removed the explicit load: A, the
                    # `loaded` boolean and the HUD bore row are all gone. The
                    # load step was introduced in batch 9 SS7 to make reloading
                    # physical ahead of batch 15's recoil - "running the piece
                    # back up is the rate-of-fire constraint" - and with recoil
                    # deferred it was ceremony with nothing behind it.
                    clear_round()
                    state, bore = fire(
                        gun, shot, charge_kg, math.radians(elevation_deg)
                    )
                    trail.append(state.pos)
                    in_flight = True
                    status = "in flight"
                elif event.key == pygame.K_d and not in_flight:
                    drag_enabled = not drag_enabled
                elif event.key == pygame.K_UP and not in_flight:
                    elevation_deg = min(
                        elevation_deg + _elevation_step(settings, fine),
                        ELEVATION_LIMITS_DEG[1],
                    )
                elif event.key == pygame.K_DOWN and not in_flight:
                    elevation_deg = max(
                        elevation_deg - _elevation_step(settings, fine),
                        ELEVATION_LIMITS_DEG[0],
                    )
                elif event.key == pygame.K_RIGHT and not in_flight:
                    charge_kg = min(
                        charge_kg + (0.01 if fine else 0.1), CHARGE_LIMITS_KG[1]
                    )
                elif event.key == pygame.K_LEFT and not in_flight:
                    charge_kg = max(
                        charge_kg - (0.01 if fine else 0.1), CHARGE_LIMITS_KG[0]
                    )
                elif event.key in (pygame.K_F11, pygame.K_f):
                    toggle_fullscreen()
                # Matched on the typed character, not the keycode: pygame
                # keycodes follow the US layout, so K_LEFTBRACKET is not
                # where these characters live on a Latin American keyboard.
                elif event.unicode == "}":
                    time_scale_index = min(time_scale_index + 1, len(TIME_SCALES) - 1)
                elif event.unicode == "{":
                    time_scale_index = max(time_scale_index - 1, 0)
                elif event.unicode == "+":
                    zoom_by(1.25)
                elif event.unicode == "-":
                    zoom_by(1.0 / 1.25)

        # PREDICTED LANDING MARKER - SS11.
        #
        # ONE WRITER, KEYED ON THE LAYING AND THE FIELD. There were two: this
        # branch, and an in-flight branch that overwrote the same variable every
        # 100 ms from the live shot's state. After the round rebounded off a
        # block, that in-flight solve was a different point entirely - and
        # nothing restored the pre-shot value when the round stopped, because
        # the key did not include "a shot has been fired" and so could not
        # retrigger. Gate 115 measured 0.74 m of drift with elevation, charge
        # and the field all unchanged.
        #
        # There is no live in-flight readout. Nothing asked for one; if one is
        # wanted it gets its own name and its own variable, not this one.
        key = prediction_key(elevation_deg, charge_kg, drag_enabled, blocks)
        if key != last_prediction_key and not settings_open:
            predicted_landing = compute_prediction(
                gun, shot, charge_kg, elevation_deg, drag_enabled, blocks
            )
            last_prediction_key = key

        time_scale = TIME_SCALES[time_scale_index]

        # Settings is modal: nothing simulates behind it.
        if settings_open:
            pass
        elif in_flight:
            accumulator_s += min(frame_dt_s * time_scale, MAX_SIM_PER_FRAME_S)
            while accumulator_s >= FREE_FLIGHT_DT_S:
                # Keyword arguments deliberately. Both of these used to take
                # a trailing `ground_y_m=0.0` and were called positionally;
                # batch 8 replaced that parameter, so the 0.0 silently became
                # `ground_enabled=False` in one call (ground switched off, no
                # error) and `material=0.0` in the other (crash at impact).
                # Passing them by name is what stops a signature change
                # landing on the wrong parameter again.
                #
                # ALWAYS THROUGH THE SWEEP, EVEN WITH NO BLOCKS - batch 16
                # Part II SS13. The `if not blocks` shortcut used to call
                # `step_free_flight` directly, which knows only the old hard
                # stop; routing through `step_shot_and_blocks` unconditionally
                # (it already handles an empty tuple correctly) means there is
                # exactly one place the ground's material response lives.
                state, stepped, hit_ground, settled, elapsed_s, impacts = (
                    step_shot_and_blocks(
                        state,
                        blocks,
                        FREE_FLIGHT_DT_S,
                        drag_enabled=drag_enabled,
                    )
                )
                blocks = list(stepped)
                accumulator_s -= FREE_FLIGHT_DT_S
                flight_time_s += elapsed_s
                step_count += 1

                if impacts:
                    trail.append(state.pos)
                    # Name WHICH target was struck. The fastest-moving one is
                    # the one that just took the impulse.
                    struck = max(blocks, key=lambda b: b.vel.length())
                    status = (
                        f"struck the {struck.label} at {state.speed_ms:.0f} m/s, "
                        f"{struck.vel.length():.2f} m/s "
                        f"{struck.omega:+.2f} rad/s"
                    )
                if hit_ground:
                    trail.append(state.pos)
                    # TOUCHED IS NOT STOPPED - a shallow, fast graze skips
                    # onward and keeps flying; only a settled round leaves
                    # the dynamic set. See `step_shot_and_blocks`.
                    if not impacts and "struck" not in status:
                        verb = "impact" if settled else "skips"
                        status = (
                            f"{verb} at {state.pos.x:.0f} m, "
                            f"{state.speed_ms:.0f} m/s"
                        )
                if settled:
                    accumulator_s = 0.0
                    # The round has come to rest: it leaves the dynamic set
                    # and becomes scenery on the field.
                    retire_shot()
                    break
                if step_count % TRAIL_STRIDE == 0:
                    trail.append(state.pos)

        elif any(_block_is_moving(block) for block in blocks):
            # Let the targets finish settling after the shot has gone to rest.
            accumulator_s += min(frame_dt_s * time_scale, MAX_SIM_PER_FRAME_S)
            while accumulator_s >= FREE_FLIGHT_DT_S:
                blocks = [step_block(block, FREE_FLIGHT_DT_S) for block in blocks]
                accumulator_s -= FREE_FLIGHT_DT_S

        # THE INVARIANT, EVERY FRAME - batch 16 Part II SS14. Checked here,
        # after this frame's physics and before anything is drawn, so it
        # catches exactly the state the player is about to see.
        _assert_no_floating_static_bodies(spent_shots, blocks)

        # The zoom glide advances once a frame, re-holding its anchor on every
        # intermediate frame - not only at the end, which would let the view
        # drift during the glide.
        camera.tick_zoom(frame_dt_s)
        camera.update(state.pos if in_flight else None)

        focused = focus.find(focus_targets(), focus_key)
        moving = max(
            blocks, key=lambda b: b.vel.length(), default=None
        ) if any(_block_is_moving(block) for block in blocks) else None

        hud = Hud(
            elevation_deg=elevation_deg,
            charge_kg=charge_kg,
            muzzle_velocity_ms=bore.muzzle_speed_ms if bore else 0.0,
            speed_ms=state.speed_ms if state else 0.0,
            mach=state.mach if state else 0.0,
            # ALTITUDE ABOVE THE GROUND, |r| - R_EARTH, not world y - which
            # under the planet-centred frame is ~6.371e6 at the muzzle.
            # Clamped at zero because the bisected ground landing converges
            # from below and would otherwise read as "-0.0 m".
            altitude_m=max(
                0.0,
                altitude_m(state.pos) if state
                else altitude_m(gun.muzzle_m(math.radians(elevation_deg))),
            ),
            # Arc length along the surface, not Cartesian x.
            downrange_m=downrange_m(state.pos) if state else 0.0,
            flight_path_deg=(
                math.degrees(flight_path_angle_rad(state.pos, state.vel))
                if state else 0.0
            ),
            flight_time_s=flight_time_s,
            time_scale=time_scale,
            drag_enabled=drag_enabled,
            # THE MOVING TARGET, not "the block". With two of them the readout
            # has to say which one it is describing, and the one worth
            # describing is whichever is actually moving.
            block_speed_ms=moving.vel.length() if moving else 0.0,
            block_omega_rads=moving.omega if moving else 0.0,
            block_label=moving.label if moving else "",
            focus_label=focused.label if focused else "",
            rounds_on_field=len(spent_shots),
            status=status,
        )
        # ONE DRAW PATH. There were two - draw_scene and draw_map - selected by
        # the mode flag that no longer exists. draw_scene now covers the whole
        # band and decides what to draw from the SCALE, which is the same rule
        # SS8 generalises to per-object substitution.
        draw_scene(
            screen, camera, gun, math.radians(elevation_deg),
            state, trail, hud, fonts,
            blocks=blocks,
            spent_shots=spent_shots,
            predicted_landing=predicted_landing,
            focus_position_m=focused.position_m if focused else None,
            # The selection brackets stand off the body's DRAWN extent, so they
            # tighten as you zoom in and hold a minimum box when the body is
            # sub-pixel - which is the case the 4 px floor exists for.
            focus_extent_px=(
                2.0 * focused.extent_m * camera.scale_px_per_m
                if focused else 0.0
            ),
            overlay_conic=overlay_conic,
            mu_m3_s2=MU_EARTH_M3_S2,
        )
        if settings_open:
            # The apply-point sentence is now a permanent footer line inside
            # draw_settings rather than a note passed in from here. It said
            # "reset (R)" from stage 4.1 until batch 7, while the apply point
            # had moved to C in batch 5 when R became reload - and it stayed
            # wrong because it was only shown while changes were pending.
            draw_settings(screen, fonts, settings, settings_index)
        pygame.display.flip()

    pygame.quit()
    return 0


def main(argv=None):
    args = parse_args(argv)
    return report(args) if args.report else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
