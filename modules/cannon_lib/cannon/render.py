"""pygame view layer. Reads state, writes nothing.

Nothing in this module is imported by `physics`, `ballistics` or `gun`, and
it never will be. It converts metres to pixels at the last possible moment;
no pixel value is passed back into a physics function.

CAMERA BEHAVIOUR
----------------
ONE CAMERA, ONE CONTINUOUS BAND - batch 16 SS1. There were two camera objects
with two disjoint scale bands and M swapped between them; that split is deleted,
along with the clamp and the snapshot/restore that kept the copies agreeing.
See `Camera`.

The band runs from `zoom_floor` - the whole world in frame - to
ZOOM_CEILING_PX_PER_M, about 13 decades, with no boundary and nothing to clamp.
Zoom is under manual control and is the one thing the view will not decide for
you, except on Z, which frames the focused body. It is never animated during a
shot.

Scale is fixed WITHIN a flight, and the consequence is that no single zoom shows
both a 2.00 m barrel and a 2.7 km trajectory. What answers that is not zoom but
per-object marker substitution: below a few pixels of drawn extent an object is
replaced by a constant-size marker rather than drawn at true scale and made
invisible. That is SS8 and it is not built yet - the 4 px minimum-size floor and
the corner brackets are its ancestors.

The stage-3 centre-cross latch this note described until batch 8 is gone, and
the note said otherwise for four batches after the code stopped doing it.
"""

import math
from dataclasses import dataclass

import pygame

from . import hud
from .focus import GUN_EXTENT_M
from .physics import Vec2
from .planet import altitude_m, downrange_m, local_up, position_at
from .settings import FIELDS, ZOOM_DEFAULT_PX_PER_M, format_value

SKY_TOP = (86, 122, 164)
SKY_HORIZON = (168, 190, 206)
GROUND = (74, 64, 48)
GROUND_LINE = (120, 106, 82)
MARKER = (150, 138, 116)
MARKER_TEXT = (206, 198, 180)
ALTITUDE_LINE = (108, 138, 166)
BARREL = (58, 54, 50)
CARRIAGE = (96, 74, 46)
SHOT = (24, 22, 20)
SPENT_SHOT = (58, 54, 50)
BLOCK = (132, 140, 148)
BLOCK_EDGE = (196, 204, 212)

#: Same four signs as `impact._CORNER_SIGNS`. Not imported from there - that
#: name is private to impact.py, and duplicating four literal tuples is
#: cheaper than exporting it for one caller.
_BLOCK_CORNER_SIGNS = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
TRAIL = (226, 138, 74)
HUD_TEXT = (238, 236, 230)
HUD_DIM = (166, 176, 186)
HUD_PANEL = (18, 22, 30)
CROSSHAIR = (255, 255, 255)

#: Distant-object visibility. See the module note below.
#:
#: NOT SOURCED. These are the standard treatment in flight and space sims,
#: not figures traceable to a reference - a search turned up nothing
#: authoritative. They are starting points to be adjusted at the monitor.
#:
#: The underlying problem is that a 0.5 m block seen at low zoom is a
#: sub-pixel object: correctly placed, correctly drawn, and invisible. None
#: of these three mechanisms moves the camera; they change how a correctly
#: positioned object is drawn.
MIN_DRAWN_SIZE_PX = 4  # no world object renders smaller than this

#: PER-OBJECT MARKER SUBSTITUTION - batch 16 Part II SS8. Above this drawn
#: extent an object is geometry, as now; below it, a constant-size marker
#: from the registry. THIS SUPERSEDES `MIN_DRAWN_SIZE_PX` AS THE ANSWER TO
#: "correctly placed and invisible" - that floor kept a shrinking shape
#: nominally visible by inflating it past its true size, which is a lie
#: about geometry at any real zoom-out; a marker says "there is an object
#: here" instead of pretending to still show its shape. `MIN_DRAWN_SIZE_PX`
#: stays, because geometry drawn just above this threshold still wants a
#: floor so it does not round to a zero-width polygon.
#:
#: MEASURED, NOT TAKEN FROM THE BRIEF'S OWN "roughly 8 px": checked by eye
#: against a rendered frame (see the report) at a zoomed-out scale with a
#: block near the crossover. 8 px reads clearly as a shrinking square right
#: up to the point it becomes a marker, and the marker (a 4 px-radius ring,
#: 8 px across) does not visually jump in size at the swap - it lands within
#: 1 px of the geometry it replaces, which is what makes the transition
#: unnoticeable rather than a visible pop.
GEOMETRY_THRESHOLD_PX = 8.0

#: Below this on-screen span, a trajectory is a smear rather than a line and
#: is omitted rather than drawn - SS8's "a few pixels". Kept just under the
#: marker's own diameter (8 px, `2 * hud.MARKER_RADIUS_PX`) so a trail can
#: never read as a bigger object than the marker it would otherwise trail.
TRAIL_MIN_SPAN_PX = 6.0

#: The corner-bracket constants that lived here are gone with
#: `_draw_brackets`. The bracket is a registry glyph now and its sizes
#: are in `hud` - one definition, used by the one thing that draws it.
#: How far in from the frame edge the off-screen bearing is projected before
#: the registry's caret takes over. Only the MARGIN survives here; the arrow's
#: own size went with the arrow, and `hud.CARET_PX` is the one definition now.
OFFSCREEN_ARROW_MARGIN_PX = 34

#: Ground distance markers, m.
MARKER_SPACING_M = 100.0
MARKER_LABEL_SPACING_M = 500.0

#: Minimum clear space between adjacent ground labels, px.
MARKER_LABEL_GAP_PX = 12

#: Altitude reference lines, m.
ALTITUDE_SPACING_M = 100.0

#: Map-regime colours and sizes. The REGIME is gone - batch 16 SS1 united the
#: two cameras and SS8 makes each object choose its own representation from its
#: own drawn size - but these colours are still what the zoomed-out world is
#: drawn in, so they keep their names until SS5's registry replaces them.
#: Space is BLACK, not sky blue. The zoomed-out background used to be filled
#: with SKY_TOP, which is the atmospheric colour - and above the atmosphere
#: there is nothing to scatter light.
MAP_SPACE = (6, 8, 14)
MAP_PLANET = (58, 74, 92)
MAP_PLANET_EDGE = (122, 150, 176)
MAP_CONIC = (238, 206, 120)
MAP_CONIC_FLOWN = (226, 138, 74)
MAP_ICON_PX = 5
MAP_APSIS_PX = 4
MAP_CIRCLE_SAMPLES = 192

#: The focus ring is DELETED - batch 16 SS108. The corner bracket is the one
#: selection glyph and it lives in `hud`, not here.


@dataclass(frozen=True, slots=True)
class Hud:
    """Values the HUD displays. Built by the loop, read-only here."""

    #: LAUNCH elevation, relative to the local vertical AT THE GUN. With
    #: radial gravity the local vertical rotates as the shot travels, so
    #: elevation is a property of the launch, not of the flight. The angle
    #: between velocity and the local horizontal at the shot's CURRENT
    #: position is `flight_path_deg`, which is a different quantity.
    elevation_deg: float
    charge_kg: float
    muzzle_velocity_ms: float
    speed_ms: float
    mach: float
    altitude_m: float
    downrange_m: float
    flight_path_deg: float
    flight_time_s: float
    time_scale: float
    drag_enabled: bool
    block_speed_ms: float
    block_omega_rads: float
    rounds_on_field: int
    status: str
    #: Appended with defaults so the existing positional constructions in the
    #: gate suites keep working unchanged. Which target the speed readout is
    #: describing, whether the piece has a round in it, and what the camera is
    #: focused on.
    block_label: str = ""
    focus_label: str = ""


#: Zoom ceiling, px/m. At 1000 px/m one pixel is a millimetre.
ZOOM_CEILING_PX_PER_M = 1000.0

#: Half-extent of the world the camera must be able to frame, m.
#:
#: Neptune's aphelion, near enough. Batch 16 SS1 replaces the two disjoint zoom
#: bands with their union, extended down far enough to hold the solar system
#: SS9 adds - so the floor is derived from this rather than chosen. It gives a
#: band of about 13 decades against the previous surface band's 4.
#:
#: SS9's element tables are the authority on planetary distances once they land;
#: this is the view extent, and it only has to be big enough.
WORLD_EXTENT_M = 4.55e12

#: Margin left around a body when fitting it to the viewport - SS1's 10%.
FIT_MARGIN = 0.10

#: Time constant of the zoom glide, s. Zoom eases; pan never does.
ZOOM_TIME_CONSTANT_S = 0.10


def fit_scale(viewport_px, extent_m):
    """Scale at which a body of half-extent `extent_m` fits, px/m.

    ONE DEFINITION FOR EVERY FIT. Z uses it on the focused body, the zoom floor
    uses it on the whole world, and the old `fit_zoom` used it on the planet.
    Hard-coding any of them put the subject off-frame on a narrower viewport,
    which is the fit_zoom lesson from Progress Report 3.
    """
    if extent_m <= 0.0:
        return ZOOM_CEILING_PX_PER_M
    return (1.0 - FIT_MARGIN) * min(viewport_px) / (2.0 * extent_m)


def fit_zoom(viewport_px):
    """Scale at which the whole planet fits. Kept as the named planet case."""
    from .planet import R_EARTH_M

    return fit_scale(viewport_px, R_EARTH_M)


def zoom_floor(viewport_px):
    """The one zoom floor: whole world in frame."""
    return fit_scale(viewport_px, WORLD_EXTENT_M)


class Camera:
    """The camera. There is exactly one, and it has one scale.

    ONE CAMERA, ONE BAND - batch 16 SS1. Until this batch there were two camera
    objects with two scales and two disjoint bands, and M swapped which one was
    looking. That is the duplication this batch exists to remove: the same
    concept implemented twice, with a clamp and a snapshot/restore dance holding
    the copies together. Gate 30 asserted that neither could corrupt the other,
    which is a question that stops existing once there is one.

    The band now runs from `zoom_floor` - the whole world in frame - up to
    ZOOM_CEILING_PX_PER_M, continuously, with no boundary to cross and nothing
    to clamp. What used to be "map mode" is just being zoomed out; what used to
    be a representation switch is each object deciding for itself how to draw
    (SS8), and M survives only as an overlay toggle.

    ALIGNED TO THE LOCAL HORIZON, kept from Progress Report 3. Screen-up is the
    local r-hat at the view centre, so the horizon is level wherever you stand
    rather than tilting by your angle from downrange zero. At downrange zero the
    local up IS (0, 1), which is why this reduced exactly to the old world-axis
    framing at the gun. Zoomed out and centred on the planet it degrades
    gracefully - screen-up is up at the centre of view, which is what a
    planet-centred view wants anyway.

    Either follows the shot always, or never moves. THERE IS NO TRANSITION
    BETWEEN THE TWO AND NO SMOOTHING STATE: the stage-3 centre-cross latch was
    deleted outright rather than left dormant behind a flag, because a dormant
    transition is exactly the thing that comes back as a bug.

    THE CAMERA DOES NOT MOVE OF ITS OWN ACCORD except when focus changes or Z
    is pressed, both of which are direct commands. A cinematic pan-in was
    proposed and DROPPED by the PM - not parked, dropped.
    """

    #: Attributes that fully determine what this camera shows. Declared rather
    #: than discovered, so `snapshot` cannot quietly stop covering a field
    #: someone adds later - it would have to be added here too, and gate 70
    #: compares snapshots field by field.
    _STATE_FIELDS = (
        "scale_px_per_m", "pan_offset_m", "center_m", "follow",
        "anchor_m", "anchor_frac", "basis_m",
    )

    def __init__(
        self,
        viewport_px,
        anchor_m,
        *,
        scale_px_per_m=ZOOM_DEFAULT_PX_PER_M,
        anchor_frac=(0.18, 0.82),
        follow=True,
    ):
        self.viewport_px = viewport_px
        self.scale_px_per_m = scale_px_per_m
        self.pan_offset_m = Vec2(0.0, 0.0)
        self.anchor_m = anchor_m
        self.anchor_frac = anchor_frac
        self.follow = follow
        #: The point the basis is evaluated at. The anchor normally, the shot
        #: while following one. Never the centre - see `_basis`.
        self.basis_m = anchor_m
        #: Where the zoom glide is heading, and the screen point it is anchored
        #: on. None when no zoom is in flight.
        self.zoom_target_px_per_m = None
        self.zoom_anchor_px = None
        self.center_m = self._anchored_center()

    # -- band ------------------------------------------------------------
    def zoom_limits(self):
        """(floor, ceiling) px/m. One continuous band, derived not chosen."""
        return zoom_floor(self.viewport_px), ZOOM_CEILING_PX_PER_M

    def clamp_scale(self, scale_px_per_m):
        low, high = self.zoom_limits()
        return min(max(scale_px_per_m, low), high)

    def fit(self, extent_m):
        """Z. Frame a body of half-extent `extent_m` with FIT_MARGIN to spare."""
        self.set_scale(self.clamp_scale(fit_scale(self.viewport_px, extent_m)))

    # -- state -----------------------------------------------------------
    def snapshot(self):
        """Everything needed to restore this camera bit-identically.

        A dict rather than a tuple so a mismatch names the field that moved.
        """
        return {name: getattr(self, name) for name in self._STATE_FIELDS}

    def restore(self, snapshot):
        """Put the camera back exactly as `snapshot` found it.

        NO reframe() AFTERWARDS, and the reason is narrower than it looks. The
        pan offset is in the snapshot, so re-deriving the centre from the anchor
        would usually land on the same value - which is exactly why this is easy
        to get wrong and hard to notice.

        It breaks WHILE FOLLOWING A SHOT. A following camera's centre is the
        shot's position, which is not a function of the anchor and the pan at
        all; reframe() would throw it back to the gun. Gate 70 covers that case
        specifically, and caught itself failing to before it did.
        """
        for name, value in snapshot.items():
            setattr(self, name, value)

    def set_viewport(self, viewport_px):
        self.viewport_px = viewport_px
        self.reframe()

    def set_scale(self, scale_px_per_m):
        self.scale_px_per_m = scale_px_per_m
        self.reframe()

    def reset_pan(self):
        self.pan_offset_m = Vec2(0.0, 0.0)
        self.reframe()

    def move_camera_by_pixels(self, delta_x_px, delta_y_px):
        """Move the CAMERA by a screen-space amount, y measured downward.

        THE SINGLE SIGNED OFFSET. Every pan input routes through this one
        method, so the keyboard and the mouse cannot drift apart in sign -
        which is exactly how J and L ended up inverted, from two independent
        handlers with independently chosen signs.

        KEYBOARD ONLY NOW. The mouse used to come through here too, with the
        caller negating the delta to express "the drag moves the content" - two
        mental models meeting at one method, and a sign convention that gate 80
        existed to police. The tranche 2 addendum replaced the mouse path with
        `hold`, where the sign falls out of the transform and cannot be
        inverted the way batch 13 SS0 inverted it.

        What remains is the nudge: IJKL move the CAMERA, so content slides
        opposite. J and a rightward drag still agree on screen, and gate 80 now
        proves that agreement rather than asserting it as a convention.

        INERT IN FOLLOW MODE, deliberately. A camera locked to the shot that
        could also be nudged would be a third behaviour between the two the
        PM chose, and the standing rule is that there is no transition
        between follow and fixed.
        """
        if self.follow:
            return
        self.pan_offset_m = self.pan_offset_m + self._pan_delta(
            delta_x_px, delta_y_px
        )
        self.reframe()

    def visible_world_rect(self):
        """(left_m, right_m, bottom_m, top_m) currently in frame."""
        width_px, height_px = self.viewport_px
        half_w_m = 0.5 * width_px / self.scale_px_per_m
        half_h_m = 0.5 * height_px / self.scale_px_per_m
        return (
            self.center_m.x - half_w_m,
            self.center_m.x + half_w_m,
            self.center_m.y - half_h_m,
            self.center_m.y + half_h_m,
        )


    def _basis(self):
        """(screen-right, screen-up) as world unit vectors.

        TAKEN AT `basis_m`, NOT AT THE CENTRE, and the distinction is not
        cosmetic. `_anchored_center` places the centre by stepping away from the
        anchor along this basis; if `to_screen` then evaluated the basis at the
        resulting centre, the two would disagree by the angle between anchor and
        centre. That angle is ~1e-7 rad at 100 px/m, which is why it went
        unnoticed for eight batches - and 0.65 rad at 1e-4 px/m, where the offset
        is 4.1e6 m and comparable to R_EARTH. Gate 97 measured the anchor landing
        205 px from its screen fraction before this was one point.

        The basis point is what the view is ABOUT: the anchored body, or the shot
        while following it. Screen-up is the local vertical there.
        """
        up = local_up(self.basis_m)
        return up.perp() * -1.0, up

    def _pan_delta(self, delta_x_px, delta_y_px):
        east, up = self._basis()
        return (
            east * (delta_x_px / self.scale_px_per_m)
            + up * (-delta_y_px / self.scale_px_per_m)
        )

    def _anchored_center(self):
        """World point at screen centre that pins the anchor at its fraction."""
        width_px, height_px = self.viewport_px
        frac_x, frac_y = self.anchor_frac
        offset_x_px = (0.5 - frac_x) * width_px
        offset_y_px = (0.5 - frac_y) * height_px
        up = local_up(self.basis_m)
        east = up.perp() * -1.0
        return (
            self.anchor_m
            + east * (offset_x_px / self.scale_px_per_m)
            + up * (-offset_y_px / self.scale_px_per_m)
            + self.pan_offset_m
        )

    def reframe(self):
        self.basis_m = self.anchor_m
        self.center_m = self._anchored_center()

    def set_anchor(self, anchor_m, *, anchor_frac=None):
        """Point the camera at a different body.

        THE PAN OFFSET IS CLEARED. Re-anchoring with the old offset still
        applied would place the new body wherever the previous pan happened to
        leave it, so HOME would not actually re-centre anything - which is the
        one thing gate 97 asserts about it.
        """
        self.anchor_m = anchor_m
        if anchor_frac is not None:
            self.anchor_frac = anchor_frac
        self.pan_offset_m = Vec2(0.0, 0.0)
        self.reframe()

    def set_follow(self, follow):
        self.follow = follow
        self.reframe()

    def update(self, shot_pos_m):
        """Track the shot. Pass None when no shot is in flight."""
        if self.follow and shot_pos_m is not None:
            # Following: the shot is both the centre and what the view is about,
            # so it is the basis point too.
            self.basis_m = shot_pos_m
            self.center_m = shot_pos_m
        else:
            self.reframe()

    # -- direct manipulation ---------------------------------------------
    def hold(self, world_m, point_px):
        """Move the camera so `world_m` projects EXACTLY to `point_px`.

        THE ONE INVARIANT BOTH GESTURES REDUCE TO: the point under the cursor
        stays under the cursor. Grab-pan holds the point grabbed on button-down;
        cursor-anchored zoom holds the point under the wheel.

        SOLVED FROM to_screen, NOT BY A HAND-DERIVED PIXEL OFFSET. Inverting
        the projection here would be a second copy of the transform, and this
        codebase has been bitten three times by two copies of one concept
        drifting apart - the pan signs, the positional bind, the camera basis.
        The algebra below is `to_screen` rearranged for the centre and nothing
        else, so it cannot disagree with it.

            sx = W/2 + (p - c).east * scale      ->  (p - c).east = ox / scale
            sy = H/2 - (p - c).up   * scale      ->  (p - c).up   = -oy / scale

        so  c = p - east * (ox / scale) + up * (oy / scale).
        """
        east, up = self._basis()
        width_px, height_px = self.viewport_px
        offset_x = point_px[0] - 0.5 * width_px
        offset_y = point_px[1] - 0.5 * height_px
        wanted_center = (
            world_m
            - east * (offset_x / self.scale_px_per_m)
            + up * (offset_y / self.scale_px_per_m)
        )
        # The centre is derived from anchor + pan, so the pan is what moves.
        self.pan_offset_m = self.pan_offset_m + (wanted_center - self.center_m)
        self.reframe()

    def zoom_to(self, scale_px_per_m, anchor_px):
        """Set the scale, keeping the world point under `anchor_px` fixed.

        REPLACES CENTRE-ANCHORED ZOOM. Scaling about the screen centre slides
        whatever you were looking at away as you zoom, so you chase it with the
        pan - which is the "unwieldy" the PM reported, and it is worse the
        further out you go. Across 13.1 decades it is unusable.

        Order matters: read the world point BEFORE the scale changes, then
        re-hold it after, because `_anchored_center` is itself scale-dependent.
        """
        held = self.to_world(anchor_px)
        self.scale_px_per_m = self.clamp_scale(scale_px_per_m)
        self.reframe()
        self.hold(held, anchor_px)

    def drop_follow_in_place(self):
        """Stop following without moving the view.

        DRAGGING WHILE FOLLOWING DROPS FOLLOW - tranche 2 addendum SS6, and it
        reverses gate 80's inert-while-following case. Under direct
        manipulation, grabbing the world while the camera tracks a shot must do
        something rather than nothing: the gesture wins.

        The view must not jump when it happens, so the anchor is re-pinned to
        wherever the camera already is, centred. Follow does not silently
        resume.
        """
        if not self.follow:
            return
        here = self.center_m
        self.follow = False
        self.anchor_m = here
        self.anchor_frac = (0.5, 0.5)
        self.basis_m = here
        self.pan_offset_m = Vec2(0.0, 0.0)
        self.reframe()

    def request_zoom(self, factor, anchor_px):
        """Ask for a scale change, to be eased in over the next few frames.

        SMOOTHING IS ON ZOOM ONLY. Pan must be instantaneous or direct
        manipulation feels like ice; zoom benefits from a glide. Getting that
        asymmetry backwards is the usual way this feels wrong.
        """
        self.zoom_target_px_per_m = self.clamp_scale(
            (self.zoom_target_px_per_m or self.scale_px_per_m) * factor
        )
        self.zoom_anchor_px = anchor_px

    def tick_zoom(self, dt_s):
        """Ease the displayed scale towards the target. Called once a frame.

        THE ANCHOR IS RE-HELD ON EVERY INTERMEDIATE FRAME, not only at the end.
        Holding it only at rest lets the view drift during the glide, which is
        the characteristic ugly failure of this feature.
        """
        if self.zoom_target_px_per_m is None:
            return
        target = self.zoom_target_px_per_m
        if abs(target - self.scale_px_per_m) <= target * 1e-6:
            self.zoom_to(target, self.zoom_anchor_px)
            self.zoom_target_px_per_m = None
            return
        # Exponential approach, frame-rate independent.
        blend = 1.0 - math.exp(-dt_s / ZOOM_TIME_CONSTANT_S)
        stepped = self.scale_px_per_m + (target - self.scale_px_per_m) * blend
        self.zoom_to(stepped, self.zoom_anchor_px)

    def to_screen(self, pos_m):
        east, up = self._basis()
        offset = pos_m - self.center_m
        width_px, height_px = self.viewport_px
        return (
            0.5 * width_px + offset.dot(east) * self.scale_px_per_m,
            0.5 * height_px - offset.dot(up) * self.scale_px_per_m,
        )

    def to_world(self, point_px):
        """Inverse of to_screen. The only correct way back from a cursor."""
        east, up = self._basis()
        width_px, height_px = self.viewport_px
        return (
            self.center_m
            + east * ((point_px[0] - 0.5 * width_px) / self.scale_px_per_m)
            + up * (-(point_px[1] - 0.5 * height_px) / self.scale_px_per_m)
        )


def surface_background_scale(viewport_px):
    """Scale above which the ground arc fills the view, px/m.

    DERIVED, NOT CHOSEN. Above this the planet's drawn radius exceeds the
    viewport diagonal, so the ground reads as a horizon and the sky is what is
    behind everything; below it the planet is a disc against space. There is no
    mode flag and no user-visible boundary - it falls out of the geometry, and
    it is the same shape of rule SS8 generalises to every object individually.

    At 1280x720 this is 2.3e-4 px/m.
    """
    from .planet import R_EARTH_M

    return math.hypot(*viewport_px) / R_EARTH_M


def _apparent_extent_px(camera, half_extent_m):
    """A body's own drawn size, in screen pixels, at its TRUE scale.

    UNFLOORED, deliberately - this is the number the geometry/marker decision
    is made from, and applying `MIN_DRAWN_SIZE_PX` here would make every body
    read as at-least-4px forever and the marker branch would never fire.
    """
    return 2.0 * half_extent_m * camera.scale_px_per_m


def _draw_marker(instrument, camera, position_m, label, colour):
    """An object substituted for a registry marker - SS8.

    Still just a point in world space, so it is exactly as selectable,
    clickable and Z-framable as the geometry it replaces: `focus.pick` picks
    by proximity to `position_m` regardless of how the body is drawn, and Z
    frames its true `extent_m`, not its drawn size. Substitution is a render-
    layer decision only.
    """
    point_px = camera.to_screen(position_m)
    if not _on_screen(instrument.surface, point_px, margin_px=hud.MARKER_RADIUS_PX):
        return
    rect = instrument.marker(point_px, colour)
    if label:
        instrument.submit_label(label, point_px, colour, priority=10, glyph_rect=rect)


def _trail_span_px(camera, trail):
    """The trail's own on-screen extent - its bounding-box diagonal, px."""
    if len(trail) < 2:
        return 0.0
    points = [camera.to_screen(p) for p in trail]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def draw_scene(surface, camera, gun, elevation_rad, state, trail, hud_values,
               fonts, *, blocks=(), spent_shots=(), predicted_landing=None,
               focus_position_m=None, focus_extent_px=0.0, overlay_conic=False,
               mu_m3_s2=None):
    """Draw one frame. `state` may be None when no shot is in flight.

    Tail parameters are KEYWORD-ONLY. They were positional-with-defaults, and
    the loop suite's spy mirrored that signature exactly, so adding the
    predicted-landing marker broke the spy and turned every loop gate red -
    a failure that reads as a broken application rather than a broken
    harness.

    The first fix was to make the spy swallow trailing arguments with
    *extra/**keyword_extra. That stopped the false red but was the wrong
    trade: a swallowing wrapper also cannot detect a GENUINE positional
    misbind, which converts a false red into a possible false green. That is
    the more expensive failure and it is precisely what batch 8's bug was -
    `0.0` binding silently to `ground_enabled` and flying the shot through
    the Earth.

    With the bare `*` here, neither the loop nor the spy can misbind at all,
    and the swallowing wrapper stops being load-bearing.
    """
    # Background from the scale, not from a mode. Zoomed in, the ground is a
    # horizon under a sky; zoomed out, the planet is a disc against space.
    on_surface = camera.scale_px_per_m >= surface_background_scale(
        camera.viewport_px
    )
    if on_surface:
        _draw_sky(surface)
        _draw_altitude_lines(surface, camera, fonts)
        _draw_ground(surface, camera, fonts)
    else:
        surface.fill(MAP_SPACE)
        _draw_planet_disc(surface, camera)

    if overlay_conic and state is not None and mu_m3_s2 is not None:
        _draw_conic(surface, camera, state, mu_m3_s2, fonts)

    # PER-OBJECT MARKER SUBSTITUTION - SS8. Each body below decides for
    # itself, from its OWN drawn size, whether it is geometry or a marker -
    # not one global crossover, which would put the gun and a distant spent
    # round on the same side of the line for no reason connected to either.
    instrument = hud.Instrument(surface, fonts["small"])

    if _apparent_extent_px(camera, GUN_EXTENT_M) >= GEOMETRY_THRESHOLD_PX:
        _draw_gun(surface, camera, gun, elevation_rad)
    else:
        _draw_marker(instrument, camera, gun.trunnion_m, "gun", hud.ACTUAL)

    for index, spent in enumerate(spent_shots):
        if _apparent_extent_px(camera, spent.radius) >= GEOMETRY_THRESHOLD_PX:
            _draw_shot(surface, camera, spent, colour=SPENT_SHOT)
        else:
            _draw_marker(
                instrument, camera, spent.pos, f"spent {index + 1}", hud.ACTUAL
            )

    for block in blocks:
        half_extent_m = max(block.half_width, block.half_height)
        if _apparent_extent_px(camera, half_extent_m) >= GEOMETRY_THRESHOLD_PX:
            _draw_block(surface, camera, block)
        else:
            _draw_marker(instrument, camera, block.pos, block.label, hud.ACTUAL)

    if predicted_landing is not None:
        _draw_landing_marker(instrument, camera, predicted_landing)

    # Sub-pixel trajectories are omitted, not smeared - a few pixels of paint
    # standing in for 2.7 km of flight is worse than nothing.
    if _trail_span_px(camera, trail) >= TRAIL_MIN_SPAN_PX:
        _draw_trail(surface, camera, trail)

    if state is not None:
        if _apparent_extent_px(camera, state.radius) >= GEOMETRY_THRESHOLD_PX:
            _draw_shot(surface, camera, state)
        else:
            _draw_marker(instrument, camera, state.pos, None, hud.ACTUAL)
    if camera.follow and state is not None:
        _draw_crosshair(surface)
    if focus_position_m is not None:
        _draw_selection(
            instrument, camera, focus_position_m, focus_extent_px
        )

    # Off-screen indicators. Every target gets one; the active shot gets one
    # only when the camera is not following it, because a following camera
    # cannot lose it.
    for block in blocks:
        _draw_offscreen_indicator(
            instrument, camera, block.pos, block.label, hud.SELECTED
        )
    if state is not None and not camera.follow:
        _draw_offscreen_indicator(
            instrument, camera, state.pos, "shot", hud.ACTUAL
        )

    instrument.flush()
    _draw_hud(surface, hud_values, fonts)


def _draw_selection(instrument, camera, position_m, extent_px):
    """The one selection glyph: corner brackets, from the registry.

    THE FOCUS RING IS DELETED. It was a closed circle drawn over the focused
    body - occluding it, and a second selection indicator alongside the batch 5
    corner brackets. Two glyphs for one concept is the duplication this batch
    exists to remove, and batch 16 SS108 chose the bracket.

    Folded into this tranche rather than SS2 because SS5 lands first: a registry
    drawing brackets while the ring was still live would have put TWO selection
    glyphs on screen at once, which is worse than either alone.
    """
    point_px = camera.to_screen(position_m)
    if not _on_screen(instrument.surface, point_px):
        return
    instrument.brackets(point_px, extent_px, hud.SELECTED)


def draw_thick_line(surface, colour, start_px, end_px, width_px,
                    *, round_caps=True):
    """A stroke of `width_px`, extended PERPENDICULAR TO ITS OWN AXIS.

    THE ONE THICK-LINE PRIMITIVE. `pygame.draw.line` with a width does not do
    this: it widens along whichever screen axis is dominant, so the end caps
    stay axis-aligned and the perpendicular thickness collapses by cos or sin
    of the angle. Measured in batch 15: a 24 px stroke rasterises 17.3 px thick
    at 45 degrees, and the gun's 11 px barrel came out 8.1 px - which is what
    the PM saw as rectangles shearing into parallelograms as elevation changed.
    It is worst at exactly 45 degrees, the factory elevation.

    Built as a four-corner polygon from the axis and its perpendicular, so the
    thickness is the thickness at every angle and the caps are square to the
    stroke. Round caps are added by default because a butt cap on a barrel
    joining a trunnion circle reads as severed; they are cosmetic and invisible
    to physics.

    Sub-pixel and degenerate cases fall back rather than dividing by zero: a
    zero-length stroke is a dot, and a width at or below 1 px is a hairline,
    where a polygon would rasterise thinner than the line primitive does.
    """
    dx = end_px[0] - start_px[0]
    dy = end_px[1] - start_px[1]
    length_px = math.hypot(dx, dy)

    if length_px <= 0.0:
        radius = max(1, int(round(0.5 * width_px)))
        pygame.draw.circle(surface, colour, start_px, radius)
        return
    if width_px <= 1.0:
        pygame.draw.line(surface, colour, start_px, end_px, 1)
        return

    half = 0.5 * width_px
    # Perpendicular to the stroke, which is the whole point.
    nx = -dy / length_px * half
    ny = dx / length_px * half
    pygame.draw.polygon(
        surface, colour,
        (
            (start_px[0] + nx, start_px[1] + ny),
            (end_px[0] + nx, end_px[1] + ny),
            (end_px[0] - nx, end_px[1] - ny),
            (start_px[0] - nx, start_px[1] - ny),
        ),
    )
    if round_caps and width_px >= 3.0:
        radius = int(round(half))
        pygame.draw.circle(surface, colour, start_px, radius)
        pygame.draw.circle(surface, colour, end_px, radius)


def _on_screen(surface, point_px, margin_px=0):
    width_px, height_px = surface.get_size()
    return (
        -margin_px <= point_px[0] <= width_px + margin_px
        and -margin_px <= point_px[1] <= height_px + margin_px
    )



def _draw_offscreen_indicator(instrument, camera, target_m, label, colour):
    """A caret at the frame edge on the bearing to an off-screen object.

    A REGISTRY GLYPH, not a filled arrow of its own. It was the fourth thing in
    the program drawing its own pointer in its own colour with a bare label
    floating beside it - the same species as the landing chevron, which is what
    batch 16 SS5 exists to end.

    `edge_caret` does the clamping and the pointing; this only has to work out
    where the bearing meets the frame.
    """
    surface = instrument.surface
    point_px = camera.to_screen(target_m)
    if _on_screen(surface, point_px):
        return

    width_px, height_px = surface.get_size()
    centre_x, centre_y = 0.5 * width_px, 0.5 * height_px
    delta_x = point_px[0] - centre_x
    delta_y = point_px[1] - centre_y
    if delta_x == 0.0 and delta_y == 0.0:
        return

    # Out along the bearing to whichever frame edge it meets first. The caret
    # itself is then clamped inside the viewport by the registry, so this only
    # has to be roughly right - and it cannot produce a caret off-screen even
    # if it is not.
    limit_x = centre_x - OFFSCREEN_ARROW_MARGIN_PX
    limit_y = centre_y - OFFSCREEN_ARROW_MARGIN_PX
    reach = min(
        limit_x / abs(delta_x) if delta_x else math.inf,
        limit_y / abs(delta_y) if delta_y else math.inf,
    )
    edge = (centre_x + delta_x * reach, centre_y + delta_y * reach)
    rect = instrument.edge_caret(edge, colour)
    instrument.submit_label(
        f"{label} {downrange_m(target_m):.0f} m", edge, colour,
        priority=20, glyph_rect=rect,
    )

def _draw_sky(surface):
    width_px, height_px = surface.get_size()
    bands = 48
    band_h = math.ceil(height_px / bands)
    for i in range(bands):
        t = i / (bands - 1)
        colour = (
            round(SKY_TOP[0] + t * (SKY_HORIZON[0] - SKY_TOP[0])),
            round(SKY_TOP[1] + t * (SKY_HORIZON[1] - SKY_TOP[1])),
            round(SKY_TOP[2] + t * (SKY_HORIZON[2] - SKY_TOP[2])),
        )
        surface.fill(colour, (0, i * band_h, width_px, band_h))


def _visible_span(camera):
    """(downrange centre, half-width, altitude centre, half-height) in metres."""
    width_px, height_px = camera.viewport_px
    return (
        downrange_m(camera.center_m),
        0.5 * width_px / camera.scale_px_per_m,
        altitude_m(camera.center_m),
        0.5 * height_px / camera.scale_px_per_m,
    )


#: Points used to draw the ground arc across the viewport.
GROUND_ARC_SAMPLES = 64


def _draw_ground(surface, camera, fonts):
    """The ground is an ARC, drawn across the viewport only.

    Never a 6371 km circle - just the piece of it that is on screen. At
    100 px/m the arc sags 0.062 m over 890 m and is sub-pixel, and it is
    still drawn as a curve rather than special-cased to a line, because map
    mode is next and will need the real thing.
    """
    width_px, height_px = surface.get_size()
    centre_m, half_width_m, _, _ = _visible_span(camera)

    arc = [
        camera.to_screen(
            position_at(
                centre_m - half_width_m
                + 2.0 * half_width_m * i / (GROUND_ARC_SAMPLES - 1),
                0.0,
            )
        )
        for i in range(GROUND_ARC_SAMPLES)
    ]

    # Fill below the arc by closing it against the bottom of the screen.
    below = arc + [(arc[-1][0], height_px + 2), (arc[0][0], height_px + 2)]
    if max(point[1] for point in arc) > 0:
        pygame.draw.polygon(surface, GROUND, below)
        pygame.draw.lines(surface, GROUND_LINE, False, arc, 2)

    first = math.floor((centre_m - half_width_m) / MARKER_SPACING_M) * MARKER_SPACING_M
    marker_m = first
    # Decluttering: only enough to stop labels overlapping. The decade
    # ladder, unit switching and the rest of the scale work belong to map
    # mode, where the span runs from metres to hundreds of kilometres - by
    # PM decision, the surface view gets the landing marker and nothing else.
    # DENSITY GUARD, the same one the altitude lines have had all along. The
    # marker loop steps by MARKER_SPACING_M across the visible span, and once the
    # band was united that span reached 4.3e6 m on the surface path - about 40,000
    # ticks and labels, measured at 220 ms a frame by gate 32. Dropping them
    # entirely is right rather than thinning them: at that scale a downrange
    # ruler is unreadable anyway, and SS6's one scale ladder is what replaces it.
    if (2.0 * half_width_m) / MARKER_SPACING_M > 200.0:
        return

    last_label_right_px = -math.inf
    while marker_m <= centre_m + half_width_m:
        base = position_at(marker_m, 0.0)
        tip_px = camera.to_screen(base)
        labelled = abs(marker_m) % MARKER_LABEL_SPACING_M < 0.5 * MARKER_SPACING_M
        tick_px = 14 if labelled else 7
        # Ticks run INTO the ground along the local down, not along -y.
        down = local_up(base) * -1.0
        end_px = camera.to_screen(base + down * (tick_px / camera.scale_px_per_m))
        pygame.draw.line(surface, MARKER, tip_px, end_px, 1)
        if labelled and marker_m >= 0.0:
            label = fonts["small"].render(
                f"{marker_m / 1000.0:.1f} km" if marker_m >= 1000.0
                else f"{marker_m:.0f} m",
                True, MARKER_TEXT,
            )
            left_px = end_px[0] + 4
            if left_px > last_label_right_px + MARKER_LABEL_GAP_PX:
                surface.blit(label, (left_px, end_px[1] + 2))
                last_label_right_px = left_px + label.get_width()
        marker_m += MARKER_SPACING_M


def _draw_altitude_lines(surface, camera, fonts):
    """Reference arcs labelled by ALTITUDE ABOVE THE GROUND.

    Labelled with |r| - R_EARTH, not world y. Reading the raw y here is what
    made the gun appear to be 6,371,000 m above sea level.
    """
    centre_m, half_width_m, altitude_centre_m, half_height_m = _visible_span(camera)
    bottom_m = altitude_centre_m - half_height_m
    top_m = altitude_centre_m + half_height_m
    if (top_m - bottom_m) / ALTITUDE_SPACING_M > 40.0:
        return  # too dense to read; drop them rather than draw mush

    altitude = max(
        ALTITUDE_SPACING_M,
        math.floor(bottom_m / ALTITUDE_SPACING_M) * ALTITUDE_SPACING_M,
    )
    while altitude <= top_m:
        arc = [
            camera.to_screen(
                position_at(
                    centre_m - half_width_m
                    + 2.0 * half_width_m * i / (GROUND_ARC_SAMPLES - 1),
                    altitude,
                )
            )
            for i in range(GROUND_ARC_SAMPLES)
        ]
        pygame.draw.lines(surface, ALTITUDE_LINE, False, arc, 1)
        label = fonts["small"].render(f"{altitude:.0f} m", True, ALTITUDE_LINE)
        surface.blit(label, (6, arc[0][1] - label.get_height() - 2))
        altitude += ALTITUDE_SPACING_M


def _draw_gun(surface, camera, gun, elevation_rad):
    """The piece pivots about its trunnions, which is where the origin is."""
    scale = camera.scale_px_per_m
    trunnion_px = camera.to_screen(gun.trunnion_m)
    muzzle_px = camera.to_screen(gun.muzzle_m(elevation_rad))
    breech_px = camera.to_screen(gun.breech_m(elevation_rad))

    # Carriage first, so the tube sits over it: a wheel below the trunnions
    # and a trail running back to the ground.
    wheel_radius_m = 0.75
    wheel_px = camera.to_screen(position_at(0.0, wheel_radius_m))
    wheel_radius_px = max(2, round(wheel_radius_m * scale))
    # draw_thick_line, not pygame.draw.line: the trail runs at a shallow angle
    # and the barrel sweeps through 45 degrees, where the built-in primitive
    # loses 26% of the requested thickness and squares the caps off to the
    # screen axes instead of to the stroke.
    draw_thick_line(
        surface, CARRIAGE, wheel_px, camera.to_screen(position_at(-2.9, 0.12)),
        max(1.0, 0.16 * scale),
    )
    pygame.draw.circle(
        surface, CARRIAGE, wheel_px, wheel_radius_px,
        max(1, wheel_radius_px // 6),
    )

    # The tube is roughly twice the bore across at the reinforce.
    barrel_width_px = max(2.0, 2.0 * gun.bore_diameter_m * scale)
    draw_thick_line(surface, BARREL, breech_px, muzzle_px, barrel_width_px)
    pygame.draw.circle(
        surface, CARRIAGE, trunnion_px, max(1, round(0.09 * scale))
    )


def block_screen_rect(camera, block):
    """The block's DRAWN extent in screen pixels: (left, top, width, height).

    Single source of truth for both drawing and hit-testing, so the grab
    region cannot drift away from the glyph. Includes the minimum-size
    floor, PER AXIS: below it that axis is drawn larger than true scale, and
    the floor-sized glyph is what the cursor must be able to catch.

    PER AXIS, NOT ONE SQUARE FLOOR - batch 16 SS4. The slab is 25 mm thick,
    which is sub-pixel at almost every usable zoom, while its 3.0 m height
    usually is not: at the factory 100 px/m the true footprint is 2.5 px wide
    by 300 px tall. A single square floor would either lose the thickness
    entirely or balloon the height to match it. This is still an
    AXIS-ALIGNED bounding rect - it does not follow theta - which is an
    existing simplification inherited here, not introduced by it.
    """
    centre_px = camera.to_screen(block.pos)
    scale = camera.scale_px_per_m
    apparent_w_px = max(MIN_DRAWN_SIZE_PX, 2.0 * block.half_width * scale)
    apparent_h_px = max(MIN_DRAWN_SIZE_PX, 2.0 * block.half_height * scale)
    return (
        centre_px[0] - 0.5 * apparent_w_px,
        centre_px[1] - 0.5 * apparent_h_px,
        apparent_w_px,
        apparent_h_px,
    )


def _draw_block(surface, camera, block):
    """The target block, drawn from its true rotated corners.

    FLOORED PER AXIS, IN WORLD METRES, BEFORE ROTATING - batch 16 SS4. The old
    scheme floored the WHOLE GLYPH to an axis-aligned square below one
    threshold, which lost theta entirely: a toppled block drawn that way
    would not visibly show its topple. That was tolerable for a square block,
    where "below the floor" meant "too far away to care about orientation".
    It is not tolerable for the slab, whose 25 mm thickness is sub-pixel at
    almost every usable zoom - THIS IS THE PRIMARY DRAW PATH for this target,
    not a rare edge case, and losing rotation on it would hide the one thing
    section 4 exists to show.

    So each axis's half-extent is floored independently, in world units,
    before the corner rotation is applied - the shape still turns with theta,
    it just never collapses to invisibility on the thin axis. POSITION STAYS
    EXACT throughout; only the glyph's size is ever clamped.
    """
    scale = camera.scale_px_per_m
    half_floor_m = 0.5 * MIN_DRAWN_SIZE_PX / scale
    half_width = max(block.half_width, half_floor_m)
    half_height = max(block.half_height, half_floor_m)

    corners_px = [
        camera.to_screen(
            block.pos + Vec2(sx * half_width, sy * half_height).rotated(block.theta)
        )
        for sx, sy in _BLOCK_CORNER_SIGNS
    ]
    pygame.draw.polygon(surface, BLOCK, corners_px)
    pygame.draw.polygon(surface, BLOCK_EDGE, corners_px, 1)
    # A mark on one face, so rotation is visible when the block rocks.
    first, second = corners_px[0], corners_px[1]
    draw_thick_line(surface, BLOCK_EDGE, first, second, 3.0, round_caps=False)


#: Predicted-landing marker, screen px. Drawn in screen space at a constant
#: size so it stays legible at every zoom, like the corner brackets.
LANDING_MARKER_PX = 11


def _draw_landing_marker(instrument, camera, position_m):
    """Where the shot is predicted to reach the ground.

    AN OPEN CROSS FROM THE REGISTRY, not a chevron. The chevron was a filled
    triangle standing ON the point with a bare number floating above it: it hid
    the thing it marked, and nothing joined the number to the glyph. Both of
    those are now invariants rather than choices - see `hud`.

    It matters more here than anywhere else, because the target is sighted onto
    the predicted impact point, so the marker and the target genuinely share
    pixels in the default scenario. Only a gapped glyph keeps them distinct.
    """
    point_px = camera.to_screen(position_m)
    if not _on_screen(instrument.surface, point_px, margin_px=60):
        return
    rect = instrument.open_cross(point_px, hud.PREDICTED)
    instrument.submit_label(
        f"{downrange_m(position_m):.0f} m", point_px, hud.PREDICTED,
        priority=40, glyph_rect=rect,
    )


def _draw_trail(surface, camera, trail):
    if len(trail) < 2:
        return
    points = [camera.to_screen(p) for p in trail]
    pygame.draw.lines(surface, TRAIL, False, points, 1)


def _draw_shot(surface, camera, state, *, colour=SHOT):
    """Position exact; glyph size clamped to the minimum drawn size."""
    centre_px = camera.to_screen(state.pos)
    true_diameter_px = 2.0 * state.radius * camera.scale_px_per_m
    radius_px = max(MIN_DRAWN_SIZE_PX, round(true_diameter_px)) * 0.5
    pygame.draw.circle(surface, colour, centre_px, radius_px)


def _draw_crosshair(surface):
    width_px, height_px = surface.get_size()
    cx, cy = width_px // 2, height_px // 2
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        pygame.draw.line(
            surface, CROSSHAIR,
            (cx + dx * 10, cy + dy * 10), (cx + dx * 20, cy + dy * 20), 1,
        )


_HUD_KEYS = (
    "A load    SPACE fire    R clear bore    C clear field    "
    "UP/DOWN elevation (SHIFT fine)    LEFT/RIGHT charge",
    "TAB focus next    click to focus    HOME recentre on focus    "
    "J/L pan across    I/K pan up-down    right-drag pans",
    "M map    D drag    { } time scale    wheel or +/- zoom    "
    "F fullscreen    S settings    ESC quit",
)

#: Rows on the settings screen beyond the fields themselves.
SETTINGS_ACTIONS = ("Reset to factory defaults", "Close and save")


def draw_settings(surface, fonts, settings, selected_index):
    """Draw the modal settings screen over a dimmed scene.

    Reads only; selection lives in the loop. Deliberately pygame text and
    rects in the existing HUD style rather than a UI toolkit - a settings
    screen is not worth a dependency.

    THE PENDING NOTE IS GONE, replaced by a permanent footer line. It was
    shown only while `settings_dirty` was set, and it was that conditional
    presentation that let it stay wrong for three batches: it named R as the
    apply key long after R had become reload, and nobody saw the sentence
    often enough to notice. A line that is always on screen is checked every
    time the screen is opened.
    """
    width_px, height_px = surface.get_size()
    veil = pygame.Surface((width_px, height_px), pygame.SRCALPHA)
    veil.fill((*HUD_PANEL, 225))
    surface.blit(veil, (0, 0))

    font = fonts["mono"]
    small = fonts["small"]
    line_h = font.get_height() + 6
    rows = len(FIELDS) + len(SETTINGS_ACTIONS)

    panel_w = min(760, width_px - 60)
    panel_h = line_h * rows + font.get_height() * 2 + small.get_height() * 4 + 60
    left = (width_px - panel_w) // 2
    top = max(20, (height_px - panel_h) // 2)

    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((*HUD_PANEL, 245))
    pygame.draw.rect(panel, HUD_DIM, panel.get_rect(), 1)
    surface.blit(panel, (left, top))

    y = top + 16
    surface.blit(font.render("Settings", True, HUD_TEXT), (left + 22, y))
    y += font.get_height() + 10

    for index, field in enumerate(FIELDS):
        selected = index == selected_index
        if selected:
            pygame.draw.rect(
                surface, (44, 54, 70), (left + 12, y - 3, panel_w - 24, line_h - 2)
            )
        colour = HUD_TEXT if selected else HUD_DIM
        marker = ">" if selected else " "
        surface.blit(font.render(f"{marker} {field.label}", True, colour),
                     (left + 22, y))
        value_surface = font.render(format_value(settings, field), True, HUD_TEXT)
        surface.blit(value_surface,
                     (left + panel_w - 26 - value_surface.get_width(), y))
        y += line_h

    y += 6
    for offset, action in enumerate(SETTINGS_ACTIONS):
        index = len(FIELDS) + offset
        selected = index == selected_index
        if selected:
            pygame.draw.rect(
                surface, (44, 54, 70), (left + 12, y - 3, panel_w - 24, line_h - 2)
            )
        colour = HUD_TEXT if selected else HUD_DIM
        marker = ">" if selected else " "
        surface.blit(font.render(f"{marker} [ {action} ]", True, colour),
                     (left + 22, y))
        y += line_h

    y += 8
    tooltip = (
        FIELDS[selected_index].tooltip
        if selected_index < len(FIELDS)
        else f"Press ENTER to {SETTINGS_ACTIONS[selected_index - len(FIELDS)].lower()}."
    )
    for line in _wrap(tooltip, 78):
        surface.blit(small.render(line, True, MARKER_TEXT), (left + 22, y))
        y += small.get_height() + 2

    y += 4
    # Two lines, as batch 15 SS0 specifies them. What each correction fixes:
    #
    #   "ENTER activate" did not say activate WHAT. Only the two bracketed
    #   rows respond to it; on a field row it does nothing, so the hint was
    #   promising a behaviour that eleven of thirteen rows do not have.
    #
    #   "C applies" listed a SIMULATION key in a settings footer. C is pressed
    #   after leaving this screen, so it belongs in a sentence about when
    #   changes land, not in the key list.
    #
    #   S is gone from the hint AND from the key handler. It is the key that
    #   OPENS this screen; having it also close it was a second way to do what
    #   ESC already does.
    for line in (
        "UP/DOWN select    LEFT/RIGHT adjust (SHIFT fine)    "
        "ENTER on [bracketed] rows    ESC close and save",
        "Changes take effect on the next CLEAR FIELD (C).",
    ):
        surface.blit(small.render(line, True, HUD_DIM), (left + 22, y))
        y += small.get_height() + 2


def _wrap(text, width):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


#: Worst-case status strings the loop can emit, used to size the HUD panel.
#:
#: MEASURED, NOT GUESSED. These are the longest form of every status message
#: in `main`, with the widest plausible numbers substituted: a four-figure
#: downrange, a three-figure speed, and signed two-decimal block motion.
#: `_hud_panel_width` renders them and takes the maximum, so the panel grows
#: if a message ever gets longer rather than silently truncating it.
_WIDEST_STATUS = (
    "struck the block at 1000 m/s, block 12.34 m/s -12.34 rad/s",
    "impact at 19740 m, 1000 m/s",
    "in flight",
    "ready",
)


def _hud_panel_width(fonts):
    label_width = max(
        fonts["mono"].size(label)[0]
        for label in (
            "elevation", "charge", "muzzle velocity", "speed", "Mach",
            "altitude", "downrange", "flight path", "flight time", "block",
            "block spin",
            "rounds on field",
        )
    )
    value_width = max(
        fonts["mono"].size(value)[0]
        for value in ("19740.0 m", "-12.34 rad/s", "1000.0 m/s", "  0.000")
    )
    status_width = max(fonts["small"].size(text)[0] for text in _WIDEST_STATUS)
    return max(label_width + value_width + 44, status_width + 32)


def _draw_hud(surface, values, fonts):
    rows = (
        ("elevation", f"{values.elevation_deg:7.2f} deg"),
        ("charge", f"{values.charge_kg:7.2f} kg"),
        ("muzzle velocity", f"{values.muzzle_velocity_ms:7.1f} m/s"),
        ("speed", f"{values.speed_ms:7.1f} m/s"),
        ("Mach", f"{values.mach:7.3f}"),
        ("altitude", f"{values.altitude_m:7.1f} m"),
        ("downrange", f"{values.downrange_m:7.1f} m"),
        ("flight path", f"{values.flight_path_deg:7.2f} deg"),
        ("flight time", f"{values.flight_time_s:7.2f} s"),
        # Names WHICH target is moving. The field is one slab by default, but
        # `blocks` stays plural for batch 10, and "block 3.20 m/s" would not
        # say which target took the hit once a second one exists.
        (values.block_label or "target", f"{values.block_speed_ms:7.2f} m/s"),
        ("target spin", f"{values.block_omega_rads:7.2f} rad/s"),
        ("rounds on field", f"{values.rounds_on_field:7d}"),
        ("focus", f"{values.focus_label:>7s}"),
    )
    font = fonts["mono"]
    small = fonts["small"]
    line_h = font.get_height() + 2
    panel_w = _hud_panel_width(fonts)
    panel_h = line_h * len(rows) + small.get_height() * 2 + 26

    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((*HUD_PANEL, 190))
    surface.blit(panel, (12, 12))

    y = 20
    for label, value in rows:
        surface.blit(font.render(label, True, HUD_DIM), (22, y))
        value_surface = font.render(value, True, HUD_TEXT)
        surface.blit(value_surface, (12 + panel_w - 14 - value_surface.get_width(), y))
        y += line_h

    y += 6
    drag_text = "drag on" if values.drag_enabled else "DRAG OFF (vacuum)"
    surface.blit(
        small.render(f"{drag_text}   time x{values.time_scale:g}", True, HUD_DIM), (22, y)
    )
    y += small.get_height() + 2
    surface.blit(small.render(values.status, True, HUD_TEXT), (22, y))

    width_px, height_px = surface.get_size()
    for i, line in enumerate(reversed(_HUD_KEYS)):
        text = small.render(line, True, HUD_DIM)
        surface.blit(text, (14, height_px - 14 - (i + 1) * (text.get_height() + 2)))


# --------------------------------------------------------------------------
# Zoomed-out drawing. Awaiting SS8's per-object substitution, which subsumes it.
# --------------------------------------------------------------------------


def draw_map(surface, camera, gun, state, hud_values, fonts, *, blocks=(),
             spent_shots=(), mu_m3_s2=None, focus_position_m=None):
    """Planet as a disc, bodies as icons, orbit as a conic.

    NO LONGER CALLED. Batch 16 SS1 united the two cameras and the one draw path
    is `draw_scene`, which picks its background from the scale. This is kept
    rather than deleted because SS8 - per-object marker substitution - is about
    to need exactly this icon-and-conic drawing, generalised to decide per object
    instead of per regime. It goes when SS8 lands, or sooner on request.

    Everything below the first paragraph describes the regime that no longer
    exists, and is left as the record of what SS8 has to subsume.

    A DISTINCT REPRESENTATION, NOT A ZOOM LEVEL. At map scale the gun is
    3e-4 px and the whole factory trajectory is 0.09 px, so nothing drawn at
    true scale is visible at all. Bodies therefore become fixed-size icons
    and the trajectory becomes a drawn conic rather than a sampled path.

    The 4 px minimum-size floor and the corner brackets from batch 5 already
    solve "correctly placed and invisible" and are reused rather than
    reinvented.
    """
    surface.fill(MAP_SPACE)
    _draw_planet_disc(surface, camera)

    if state is not None and mu_m3_s2 is not None:
        _draw_conic(surface, camera, state, mu_m3_s2, fonts)

    _draw_map_icon(surface, camera, gun.trunnion_m, MARKER_TEXT, "gun")
    for block in blocks:
        _draw_map_icon(surface, camera, block.pos, BLOCK_EDGE, block.label)
    for spent in spent_shots:
        _draw_map_icon(surface, camera, spent.pos, SPENT_SHOT, None)
    if state is not None:
        _draw_map_icon(surface, camera, state.pos, SHOT, None)
    if focus_position_m is not None:
        _draw_focus_ring(surface, camera, focus_position_m)

    _draw_hud(surface, hud_values, fonts)


def _draw_planet_disc(surface, camera):
    """The planet, drawn only where it can reach the viewport.

    CLIPPED IN WORLD SPACE, NOT HANDED WHOLE TO THE RASTERISER. Raising the
    map ceiling to 1.0 px/m made the naive version unusable: the full disc
    is a polygon spanning 1.27e7 pixels, and pygame rasterises all of it
    even when a sliver intersects the screen. Measured 9 ms per frame at the
    fit zoom, 48 ms at 1e-2, 471 ms at 1e-1 - linear in zoom, so about 4.7
    SECONDS per frame at the new ceiling.

    Below the crossover the whole disc is drawn as a circle. Above it, only
    the arc that spans the viewport is sampled, closed inward far enough to
    fill the screen. Cost is then flat in zoom rather than linear.
    """
    from .planet import R_EARTH_M

    width_px, height_px = surface.get_size()
    radius_px = R_EARTH_M * camera.scale_px_per_m
    diagonal_px = math.hypot(width_px, height_px)

    if radius_px <= 3.0 * diagonal_px:
        points = [
            camera.to_screen(
                Vec2(
                    math.cos(2.0 * math.pi * i / MAP_CIRCLE_SAMPLES) * R_EARTH_M,
                    math.sin(2.0 * math.pi * i / MAP_CIRCLE_SAMPLES) * R_EARTH_M,
                )
            )
            for i in range(MAP_CIRCLE_SAMPLES)
        ]
        if _any_finite(points):
            pygame.draw.polygon(surface, MAP_PLANET, points)
            pygame.draw.lines(surface, MAP_PLANET_EDGE, True, points, 1)
        return

    # Zoomed in past the crossover: draw the local arc only.
    centre_angle = math.atan2(camera.center_m.y, camera.center_m.x)
    half_span_rad = 0.6 * diagonal_px / radius_px
    depth_m = 2.0 * diagonal_px / camera.scale_px_per_m

    arc = []
    for index in range(MAP_CIRCLE_SAMPLES):
        angle = (
            centre_angle
            - half_span_rad
            + 2.0 * half_span_rad * index / (MAP_CIRCLE_SAMPLES - 1)
        )
        arc.append(Vec2(math.cos(angle), math.sin(angle)) * R_EARTH_M)

    filled = arc + [
        arc[-1] * ((R_EARTH_M - depth_m) / R_EARTH_M),
        arc[0] * ((R_EARTH_M - depth_m) / R_EARTH_M),
    ]
    points = [camera.to_screen(p) for p in filled]
    if _any_finite(points):
        pygame.draw.polygon(surface, MAP_PLANET, points)
        pygame.draw.lines(
            surface, MAP_PLANET_EDGE, False,
            [camera.to_screen(p) for p in arc], 1,
        )


def _draw_conic(surface, camera, state, mu_m3_s2, fonts):
    """The osculating orbit, drawn whole, plus apsides and ground crossings.

    THE ELLIPSE IS MOSTLY INSIDE THE PLANET and that is the entire point.
    The factory shot's orbit has a 3186 km semi-major axis - half the
    Earth's radius - with periapsis essentially at the centre and apoapsis
    341 m up. The flown arc is the sliver above the surface. The parabola
    was never real; the ground merely got in the way.

    LABELLED OSCULATING, not predicted. With drag this is the ellipse the
    shot is instantaneously on, tangent now and diverging immediately. It is
    drawn alongside the flown path in the surface regime so the divergence
    is visible rather than hidden.
    """
    from . import orbit
    from .planet import R_EARTH_M

    elements = orbit.elements_from_state(state.pos, state.vel, mu_m3_s2)
    if not elements.is_closed:
        # Open orbits are guarded, not drawn. Batch 9 never produces one;
        # the probe will, and the renderer must do something other than
        # silently skip - that is batch 11 SS7.
        label = fonts["small"].render(
            "hyperbolic - not drawn (batch 11)", True, MAP_CONIC
        )
        surface.blit(
            label, (surface.get_width() - label.get_width() - 14, 32)
        )
        return

    # CLIPPED AT THE SURFACE, and drawn as OPEN polylines. Closing the loop
    # here would draw a chord straight back through the planet between the two
    # ground crossings, which is the artifact the clipping exists to remove.
    for polyline in orbit.polylines_above_radius(elements, R_EARTH_M, count=256):
        points = [camera.to_screen(p) for p in polyline]
        if len(points) >= 2 and _any_finite(points):
            pygame.draw.lines(surface, MAP_CONIC, False, points, 1)

    for anomaly, colour, name in (
        (0.0, MAP_CONIC, "peri"),
        (math.pi, MAP_CONIC, "apo"),
    ):
        position, _ = orbit.state_at(elements, anomaly)
        # An apsis marker is conic geometry too. For every cannon shot
        # periapsis sits near the Earth's CENTRE, so this suppresses a "peri"
        # ring drawn in the middle of the planet - gate 99 counts it.
        if position.length() < R_EARTH_M:
            continue
        point_px = camera.to_screen(position)
        if _on_screen(surface, point_px):
            pygame.draw.circle(surface, colour, point_px, MAP_APSIS_PX, 1)
            surface.blit(
                fonts["small"].render(name, True, colour),
                (point_px[0] + 6, point_px[1] - 6),
            )

    for anomaly in orbit.radius_crossings(elements, R_EARTH_M):
        position, _ = orbit.state_at(elements, anomaly)
        point_px = camera.to_screen(position)
        if _on_screen(surface, point_px):
            pygame.draw.circle(surface, MAP_CONIC_FLOWN, point_px, 3)

    text = fonts["small"].render(
        f"osculating: a = {elements.semi_major_axis_m / 1000:.0f} km, "
        f"e = {elements.eccentricity:.4f}, "
        f"period {elements.period_s / 60:.1f} min",
        True, MAP_CONIC,
    )
    # Top right, clear of the HUD panel on the left and the key hints along
    # the bottom - which are now three lines deep and were being overlapped.
    surface.blit(text, (surface.get_width() - text.get_width() - 14, 14))


def _draw_map_icon(surface, camera, position_m, colour, label):
    """A body at map scale: a fixed-size icon, because true scale is invisible."""
    point_px = camera.to_screen(position_m)
    if not _on_screen(surface, point_px, margin_px=40):
        return
    pygame.draw.circle(surface, colour, point_px, MAP_ICON_PX)
    pygame.draw.circle(surface, (12, 14, 18), point_px, MAP_ICON_PX, 1)
    if label:
        surface.blit(
            fonts_label(surface, label, colour),
            (point_px[0] + MAP_ICON_PX + 4, point_px[1] - 7),
        )


_LABEL_CACHE = {}


def fonts_label(surface, text, colour):
    """Tiny cached label renderer, so icons can be labelled without a font arg."""
    key = (text, colour)
    if key not in _LABEL_CACHE:
        font = pygame.font.SysFont("consolas,dejavusansmono,couriernew", 13)
        _LABEL_CACHE[key] = font.render(text, True, colour)
    return _LABEL_CACHE[key]


def _any_finite(points):
    """True if the polygon is drawable.

    Screen coordinates at map scale run to millions of pixels for anything
    off-frame, and a naive int() conversion inside pygame would wrap. Points
    are checked in world space before they get that far - gate 32.
    """
    limit = 1e7
    return all(
        math.isfinite(x) and math.isfinite(y) and abs(x) < limit and abs(y) < limit
        for x, y in points
    )
