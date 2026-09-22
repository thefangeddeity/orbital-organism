"""Instrument symbology. Every glyph and every label in the program.

Idiom: Orbiter / IMFD instrument, by PM decision. Thin strokes, no fills, muted,
monospace labels on leader lines, ticks rather than arrows.

WHY A REGISTRY RATHER THAN A NICER ARROW. The complaint about the landing
chevron was not that it was ugly. It was the only member of its own species: one
glyph invented in isolation, sitting on top of the thing it marked, with a bare
number floating above it and nothing joining the two. A better-looking arrow
reproduces every one of those properties. So the glyphs live here, there are five
of them, and after batch 16 nothing in the HUD path draws directly.

TWO INVARIANTS, both asserted by gate 111:

    NON-OCCLUSION. No glyph writes the pixel it marks. Anything that would is
    drawn open or gapped. This is the direct answer to the marker covering the
    target - and the two genuinely share pixels, because the target is sighted
    onto the predicted impact point, so they are only distinguishable if neither
    fills the other's centre.

    LABELS NEVER FLOAT. Every number is attached to its glyph by a leader, in
    one monospace size, and is submitted rather than blitted.

CONSTANT PIXEL SIZE. Every glyph is the same size on screen at every zoom,
because a glyph is a symbol for a thing and not a picture of it. The moment a
glyph scales with zoom it becomes geometry, and geometry is what marker
substitution exists to replace.

HUE CARRIES MEANING, NOT IDENTITY. Four classes and no more. The
predicted-versus-actual split is the distinction this project cares about most -
osculating conic against integrated path - so the eye gets it free.

The layout resolver is batch 16 SS6. `submit_label` takes a rectangle and a
priority now and draws in submission order; SS6 fills in collision resolution
behind the same interface without changing a call site.
"""

import math

import pygame

# --------------------------------------------------------------------------
# Colour classes. Four, and every drawn thing declares one.
# --------------------------------------------------------------------------

#: Anything the model expects but has not yet done: predicted impact, the
#: osculating conic, apsides, ground crossings.
PREDICTED = (255, 184, 64)

#: Anything that happened: the flown trail, contact points, measured readouts.
ACTUAL = (96, 224, 128)

#: The current selection, and its label.
SELECTED = (236, 240, 244)

#: No solution, out of range, clamped, hyperbolic.
CAUTION = (240, 96, 64)

CLASSES = (PREDICTED, ACTUAL, SELECTED, CAUTION)

CLASS_NAMES = {
    PREDICTED: "PREDICTED",
    ACTUAL: "ACTUAL",
    SELECTED: "SELECTED",
    CAUTION: "CAUTION",
}

# --------------------------------------------------------------------------
# Stroke weights, in pixels, never scaled.
# --------------------------------------------------------------------------

#: Leaders, conics, ticks.
STROKE_HAIR = 1

#: Glyph outlines.
STROKE_GLYPH = 2

#: Selection brackets ALONE, so selection reads heavier without a fifth colour.
STROKE_SELECT = 3

# --------------------------------------------------------------------------
# Glyph geometry. Starting values, adjusted for legibility; see the report.
# --------------------------------------------------------------------------

CROSS_ARM_PX = 9
CROSS_GAP_PX = 3          # the marked pixel stays visible - non-occlusion
BRACKET_ARM_PX = 8
BRACKET_STANDOFF_PX = 4
BRACKET_MIN_BOX_PX = 12
TICK_PX = 5
CARET_PX = 7
LEADER_PX = 12
LEADER_ANGLE_RAD = math.radians(45.0)

#: THE MARKER GLYPH - batch 16 Part II SS8. What an object becomes below its
#: own geometry threshold: an open ring, not a filled dot, for the same
#: non-occlusion reason as every other glyph here - the marked point stays
#: visible at its own centre. Smaller than the cross and the bracket, since
#: it stands for "there is an object here" rather than a specific event or
#: selection; measured and reported in the SS8 report alongside the
#: geometry/marker size threshold it pairs with.
MARKER_RADIUS_PX = 4

#: How far inside the viewport edge a caret sits, px.
CARET_INSET_PX = 10


#: True while an `Instrument` is emitting through pygame. Gate 111's closure
#: check reads it: any pygame draw call made while this is False, during the HUD
#: phase, came from somewhere other than the registry.
#:
#: A FLAG RATHER THAN A GREP. Batch 8's lesson is that what a call site looks
#: like and what it binds are different questions, and grep answers the first
#: one. A counting shim answers the second.
_EMITTING = False


class _Emit:
    """Context manager marking a region as registry-owned drawing."""

    def __enter__(self):
        global _EMITTING
        self._previous = _EMITTING
        _EMITTING = True

    def __exit__(self, *exc):
        global _EMITTING
        _EMITTING = self._previous
        return False


def emitting():
    """Whether a registry glyph is being drawn right now. For gate 111."""
    return _EMITTING


class Instrument:
    """The one place glyphs and labels are drawn.

    Glyphs draw immediately. Labels are SUBMITTED with a rectangle and a
    priority and drawn on `flush`, so batch 16 SS6 can insert a collision
    resolver behind the same interface without touching a call site.
    """

    def __init__(self, surface, font):
        self.surface = surface
        self.font = font
        self._labels = []
        #: Every glyph drawn this frame, as (kind, colour, rect). Gate 108
        #: counts selection glyphs from it; gate 111 checks sizes and hues.
        self.drawn = []

    # -- glyphs ---------------------------------------------------------
    def open_cross(self, point_px, colour):
        """A gapped cross marking a point. THE PREDICTED IMPACT MARKER.

        The gap is the whole point: the marked pixel stays visible, so the
        marker and the thing it marks are never confusable. The chevron this
        replaces sat on top of the target and hid it.
        """
        x, y = point_px
        with _Emit():
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                pygame.draw.line(
                    self.surface, colour,
                    (x + dx * CROSS_GAP_PX, y + dy * CROSS_GAP_PX),
                    (x + dx * CROSS_ARM_PX, y + dy * CROSS_ARM_PX),
                    STROKE_GLYPH,
                )
        rect = (x - CROSS_ARM_PX, y - CROSS_ARM_PX,
                2 * CROSS_ARM_PX, 2 * CROSS_ARM_PX)
        self.drawn.append(("open_cross", colour, rect))
        return rect

    def brackets(self, point_px, extent_px, colour=SELECTED):
        """Four corner Ls around a body. THE SELECTION GLYPH.

        Open at the centre and on every side, so the body inside stays fully
        visible - the same non-occlusion argument as the cross. The focus ring
        this replaces was a closed circle drawn over the body.
        """
        x, y = point_px
        half = max(0.5 * extent_px + BRACKET_STANDOFF_PX,
                   0.5 * BRACKET_MIN_BOX_PX)
        with _Emit():
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                corner = (x + sx * half, y + sy * half)
                pygame.draw.line(
                    self.surface, colour, corner,
                    (corner[0] - sx * BRACKET_ARM_PX, corner[1]),
                    STROKE_SELECT,
                )
                pygame.draw.line(
                    self.surface, colour, corner,
                    (corner[0], corner[1] - sy * BRACKET_ARM_PX),
                    STROKE_SELECT,
                )
        rect = (x - half, y - half, 2 * half, 2 * half)
        self.drawn.append(("brackets", colour, rect))
        return rect

    def marker(self, point_px, colour):
        """An open ring standing in for an object too small to draw as itself.

        THE SUBSTITUTION GLYPH - batch 16 Part II SS8. Below its own drawn-
        geometry threshold, every object decides for itself to become this
        instead of a shrinking, eventually-invisible true-scale shape. Open,
        not filled, so the object's own position is never occluded by the
        symbol standing in for it - the same invariant `open_cross` and
        `brackets` already keep.
        """
        with _Emit():
            pygame.draw.circle(
                self.surface, colour, point_px, MARKER_RADIUS_PX, STROKE_GLYPH
            )
        rect = (
            point_px[0] - MARKER_RADIUS_PX, point_px[1] - MARKER_RADIUS_PX,
            2 * MARKER_RADIUS_PX, 2 * MARKER_RADIUS_PX,
        )
        self.drawn.append(("marker", colour, rect))
        return rect

    def tick(self, point_px, normal_px, colour):
        """A short stroke perpendicular to a curve or the ground.

        ONE SHAPE FOR THREE THINGS: apsides, ground crossings and scale
        graduations. They are the same idea - a marked place on a line - and
        three glyphs for one idea is the duplication this batch removes.
        """
        length = math.hypot(*normal_px) or 1.0
        nx, ny = normal_px[0] / length, normal_px[1] / length
        x, y = point_px
        with _Emit():
            pygame.draw.line(
                self.surface, colour,
                (x - nx * TICK_PX * 0.5, y - ny * TICK_PX * 0.5),
                (x + nx * TICK_PX * 0.5, y + ny * TICK_PX * 0.5),
                STROKE_HAIR,
            )
        rect = (x - TICK_PX, y - TICK_PX, 2 * TICK_PX, 2 * TICK_PX)
        self.drawn.append(("tick", colour, rect))
        return rect

    def edge_caret(self, point_px, colour):
        """An open triangle at the viewport border, pointing off-screen.

        EXACTLY ONE PER OFF-SCREEN ITEM, always inside the viewport. Drawn open
        rather than filled, for the same reason as everything else here.
        """
        width_px, height_px = self.surface.get_size()
        x = min(max(point_px[0], CARET_INSET_PX), width_px - CARET_INSET_PX)
        y = min(max(point_px[1], CARET_INSET_PX), height_px - CARET_INSET_PX)
        # Point along the direction the item lies in.
        dx = point_px[0] - x
        dy = point_px[1] - y
        length = math.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        tip = (x + ux * CARET_PX, y + uy * CARET_PX)
        left = (x - uy * CARET_PX * 0.6, y + ux * CARET_PX * 0.6)
        right = (x + uy * CARET_PX * 0.6, y - ux * CARET_PX * 0.6)
        with _Emit():
            pygame.draw.lines(
                self.surface, colour, False, (left, tip, right), STROKE_GLYPH
            )
        rect = (x - CARET_PX, y - CARET_PX, 2 * CARET_PX, 2 * CARET_PX)
        self.drawn.append(("edge_caret", colour, rect))
        return rect

    # -- labels ---------------------------------------------------------
    def submit_label(self, text, anchor_px, colour, *, priority=0,
                     glyph_rect=None):
        """Submit a label to be drawn attached to its glyph by a leader.

        NEVER BLITTED DIRECTLY, and never without a leader. A bare number
        floating near a glyph is what the chevron did.

        `priority` is carried for batch 16 SS6's resolver, which suppresses the
        lower-priority of two colliding labels. Until it exists, labels draw in
        submission order.
        """
        self._labels.append((text, anchor_px, colour, priority, glyph_rect))

    def flush(self):
        """Draw every submitted label, each joined to its anchor by a leader.

        Returns the list of (text, label_rect, leader) actually drawn, which is
        what gate 111 inspects.
        """
        width_px, height_px = self.surface.get_size()
        drawn = []
        with _Emit():
            for text, anchor_px, colour, _priority, glyph_rect in self._labels:
                # The leader runs up and outboard at 45 degrees, flipping side
                # so the label stays on screen.
                direction = -1.0 if anchor_px[0] > width_px * 0.75 else 1.0
                run = LEADER_PX * math.cos(LEADER_ANGLE_RAD) * direction
                rise = -LEADER_PX * math.sin(LEADER_ANGLE_RAD)
                # Start the leader clear of the glyph so it does not cross it.
                start_gap = 0.0
                if glyph_rect is not None:
                    start_gap = 0.5 * math.hypot(glyph_rect[2], glyph_rect[3])
                start = (
                    anchor_px[0] + math.cos(LEADER_ANGLE_RAD) * direction * start_gap,
                    anchor_px[1] - math.sin(LEADER_ANGLE_RAD) * start_gap,
                )
                end = (start[0] + run, start[1] + rise)
                pygame.draw.line(
                    self.surface, colour, start, end, STROKE_HAIR
                )
                surface_text = self.font.render(text, True, colour)
                left = end[0] if direction > 0 else end[0] - surface_text.get_width()
                top = end[1] - surface_text.get_height()
                left = min(max(left, 0), width_px - surface_text.get_width())
                top = min(max(top, 0), height_px - surface_text.get_height())
                self.surface.blit(surface_text, (left, top))
                drawn.append((
                    text,
                    (left, top, surface_text.get_width(),
                     surface_text.get_height()),
                    (start, end),
                ))
        self._labels.clear()
        return drawn
