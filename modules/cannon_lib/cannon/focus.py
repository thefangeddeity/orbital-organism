"""What the camera is looking at.

THE ANCHOR IS A BODY REFERENCE, NOT A POSITION. Until batch 15 the surface
camera was constructed around `gun.trunnion_m` and could never look at anything
else, so every new device on the ladder - probe, rover, rocket - would have
needed its own camera work to become viewable. Focus is that work done once:
the loop keeps a focus KEY, this module turns the live world into the set of
things that key can name, and the camera is pointed at whichever one it names.

FOCUSING DOES NOT MOVE THE VIEW. Focus is a selection; HOME is the movement.
Clicking a body while lining up a shot must not throw the camera somewhere
else, and the standing rule against cameras moving of their own accord applies
to a click as much as to an animation.

Keys are STABLE STRINGS, not indices into a list that changes shape. The
dynamic set gains a shot on fire and loses it on impact, and gains a spent
round each time; an index would silently re-point at a different body the
moment the list shifted. `spent:3` either exists or it does not.

Imports `vec` and the standard library. No pygame and no camera type - `pick`
takes a projection function rather than a camera, so the headless report path
never drags the render layer in.
"""

import math
from dataclasses import dataclass

from .vec import Vec2

#: Where a focused body sits on screen, as a fraction of the viewport.
#:
#: The gun keeps the low-left framing it has had since stage 3: a gun is aimed
#: downrange, so the space in front of it is the space worth showing. Every
#: other body is centred, because nothing about a block or a probe says which
#: way to leave room.
GUN_ANCHOR_FRAC = (0.18, 0.82)
CENTRED_ANCHOR_FRAC = (0.5, 0.5)

#: Click radius for picking a body, screen px. Generous on purpose: at map
#: scale a body is a 5 px icon, and at surface scale a distant block sits at
#: the 4 px minimum-size floor. Picking has to work against what is DRAWN,
#: which is the same argument that sized the block's grab rect.
PICK_RADIUS_PX = 24.0

#: Half-extent Z frames the gun at, m. The piece is about 2 m of tube on a
#: carriage; 4 m puts the whole thing on screen with room to see the ground.
GUN_EXTENT_M = 4.0

#: Floor on a body's framing extent, m. Z on the shot would otherwise ask for a
#: scale beyond the zoom ceiling and simply clamp, which reads as Z doing
#: nothing. The camera clamps anyway; this makes the request sane in the first
#: place.
MIN_EXTENT_M = 0.5


@dataclass(frozen=True, slots=True)
class FocusTarget:
    """One thing the camera can be pointed at."""

    key: str
    label: str
    position_m: Vec2
    anchor_frac: tuple
    #: Half-extent used by Z to frame this body, m. A body has to declare how
    #: big it is or "fit the focused body" has no meaning - the gun and the
    #: planet differ by six decades and a single fit distance would be wrong for
    #: both. Defaulted so a caller that does not care still constructs.
    extent_m: float = 1.0


def targets(gun, blocks=(), shot=None, spent_shots=()):
    """Every body that currently exists, in a stable cycling order.

    Order is gun, then targets, then the round in flight, then spent rounds -
    fixed rather than sorted by distance, so TAB is predictable. A cycle that
    re-ordered itself as bodies moved would make TAB unusable in flight, which
    is exactly when it is wanted.
    """
    found = [
        FocusTarget(
            "gun", "gun", gun.trunnion_m, GUN_ANCHOR_FRAC,
            # The piece plus a little air, so Z on the gun frames the carriage
            # rather than the bore.
            extent_m=GUN_EXTENT_M,
        ),
    ]
    for index, block in enumerate(blocks):
        found.append(
            FocusTarget(
                f"target:{index}",
                getattr(block, "label", None) or f"target {index + 1}",
                block.pos,
                CENTRED_ANCHOR_FRAC,
                # The LARGER of the two half-extents, so Z frames the body's
                # biggest visible dimension - the slab's 1.5 m half-height,
                # not its 12.5 mm half-width, which would zoom in absurdly.
                extent_m=max(
                    getattr(block, "half_width", 1.0),
                    getattr(block, "half_height", 1.0),
                    MIN_EXTENT_M,
                ),
            )
        )
    if shot is not None:
        found.append(
            FocusTarget(
                "shot", "shot", shot.pos, CENTRED_ANCHOR_FRAC,
                extent_m=max(getattr(shot, "radius", MIN_EXTENT_M), MIN_EXTENT_M),
            )
        )
    for index, spent in enumerate(spent_shots):
        found.append(
            FocusTarget(
                f"spent:{index}", f"spent {index + 1}", spent.pos,
                CENTRED_ANCHOR_FRAC,
                extent_m=max(getattr(spent, "radius", MIN_EXTENT_M), MIN_EXTENT_M),
            )
        )
    return tuple(found)


def find(all_targets, key):
    """The target named by `key`, or the first one if it no longer exists.

    FALLS BACK RATHER THAN RETURNING NONE. Bodies leave the world - a shot is
    retired, the field is cleared - and the focus key outlives them. Every
    caller would otherwise need the same None branch, and the one that forgot
    it would crash on the frame after an impact.
    """
    if not all_targets:
        return None
    for target in all_targets:
        if target.key == key:
            return target
    return all_targets[0]


def cycle(all_targets, key, step=1):
    """The key `step` places along from `key`. Wraps.

    Returns the first key when the current one has gone, so TAB after an
    impact lands somewhere real instead of dead-ending.
    """
    if not all_targets:
        return key
    keys = [target.key for target in all_targets]
    if key not in keys:
        return keys[0]
    return keys[(keys.index(key) + step) % len(keys)]


def pick(all_targets, to_screen, point_px, *, radius_px=PICK_RADIUS_PX):
    """The key of the body nearest `point_px`, or None if none is close.

    Nearest wins rather than first-within-radius, so overlapping icons resolve
    to the one actually clicked.
    """
    best_key = None
    best_distance_px = radius_px
    for target in all_targets:
        screen = to_screen(target.position_m)
        distance_px = math.hypot(
            screen[0] - point_px[0], screen[1] - point_px[1]
        )
        if distance_px <= best_distance_px:
            best_key = target.key
            best_distance_px = distance_px
    return best_key
