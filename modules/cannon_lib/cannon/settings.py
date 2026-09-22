"""Scenario and view defaults, persisted between runs.

PHYSICAL CONSTANTS ARE DELIBERATELY NOT HERE. `CD_SUBSONIC`,
`P0_REFERENCE_PA`, restitution, friction, block mass, trunnion geometry, the
atmosphere model and the drag table all stay in source, where they are
documented alongside the reasoning that produced them. Nothing exposed by
this module can invalidate a verification gate, so there is no "modified"
state to flag and no gate re-run to trigger. Keep it that way: the moment a
number that a gate depends on becomes user-editable, every gate result in
the reports stops meaning anything.

This module holds no UI. It defines the values, their limits, and how they
load and save; `render` draws them and `main` binds keys to them.

A settings file must never prevent the simulator from starting. Missing,
malformed, half-written, hand-edited to nonsense - every path ends with a
usable Settings object.
"""

import json
import os
import sys
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

#: Time-scale ladder. Lives here rather than in `main` because it is the
#: clamp for the default-time-scale setting.
TIME_SCALES = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

#: Elevation limits, degrees. The upper bound is the breech-clearance limit
#: established in `gun`: above about 85 degrees the breech face goes through
#: the ground.
ELEVATION_LIMITS_DEG = (0.0, 85.0)

#: Charge limits for the DEFAULT-CHARGE setting, kg.
#:
#: This is the range gate 4 sweeps for monotonicity, and that is the reason
#: for it. Below roughly 0.5 kg the interior ballistics model has no
#: bore-friction term and no windage-leakage floor, so it happily reports
#: muzzle velocities it has no business reporting - the gas simply pushes a
#: frictionless shot down a sealed tube. The clamp keeps the user inside the
#: band where the model means something.
SETTINGS_CHARGE_LIMITS_KG = (0.4, 3.6)

#: Factory zoom values, px/m. Measured at the monitor by the PM.
ZOOM_DEFAULT_PX_PER_M = 100.0



@dataclass
class Settings:
    """Everything the PM can change without a source edit."""

    zoom_default_px_per_m: float = ZOOM_DEFAULT_PX_PER_M
    start_fullscreen: bool = False
    elevation_deg: float = 45.0
    charge_kg: float = 0.4
    drag_enabled: bool = True
    time_scale: float = 1.0
    #: UP/DOWN and SHIFT+UP/DOWN elevation steps, degrees.
    #:
    #: 10 and 1, not 0.5 and 0.05. The old figures came from stage 4, where
    #: laying the gun onto a 0.5 m block was the whole exercise and a 0.05 deg
    #: screw was the instrument for it. The PM's complaint was arithmetic:
    #: 0.5 deg steps meant twenty presses to move ten degrees. The fine step
    #: is an order of magnitude below the coarse one, so UP/DOWN is the
    #: ranging control and SHIFT+UP/DOWN the laying control.
    #:
    #: THE TARGET HAS SINCE CHANGED (batch 16 SS4: one steel slab, 3.0 m tall,
    #: 25 mm thick, replacing the stone box and steel crate). Gate 109
    #: MEASURES the elevation span that still strikes it directly, by
    #: simulating the full flight and CCD sweep - rather than re-deriving it
    #: from an impact-sensitivity figure and a target footprint the way this
    #: comment used to, which does not even translate cleanly to a body this
    #: thin. Read gate 109's report for the current number; it moves with the
    #: target and is not repeated here as a value that could go stale again.
    elevation_coarse_step_deg: float = 10.0
    elevation_fine_step_deg: float = 1.0
    target_enabled: bool = True
    #: True = locked follow at all times. False = fixed at all times.
    #: There is deliberately no transition between the two.
    #:
    #: Factory default is OFF: the shot is not tracked. The corner brackets
    #: and off-screen arrow find the target without the view moving.
    camera_follow: bool = False


@dataclass(frozen=True, slots=True)
class SettingField:
    """Presentation and adjustment metadata for one setting."""

    key: str
    label: str
    kind: str  # "float" | "bool" | "ladder"
    step: float
    fine_step: float
    unit: str
    tooltip: str


FIELDS = (
    SettingField(
        "zoom_default_px_per_m", "Default zoom", "float", 25.0, 5.0, "px/m",
        "Zoom the view opens at. At 100 px/m the 2.00 m barrel reads at 200 px "
        "across a viewport about 13 m wide; the 0.115 m ball is 12 px. Distant "
        "objects fall below their true size and are drawn at a 4 px floor with "
        "corner brackets.",
    ),
    SettingField(
        "start_fullscreen", "Start fullscreen", "bool", 0.0, 0.0, "",
        "Open fullscreen instead of windowed. F or F11 still toggles at runtime.",
    ),
    SettingField(
        "elevation_deg", "Default elevation", "float", 1.0, 0.1, "deg",
        "Elevation the gun is laid at on reset. Capped at 85 deg: above that "
        "the breech face would go through the ground.",
    ),
    SettingField(
        "charge_kg", "Default charge", "float", 0.1, 0.01, "kg",
        "Powder charge on reset. Held to 0.4-3.6 kg: below ~0.5 kg the "
        "interior model has no bore-friction or leakage floor and reports "
        "velocities it cannot justify.",
    ),
    SettingField(
        "drag_enabled", "Drag", "bool", 0.0, 0.0, "",
        "Atmosphere and drag on. Off gives a vacuum trajectory.",
    ),
    SettingField(
        "time_scale", "Default time scale", "ladder", 0.0, 0.0, "x",
        "Playback rate on reset. Physics always steps at 1 ms regardless.",
    ),
    # The row increments moved with the values they edit. Leaving them at 0.5
    # and 0.05 against a factory 10.0 would reproduce the PM's own complaint
    # inside the Settings screen: twenty presses to change the step by half
    # its size.
    SettingField(
        "elevation_coarse_step_deg", "Elevation step", "float", 1.0, 0.5, "deg",
        "How far UP/DOWN moves the elevation. 10 deg is the ranging step.",
    ),
    SettingField(
        "elevation_fine_step_deg", "Elevation fine step", "float", 0.5, 0.1, "deg",
        "How far SHIFT+UP/DOWN moves the elevation. Gate 109 reports the "
        "elevation span that still strikes the slab; compare this against it "
        "if the target starts feeling unlayable.",
    ),
    SettingField(
        "target_enabled", "Target slab", "bool", 0.0, 0.0, "",
        "Emplace the steel slab at all. Its distance is no longer a "
        "setting: it is emplaced on the trajectory at startup and moved by "
        "dragging it with the mouse.",
    ),
    SettingField(
        "camera_follow", "Camera follows shot", "bool", 0.0, 0.0, "",
        "On: the view stays locked on the shot for the whole flight. Off: the "
        "view never moves. There is no transition between the two.",
    ),
)


def factory_settings():
    return Settings()


def clamp_settings(settings):
    """Force every value into range. Returns (settings, warnings)."""
    warnings = []

    def note(key, was, now):
        warnings.append(f"settings: {key} {was!r} out of range, clamped to {now!r}")

    # THE ZOOM BAND IS DERIVED, NOT SETTABLE - batch 16 SS1 replaced the two
    # user-facing bounds with `Camera.zoom_limits`, computed from the viewport
    # and the world extent. The rows lingered for two batches with no consumer,
    # which is a control that lies; Tranche 3 SS2 removed them.
    #
    # Only the DEFAULT survives, and it is clamped to the derived band at use
    # rather than here, because the band depends on a viewport this module does
    # not know about.
    zoom_default = settings.zoom_default_px_per_m
    if not _is_number(zoom_default):
        note("zoom_default_px_per_m", zoom_default, ZOOM_DEFAULT_PX_PER_M)
        zoom_default = ZOOM_DEFAULT_PX_PER_M

    # Fallbacks come from the dataclass defaults, never from literals
    # repeated here - that duplication is what goes stale when the PM
    # revises the factory table.
    factory = Settings()

    elevation = _bounded("elevation_deg", settings.elevation_deg,
                         ELEVATION_LIMITS_DEG, factory.elevation_deg, note)
    charge = _bounded("charge_kg", settings.charge_kg,
                      SETTINGS_CHARGE_LIMITS_KG, factory.charge_kg, note)

    coarse = settings.elevation_coarse_step_deg
    if not _is_number(coarse) or coarse <= 0.0:
        note("elevation_coarse_step_deg", coarse, factory.elevation_coarse_step_deg)
        coarse = factory.elevation_coarse_step_deg
    fine = settings.elevation_fine_step_deg
    if not _is_number(fine) or fine <= 0.0:
        note("elevation_fine_step_deg", fine, factory.elevation_fine_step_deg)
        fine = factory.elevation_fine_step_deg

    time_scale = settings.time_scale
    if time_scale not in TIME_SCALES:
        nearest = min(
            TIME_SCALES,
            key=lambda t: abs(t - time_scale) if _is_number(time_scale) else 0.0,
        ) if _is_number(time_scale) else 1.0
        note("time_scale", time_scale, nearest)
        time_scale = nearest

    return (
        replace(
            settings,
            zoom_default_px_per_m=float(zoom_default),
            start_fullscreen=bool(settings.start_fullscreen),
            elevation_deg=float(elevation),
            charge_kg=float(charge),
            drag_enabled=bool(settings.drag_enabled),
            time_scale=float(time_scale),
            elevation_coarse_step_deg=float(coarse),
            elevation_fine_step_deg=float(fine),
            target_enabled=bool(settings.target_enabled),
            camera_follow=bool(settings.camera_follow),
        ),
        warnings,
    )


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _bounded(key, value, limits, fallback, note):
    low, high = limits
    if not _is_number(value):
        note(key, value, fallback)
        return fallback
    bounded = min(max(value, low), high)
    if bounded != value:
        note(key, value, bounded)
    return bounded


def settings_path():
    """%APPDATA%\\cannon\\settings.json, with a POSIX fallback."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "cannon" / "settings.json"
    return Path.home() / ".cannon" / "settings.json"


def load_settings(path=None, warn=None):
    """Load settings, falling back to factory defaults on any problem.

    Never raises and never returns None. A malformed file is NOT overwritten
    here - it is left alone until the user explicitly saves, so a hand-edited
    file with a typo in it can still be recovered by hand.
    """
    path = Path(path) if path is not None else settings_path()
    emit = warn if warn is not None else _warn_to_stderr

    if not path.exists():
        return factory_settings()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        emit(f"settings: {path} unreadable ({exc}); using factory defaults")
        return factory_settings()

    if not isinstance(raw, dict):
        emit(f"settings: {path} is not a JSON object; using factory defaults")
        return factory_settings()

    # UNKNOWN KEYS ARE IGNORED, NOT FAULTED, and Tranche 3 SS2 depends on it: a
    # settings.json written before the zoom bounds were removed still carries
    # zoom_min_px_per_m and zoom_max_px_per_m, and must still load. A file that
    # faults on a key it wrote itself is the kind of defect that only appears on
    # someone else's machine.
    known = {field.name for field in fields(Settings)}
    accepted = {key: value for key, value in raw.items() if key in known}
    settings, warnings = clamp_settings(replace(factory_settings(), **accepted))
    for warning in warnings:
        emit(warning)
    return settings


def save_settings(settings, path=None):
    """Write settings as JSON. Returns the path written, or None on failure."""
    path = Path(path) if path is not None else settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(settings), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        _warn_to_stderr(f"settings: could not write {path} ({exc})")
        return None
    return path


def _warn_to_stderr(message):
    print(message, file=sys.stderr)


# --------------------------------------------------------------------------
# Adjustment
# --------------------------------------------------------------------------


def adjust(settings, field, direction, fine=False):
    """Return settings with one field moved by `direction` (-1 or +1)."""
    if field.kind == "bool":
        updated = replace(settings, **{field.key: not getattr(settings, field.key)})
    elif field.kind == "ladder":
        index = TIME_SCALES.index(settings.time_scale)
        index = min(max(index + direction, 0), len(TIME_SCALES) - 1)
        updated = replace(settings, time_scale=TIME_SCALES[index])
    else:
        step = field.fine_step if fine else field.step
        updated = replace(
            settings, **{field.key: getattr(settings, field.key) + direction * step}
        )

    clamped, _ = clamp_settings(updated)
    return clamped


def format_value(settings, field):
    """Display string for one setting."""
    value = getattr(settings, field.key)
    if field.kind == "bool":
        return "on" if value else "off"
    if field.kind == "ladder":
        return f"x{value:g}"
    if abs(value) < 1.0:
        return f"{value:.3f} {field.unit}".rstrip()
    return f"{value:.2f} {field.unit}".rstrip()
