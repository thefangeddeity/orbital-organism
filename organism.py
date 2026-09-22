from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

MODULE_DIR = Path(__file__).resolve().parent / "modules"
sys.path.insert(0, str(MODULE_DIR))

from modules.registry import ModuleRegistry
from modules.organic_budget import OrganicBudget, FIDELITY_LEVELS
from modules.compute_provider import build_providers
from modules.real_systems import build_world


ROOT = Path(__file__).resolve().parents[2]
SIMULATOR = ROOT / "orbital-sandbox.py"
MODULE_DIR = Path(__file__).resolve().parent / "modules"
sys.path.insert(0, str(MODULE_DIR))
CONFIG = Path(__file__).resolve().parent / "organism.json"


def load_simulator():

    spec = importlib.util.spec_from_file_location(
        "orbital_sandbox",
        SIMULATOR,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load {SIMULATOR}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules["orbital_sandbox"] = module

    spec.loader.exec_module(module)

    return module


def load_config():
    with CONFIG.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        return json.load(f)

def main():

    print("=" * 60)
    print("ORBITAL ORGANISM")
    print("=" * 60)

    print()
    print("Loading simulator...")

    sim = load_simulator()

    print("Running simulator self-tests...")

    if not sim.run_self_tests():
        raise RuntimeError(
            "Simulator self-tests failed."
        )

    print("Simulator verified.")

    config = load_config()

    world_mode = config.get("world", {}).get("mode", "solar_system")
    world = build_world(sim, world_mode)

    print(
        f"World: {getattr(world, 'label', 'Solar System')} "
        f"(mode={world_mode})"
    )

    registry = ModuleRegistry(
        MODULE_DIR
    )

    discovered = registry.discover()

    print()
    print("DISCOVERED LEGOS")

    for name in discovered:
        print(f"  + {name}")

    observer = registry.activate(
        "orbit_observer",
        config["modules"]["orbit_observer"]["parameters"],
    )

    observer.attach_simulator(world)

    learner = None

    if (
        config["modules"]
        .get("neural_learner", {})
        .get("enabled", False)
    ):
        learner = registry.activate(
            "neural_learner",
            config["modules"]["neural_learner"]["parameters"],
        )

        learner.attach_simulator(world)

    print()
    print("ACTIVE LEGOS")

    for name in registry.active():
        print(f"  * {name}")

    providers = build_providers(config)

    print()
    print("COMPUTE PROVIDERS")

    for provider_name, provider in providers.items():
        marker = "*" if provider.available() else "o"
        role = getattr(provider, "role", "coordinator + renderer")
        print(f"  {marker} {provider_name:<12} {role}")

    budget = OrganicBudget(
        config.get("organic_budget", {}),
        body_count=len(world.PLANETS),
    )
    budget.attach_state_path(
        Path(__file__).resolve().parent / "state" / "organic_budget.json"
    )

    print()
    print(
        f"METABOLISM  fidelity=L{budget.fidelity_level} "
        f"({FIDELITY_LEVELS[budget.fidelity_level]['name']})  "
        f"cpu_cap={budget.max_cpu_ms_per_frame:.0f}ms  "
        f"mem_cap={budget.max_memory_mb:.0f}MB"
    )

    print()
    print("Starting organism.")
    print()

    plt.ion()

    # ------------------------------------------------------------------
    # 3-D organism renderer.
    #
    # The simulation and learning system remain untouched. This renderer
    # consumes their runtime state and adds the missing spatial view.
    # ------------------------------------------------------------------

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Sci-fi HUD palette, not retro terminal: dark navy (not flat black)
    # + cyan accent (not phosphor green), matching the reference mood
    # boards -- flat black + pure green read as a 1980s CRT, which
    # wasn't the intent. Matplotlib can't do true glow/bloom or custom
    # vector iconography, so this targets what's actually achievable:
    # palette, angled corner-bracket framing, and a radial gauge,
    # not a pixel-identical recreation of a game HUD.
    NAVY_BG = (0.02, 0.045, 0.09)

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(NAVY_BG)

    # Shared layout system: every panel uses the same top/bottom bounds
    # and the same framed treatment, so the window reads as one
    # composed design instead of three independently-styled boxes.
    PANEL_BOTTOM = 0.05
    PANEL_TOP = 0.93
    PANEL_HEIGHT = PANEL_TOP - PANEL_BOTTOM

    dashboard_ax = fig.add_axes([0.015, PANEL_BOTTOM, 0.185, PANEL_HEIGHT])
    ax = fig.add_axes([0.21, PANEL_BOTTOM, 0.56, PANEL_HEIGHT], projection="3d")
    right_ax = fig.add_axes([0.80, PANEL_BOTTOM, 0.18, PANEL_HEIGHT])

    PHOSPHOR = (0.0, 0.85, 1.0)

    # Type scale: five steps, not the nine this had grown. Sizes that
    # differ by 0.5pt read as a mistake rather than a decision.
    FS_TITLE = 13
    FS_PANEL = 11
    FS_SUBHEAD = 9
    FS_BODY = 7.5
    FS_CAPTION = 6.5

    # Two text opacities, not six.
    A_PRIMARY = 0.92
    A_SECONDARY = 0.55

    # Shared vertical rhythm. Every panel header sits on the same
    # baseline, and header->content / section gaps use one value each
    # instead of 0.050-vs-0.055 and 0.018-vs-0.020 pairs.
    PANEL_TITLE_Y = 0.962
    GAP_HEAD_RULE = 0.022
    GAP_HEAD_CONTENT = 0.052
    GAP_SECTION = 0.030

    def _draw_corner_brackets(panel_ax, size=0.045, alpha=0.7, linewidth=1.4):
        # Angled L-shaped corner brackets, not a plain rectangle border
        # -- the one framing device every reference image shares.
        corners = [
            (0.0, 0.0, 1, 1),
            (1.0, 0.0, -1, 1),
            (0.0, 1.0, 1, -1),
            (1.0, 1.0, -1, -1),
        ]
        for cx, cy, dx, dy in corners:
            panel_ax.plot(
                [cx, cx + dx * size], [cy, cy],
                color=(*PHOSPHOR, alpha), linewidth=linewidth,
                transform=panel_ax.transAxes, clip_on=False, solid_capstyle="butt",
            )
            panel_ax.plot(
                [cx, cx], [cy, cy + dy * size],
                color=(*PHOSPHOR, alpha), linewidth=linewidth,
                transform=panel_ax.transAxes, clip_on=False, solid_capstyle="butt",
            )

    def _draw_separator(panel_ax, y, alpha=0.18):
        panel_ax.plot(
            [0.05, 0.95], [y, y],
            color=(*PHOSPHOR, alpha), linewidth=0.6,
            transform=panel_ax.transAxes,
        )

    def _style_2d_panel(panel_ax):
        panel_ax.set_facecolor(NAVY_BG)
        for spine in panel_ax.spines.values():
            # Above the internal separators' alpha: a divider inside a
            # frame should never out-weigh the frame containing it.
            spine.set_color((*PHOSPHOR, 0.30))
        panel_ax.set_xticks([])
        panel_ax.set_yticks([])
        panel_ax.set_xlim(0, 1)
        panel_ax.set_ylim(0, 1)
        _draw_corner_brackets(panel_ax)

    # Diverging scale anchored at a midpoint, not a direct RGB blend
    # between the two endpoint hues -- cyan and red are near-
    # complementary, so linearly blending them passes straight through
    # a muddy gray at the midpoint (confirmed by rendering and
    # inspecting the actual output). Used for both the Tanzania sweep
    # grid and the weight heatmaps below, so "warm = bad/negative,
    # cyan = good/positive" reads as one consistent visual language.
    GRID_WARN = (0.85, 0.25, 0.2)
    BRAIN_CMAP = LinearSegmentedColormap.from_list(
        "brain", [GRID_WARN, NAVY_BG, PHOSPHOR],
    )

    # Persistent visual trails belong to the renderer, not the simulator.
    render_trails = {
        body.name: []
        for body in world.PLANETS
    }

    # Manual zoom is a multiplier on the auto-fit cube extent computed
    # every frame below, not a direct axis-limit write -- otherwise the
    # per-frame auto-fit would silently undo any scroll zoom the instant
    # the next frame drew.
    view_state_path = Path(__file__).resolve().parent / "state" / "view.json"

    def _load_view_state():
        try:
            data = json.loads(view_state_path.read_text(encoding="utf-8-sig"))
            return {"zoom_factor": float(data.get("zoom_factor", 1.0))}
        except Exception:
            return {"zoom_factor": 1.0}

    def _save_view_state():
        try:
            view_state_path.parent.mkdir(parents=True, exist_ok=True)
            temp = view_state_path.with_suffix(".tmp")
            temp.write_text(
                json.dumps({"zoom_factor": camera_state["zoom_factor"]}),
                encoding="utf-8",
            )
            temp.replace(view_state_path)
        except Exception:
            pass

    camera_state = _load_view_state()

    def _on_scroll(event):
        if event.button == "up":
            camera_state["zoom_factor"] *= 0.9
        elif event.button == "down":
            camera_state["zoom_factor"] *= 1.1

        camera_state["zoom_factor"] = max(
            0.05,
            min(20.0, camera_state["zoom_factor"]),
        )
        # Saved on every scroll, not just at shutdown -- a force-killed
        # process (window closed via task manager, not the 'x' button)
        # must not silently revert the zoom to default.
        _save_view_state()

    fig.canvas.mpl_connect("scroll_event", _on_scroll)

    # Grid defaults OFF -- press 'g' to toggle. Even when on, mplot3d's
    # gridlines don't respect ax.grid(color=..., alpha=...) the way 2-D
    # axes do (they hardcode '#b0b0b0' internally); the real fix lives
    # in _apply_dark_theme() below, which writes the pane's private
    # _axinfo grid style directly.
    grid_state = {"visible": False}

    # The organism's two clocks (wall-clock age vs. simulated universe
    # time) are deliberately different rates -- default is fast-forward
    # (2 sim-days/real-second) so orbits are watchable at all. 't' snaps
    # back to true real time (1 sim-second per real-second) and back
    # again, so the relationship never gets lost even though it's not
    # running at it by default.
    REAL_TIME_DAYS_PER_SECOND = 1.0 / 86400.0
    configured_days_per_second = float(
        config["modules"]["orbit_observer"]["parameters"]
        ["simulation_days_per_second"]
    )
    time_state = {
        "days_per_second": configured_days_per_second,
        "real_time": False,
    }

    def _on_key(event):
        if event.key == "g":
            grid_state["visible"] = not grid_state["visible"]
        elif event.key == "t":
            time_state["real_time"] = not time_state["real_time"]
            time_state["days_per_second"] = (
                REAL_TIME_DAYS_PER_SECOND
                if time_state["real_time"]
                else configured_days_per_second
            )

    fig.canvas.mpl_connect("key_press_event", _on_key)

    def _sphere_mesh(cx, cy, cz, radius, resolution=10):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        xs = cx + radius * np.outer(np.cos(u), np.sin(v))
        ys = cy + radius * np.outer(np.sin(u), np.sin(v))
        zs = cz + radius * np.outer(np.ones_like(u), np.cos(v))
        return xs, ys, zs

    # Real-color approximations, standard astronomy-visualization values.
    # A fixed prior for now; a natural later hook is letting the organism
    # refine these once it has real spectral/photometric data to learn
    # from, the same way it refines loss functions today.
    BODY_COLORS = {
        "Sun": "#FDB813",
        "Mercury": "#9C9C9C",
        "Venus": "#DDBD8B",
        "Earth": "#4F82C4",
        "Mars": "#B7410E",
        "Jupiter": "#D8A76A",
        "Saturn": "#E3C16F",
        "Uranus": "#9FE3F0",
        "Neptune": "#3D5FE0",
    }

    def _apply_dark_theme(ax):
        # ax.clear() resets these every frame, so this must be re-applied
        # every frame too -- not just once at setup.
        ax.set_facecolor(NAVY_BG)

        visible = grid_state["visible"]
        grid_alpha = 0.10 if visible else 0.0
        edge_alpha = 0.14 if visible else 0.0

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            # mplot3d's pane is a Collection with its own artist-level
            # alpha (0.5 by default) that silently overrides whatever
            # alpha is embedded in set_facecolor/set_edgecolor's RGBA
            # tuples -- confirmed by direct inspection. set_alpha(None)
            # un-overrides it so the per-channel alphas below actually
            # take effect; without this the box edges render at a fixed
            # 0.5 alpha no matter what color is requested.
            axis.pane.set_alpha(None)
            axis.pane.set_facecolor((*NAVY_BG, 1.0))
            axis.pane.set_edgecolor((*PHOSPHOR, edge_alpha))

            # The axis spines default to literal 'black' in _axinfo,
            # which on a navy background reads as visible dark lines
            # rather than as nothing -- that's the stray line crossing
            # the lower scene. Folded into the same 'g' toggle as the
            # rest of the reference frame instead of left hardcoded.
            axis.line.set_color((*PHOSPHOR, edge_alpha))

            # mplot3d hardcodes '#b0b0b0' in the pane's private _axinfo
            # dict and ignores ax.grid()'s color/alpha kwargs entirely --
            # this is the actual fix, not just a dimmer number.
            try:
                axis._axinfo["grid"]["color"] = (*PHOSPHOR, grid_alpha)
                axis._axinfo["grid"]["linewidth"] = 0.3
                axis._axinfo["axisline"]["color"] = (*PHOSPHOR, edge_alpha)
            except (AttributeError, KeyError):
                pass

        ax.grid(visible)

        # Ticks/labels/axis-titles all fold into the same toggle as the
        # grid and box edges now -- off by default, 'g' brings the whole
        # scale reference frame back at once, dimmed below full-bright.
        PHOSPHOR_DIM = (*PHOSPHOR, 0.45 if visible else 0.0)

        ax.tick_params(colors=PHOSPHOR_DIM)

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_color(PHOSPHOR_DIM)

    def _visual_radius_au(radius_km):
        # True-to-scale, deliberately. Bodies are astronomically tiny
        # next to their orbits (Earth's radius is ~4e-5 AU) -- most
        # planets will be sub-pixel or invisible at solar-system-wide
        # zoom. That's correct, not a bug: scroll-zoom in close and the
        # real relative sizes (Jupiter dwarfing Mercury, etc.) show up
        # honestly instead of a make-believe uniform scale.
        return radius_km / world.AU_KM

    tanzania_provider = providers["tanzania"]
    tanzania_cap_pct = float(
        config.get("providers", {})
        .get("tanzania", {})
        .get("resource_cap_pct", 100)
    )
    tanzania_results_dir = (
        Path(__file__).resolve().parent / "state" / "tanzania_results"
    )

    def _tanzania_result_files():
        # tools/dispatch_tanzania.py runs as a completely separate
        # process, out-of-band from this render loop -- the live
        # organism has no in-memory way to know a dispatch happened.
        # Its only record is the result files it writes to disk, so
        # that's what everything here reads. Real history, not a
        # running total this process tracked itself.
        if not tanzania_results_dir.exists():
            return []

        return sorted(
            tanzania_results_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
        )

    def _tanzania_dispatch_history():
        result_files = _tanzania_result_files()

        if not result_files:
            return 0, None

        latest = result_files[-1]
        task_name = latest.stem.rsplit("-", 2)[0]

        return len(result_files), task_name

    def _tanzania_latest_result():
        result_files = _tanzania_result_files()

        if not result_files:
            return None, None

        latest = result_files[-1]

        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            return latest.name, None

        return latest.name, data

    # The panel's physical aspect ratio, needed anywhere a shape has to
    # stay visually circular: patches draw in DATA coordinates, and
    # this panel is 2.88in wide x 7.92in tall, so equal data-space
    # radii render as a stretched ellipse without this correction.
    RIGHT_PANEL_ASPECT = (9 * PANEL_HEIGHT) / (16 * 0.18)

    def _caption(panel_ax, x, y, text, ha="left"):
        """
        One treatment for every caption sitting above a graphic --
        'LAST SWEEP' and 'W1 4x24' are the same thing structurally and
        were styled differently (bold/7pt/a0.7 vs regular/6pt/a0.5).
        """
        panel_ax.text(
            x, y, text,
            color=PHOSPHOR, fontsize=FS_CAPTION, weight="bold",
            va="center", ha=ha, alpha=0.6,
        )

    def _render_brain_section(y_top):
        """The organism itself: its live weight matrices."""
        right_ax.text(
            0.05, y_top, "BRAIN",
            color=PHOSPHOR, fontsize=FS_PANEL, weight="bold", va="center",
        )
        _draw_separator(right_ax, y_top - GAP_HEAD_RULE)

        if learner is None:
            right_ax.text(
                0.05, y_top - GAP_HEAD_CONTENT, "engine disabled",
                color=PHOSPHOR, fontsize=FS_BODY,
                va="center", alpha=A_SECONDARY,
            )
            return y_top - 0.10

        # A graphical fingerprint of the actual network, not
        # decoration -- same idea as SSH randomart: a dense, unique
        # pattern deterministically generated from real data (here the
        # live weight matrices), which visibly changes as the organism
        # actually learns. Real values through a real colormap.
        weight_matrices = [
            ("W1", learner.w1),
            ("W2", learner.w2),
            ("W3", learner.w3),
        ]

        max_abs = max(
            float(np.max(np.abs(w))) for _, w in weight_matrices
        ) or 1.0
        brain_norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)

        # W2 is the 24x24 bulk of the network; giving it proportionally
        # more height than the thin 4x24 and 24x2 edge layers reflects
        # what's actually there instead of three identical bands.
        # Sized down from the first pass, which crowded the EXTENSIONS
        # section below it -- these read fine small, and the panel
        # needs the breathing room more than the pixels.
        band_heights = [0.038, 0.080, 0.038]
        label_gap = 0.020
        band_gap = label_gap

        y = y_top - GAP_HEAD_CONTENT

        for (label, matrix), band_h in zip(weight_matrices, band_heights):
            _caption(
                right_ax, 0.05, y,
                f"{label}  {matrix.shape[0]}×{matrix.shape[1]}",
            )
            y -= label_gap
            right_ax.imshow(
                matrix.T,
                extent=[0.05, 0.95, y - band_h, y],
                cmap=BRAIN_CMAP, norm=brain_norm,
                aspect="auto", interpolation="nearest",
            )
            y -= band_h + band_gap

        return y

    def _render_extension(provider, y_top, cap_pct, show_dispatch=False):
        """
        One fleet extension row. Every configured machine gets one --
        Tina and Ariana were real, configured providers that never
        appeared anywhere in this panel despite the section being
        called EXTENSIONS.
        """
        online = provider.available()
        # One hue, not a red/cyan switch -- online is full phosphor
        # brightness, offline is the same phosphor dimmed.
        status_color = (*PHOSPHOR, 0.9) if online else (*PHOSPHOR, 0.4)

        right_ax.text(
            0.05, y_top, provider.name.upper(),
            color=PHOSPHOR, fontsize=FS_SUBHEAD, weight="bold", va="center",
        )

        # Badge text is centered in its box on both axes. va="center"
        # centers the full font bounding box, which sits slightly low
        # for all-caps text with no descenders, so the label carries a
        # small upward optical correction.
        badge_left, badge_right = 0.60, 0.92
        badge_bottom, badge_h = y_top - 0.017, 0.034
        right_ax.add_patch(
            Rectangle(
                (badge_left, badge_bottom), badge_right - badge_left, badge_h,
                edgecolor=(*status_color[:3], 0.7),
                facecolor=(*status_color[:3], 0.1),
                linewidth=1.0,
            )
        )
        right_ax.text(
            (badge_left + badge_right) / 2.0,
            badge_bottom + badge_h / 2.0 + 0.0015,
            "ONLINE" if online else "OFFLINE",
            color=status_color, fontsize=8, weight="bold",
            ha="center", va="center",
        )

        # Resource cap is a static config constant, so it belongs on
        # one line here rather than as the largest graphic in the
        # panel -- a radial gauge animating nothing was the least
        # earned pixel-spend on screen.
        y = y_top - 0.032
        right_ax.text(
            0.05, y,
            f"{provider.host_info} · cap {cap_pct:.0f}%",
            color=PHOSPHOR, fontsize=FS_CAPTION, va="center", alpha=A_SECONDARY,
        )

        # Prose, not a numeric column -- sans, per the rule that
        # monospace is reserved for values that need to align.
        y -= 0.026
        right_ax.text(
            0.05, y, provider.role or "n/a",
            color=PHOSPHOR, fontsize=FS_BODY, va="top", alpha=A_SECONDARY,
        )
        y -= 0.024

        if show_dispatch:
            job_count, last_task = _tanzania_dispatch_history()
            for label, value in (
                ("jobs sent", str(job_count)),
                ("last", last_task or "none yet"),
            ):
                right_ax.text(
                    0.05, y, label,
                    color=PHOSPHOR, fontsize=FS_BODY,
                    va="top", alpha=A_SECONDARY,
                )
                right_ax.text(
                    0.42, y, value,
                    color=PHOSPHOR, fontsize=FS_BODY, family="monospace",
                    va="top", alpha=A_PRIMARY,
                )
                y -= 0.024

        return y

        return y - 0.10

    def _render_sweep_grid(y_top):
        # Honest about what it can show: dispatch_tanzania.py runs as a
        # single blocking SSH call from a separate process, with no
        # channel to stream partial progress back mid-run -- there is
        # no "live" to visualize. What IS real: every cell of the last
        # completed sweep, colored by how it actually scored.
        _, latest_result = _tanzania_latest_result()

        if not (latest_result and latest_result.get("results")):
            return y_top

        LOSS_ORDER = ["mse", "mae", "huber", "weighted_mse"]
        ACTIVATION_ORDER = ["relu", "gelu", "tanh"]

        scored = {
            (r["loss_variant"], r["activation_variant"]): r["validation_loss"]
            for r in latest_result["results"]
            if math.isfinite(r["validation_loss"])
        }

        if not scored:
            return y_top

        # Log scale: these losses span 4 orders of magnitude (0.07 to
        # 1500+ once a bad pairing diverges). A linear scale gets
        # dominated by the outlier and makes every reasonable
        # combination look identically "best".
        log_scored = {
            key: math.log(max(value, 1e-12))
            for key, value in scored.items()
        }
        lo = min(log_scored.values())
        hi = max(log_scored.values())
        span = (hi - lo) or 1.0

        _caption(right_ax, 0.05, y_top, "LAST SWEEP")

        # Clearance below the label's own text height -- at 0.030 the
        # first row was drawn straight through "LAST SWEEP".
        grid_top = y_top - 0.042
        grid_left = 0.05
        cell_w = 0.90 / len(LOSS_ORDER)
        cell_h = 0.036
        gap = 0.006

        for col, loss_name in enumerate(LOSS_ORDER):
            for row, activation_name in enumerate(ACTIVATION_ORDER):
                key = (loss_name, activation_name)
                x = grid_left + col * cell_w
                y = grid_top - row * (cell_h + gap)

                if key in log_scored:
                    # Rank within [0, 1]: 0 = worst, 1 = best.
                    rank = 1.0 - (log_scored[key] - lo) / span

                    if rank >= 0.5:
                        t = (rank - 0.5) * 2.0
                        color = (*PHOSPHOR, 0.35 + 0.5 * t)
                    else:
                        t = rank * 2.0
                        color = (*GRID_WARN, 0.85 - 0.45 * t)
                else:
                    color = (0.3, 0.3, 0.3, 0.25)

                right_ax.add_patch(
                    Rectangle((x, y), cell_w - gap, cell_h, color=color)
                )

        # Horizontal, not rotated (rotated 5.5pt was unreadable), and
        # abbreviated deliberately rather than truncated mid-word --
        # loss_name[:4] produced "hube" and "weig", which read as typos.
        SHORT_NAME = {
            "mse": "mse",
            "mae": "mae",
            "huber": "huber",
            "weighted_mse": "w-mse",
        }
        label_y = grid_top - len(ACTIVATION_ORDER) * (cell_h + gap) - 0.006
        for col, loss_name in enumerate(LOSS_ORDER):
            right_ax.text(
                grid_left + col * cell_w + (cell_w - gap) / 2,
                label_y, SHORT_NAME.get(loss_name, loss_name),
                color=PHOSPHOR, fontsize=FS_CAPTION,
                alpha=0.6, ha="center", va="top",
            )

        # The rows had no legend at all before -- the grid was
        # unreadable without knowing what they meant.
        right_ax.text(
            grid_left, label_y - 0.026,
            "rows: " + " / ".join(ACTIVATION_ORDER),
            color=PHOSPHOR, fontsize=FS_CAPTION,
            alpha=A_SECONDARY, va="top",
        )

        return label_y - 0.050

    def _render_right_panel():
        right_ax.clear()
        _style_2d_panel(right_ax)

        brain_bottom = _render_brain_section(PANEL_TITLE_Y)

        # Flow from where the brain section actually ended rather than
        # a guessed constant -- a hardcoded anchor put this header on
        # top of the W3 heatmap. Band heights are fixed, so this is
        # still deterministic frame to frame, just correct.
        extensions_top = brain_bottom - GAP_SECTION
        right_ax.text(
            0.05, extensions_top, "EXTENSIONS",
            color=PHOSPHOR, fontsize=FS_PANEL, weight="bold", va="center",
        )
        _draw_separator(right_ax, extensions_top - GAP_HEAD_RULE)

        provider_caps = config.get("providers", {})

        y = _render_extension(
            tanzania_provider, extensions_top - GAP_HEAD_CONTENT,
            tanzania_cap_pct, show_dispatch=True,
        )
        y = _render_sweep_grid(y - 0.012)

        for name in ("tina", "ariana"):
            provider = providers.get(name)
            if provider is None:
                continue
            cap = float(provider_caps.get(name, {}).get("resource_cap_pct", 100))
            y = _render_extension(provider, y - GAP_SECTION, cap)

        # imshow autoscales the axes to the image extent; re-assert the
        # panel's own coordinate frame so every later placement stays
        # in the 0-1 space the rest of this function assumes.
        right_ax.set_xlim(0, 1)
        right_ax.set_ylim(0, 1)

    last = time.perf_counter()
    organism_born = time.perf_counter()
    last_status_print = 0.0

    def _format_age(seconds):
        seconds = int(seconds)
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        if h:
            return f"{h}h {m:02d}m {s:02d}s"
        if m:
            return f"{m}m {s:02d}s"
        return f"{s}s"

    try:

        tick_count = 0
        while plt.fignum_exists(fig.number):

            now = time.perf_counter()

            dt = min(
                now - last,
                0.25,
            )

            last = now

            observer.advance(
                dt
                * time_state["days_per_second"]
                * 86400.0
            )

            # ----------------------------------------------------------
            # ORGANISM CORE:
            # observer -> world state -> learner
            #
            # This is deliberately unchanged from the existing organism.
            # ----------------------------------------------------------

            result = observer.observe(None)

            learning_result = None
            upgrade_report = None

            if learner is not None:
                evolution_runs_before = learner.evolution_runs

                learning_result = learner.observe(result)

                if learner.evolution_runs > evolution_runs_before:
                    upgrade_report = budget.consider_upgrade(
                        learner.evolution_runs,
                        learner.best_loss,
                    )

            bodies = result.observations["bodies"]

            # ----------------------------------------------------------
            # Obtain the current 3-D position.
            #
            # Orbit.state_at_time() supplies the authoritative current
            # x/y state. sample_path_3d() supplies the corresponding
            # inclined orbital geometry. The nearest XY point gives the
            # Z coordinate belonging to the current orbital location.
            # ----------------------------------------------------------

            positions_3d = {}

            for body in world.PLANETS:


                state = bodies.get(body.name)

                if state is None:
                    continue

                orbit = world.planet_orbit(body)

                path3 = orbit.sample_path_3d(
                    n_points=720
                )

                target_x = state["x_au"]
                target_y = state["y_au"]

                best = min(
                    path3,
                    key=lambda point:
                        (
                            point[0] / world.AU_KM - target_x
                        ) ** 2
                        +
                        (
                            point[1] / world.AU_KM - target_y
                        ) ** 2
                )

                x = best[0] / world.AU_KM
                y = best[1] / world.AU_KM
                z = best[2] / world.AU_KM

                positions_3d[body.name] = (x, y, z)

                trail = render_trails[body.name]

                trail.append((x, y, z))

                limit = int(
                    config["modules"]
                    ["orbit_observer"]
                    ["parameters"]
                    ["trail_length"]
                )

                if len(trail) > limit:
                    del trail[:-limit]

            # ----------------------------------------------------------
            # REDRAW FRAME
            # ----------------------------------------------------------

            ax.clear()
            _apply_dark_theme(ax)
            _render_right_panel()

            # ----------------------------------------------------------
            # Orbital planes / paths.
            # ----------------------------------------------------------

            for body in world.PLANETS:


                orbit = world.planet_orbit(body)

                points = orbit.sample_path_3d(
                    n_points=360
                )

                xs = [
                    point[0] / world.AU_KM
                    for point in points
                ]

                ys = [
                    point[1] / world.AU_KM
                    for point in points
                ]

                zs = [
                    point[2] / world.AU_KM
                    for point in points
                ]

                ax.plot(
                    xs,
                    ys,
                    zs,
                    linewidth=0.7,
                    alpha=0.28,
                )

            # ----------------------------------------------------------
            # Trajectory history.
            # ----------------------------------------------------------

            for name, trail in render_trails.items():

                if len(trail) < 2:
                    continue

                xs = [point[0] for point in trail]
                ys = [point[1] for point in trail]
                zs = [point[2] for point in trail]

                ax.plot(
                    xs,
                    ys,
                    zs,
                    linewidth=1.2,
                    alpha=0.55,
                )

            # ----------------------------------------------------------
            # Sun and planetary bodies.
            #
            # L0 renders bare dots (cheapest, current default). L1+
            # renders true-radius spheres, mass-scaled -- earned via
            # OrganicBudget, not assumed. This is the "inward" growth:
            # the organism's own rendering complexity grows only when
            # learning progress and machine budget both justify it.
            # ----------------------------------------------------------

            if budget.fidelity_level >= 1:

                sun_radius_au = _visual_radius_au(world.SUN.radius)

                sxs, sys_, szs = _sphere_mesh(
                    0.0, 0.0, 0.0, sun_radius_au,
                )

                ax.plot_surface(
                    sxs, sys_, szs,
                    color=BODY_COLORS["Sun"],
                    linewidth=0,
                    antialiased=False,
                )

                body_by_name = {
                    body.name: body
                    for body in world.PLANETS
                }

                for name, position in positions_3d.items():

                    x, y, z = position
                    body = body_by_name.get(name)

                    radius_au = _visual_radius_au(
                        body.radius if body is not None else 6371.0
                    )

                    pxs, pys, pzs = _sphere_mesh(
                        x, y, z, radius_au,
                    )

                    ax.plot_surface(
                        pxs, pys, pzs,
                        color=BODY_COLORS.get(name, "#AAAAAA"),
                        linewidth=0,
                        antialiased=False,
                    )

                    label = ax.text(
                        x + 0.035,
                        y + 0.035,
                        z + 0.035,
                        name,
                        fontsize=8,
                        color=PHOSPHOR,
                    )
                    # Axes3D text isn't clipped to the axes' screen
                    # bounds by default -- a body near the edge of the
                    # cube can project to a screen position outside the
                    # 3-D panel entirely, drawing over the dashboard.
                    # The clip box has to be set explicitly per artist.
                    label.set_clip_on(True)
                    label.set_clip_box(ax.bbox)

                # Fidelity-rendered bodies handle their own labels above;
                # the L0 scatter path below is skipped this frame.
                positions_3d_for_scatter = {}

            else:

                ax.scatter(
                    [0],
                    [0],
                    [0],
                    s=130,
                    marker="o",
                    color=BODY_COLORS["Sun"],
                )

                positions_3d_for_scatter = positions_3d

            # ----------------------------------------------------------
            # Current planetary bodies (L0 scatter path).
            # ----------------------------------------------------------

            for name, position in positions_3d_for_scatter.items():

                x, y, z = position

                ax.scatter(
                    [x],
                    [y],
                    [z],
                    s=42,
                    marker="o",
                    color=BODY_COLORS.get(name, "#AAAAAA"),
                )

                label = ax.text(
                    x + 0.035,
                    y + 0.035,
                    z + 0.035,
                    name,
                    fontsize=8,
                    color=PHOSPHOR,
                )
                label.set_clip_on(True)
                label.set_clip_box(ax.bbox)

            # ----------------------------------------------------------
            # Camera / spatial scale.
            #
            # The world is rendered as a true cubic coordinate volume.
            # One AU has the same visual scale on X, Y and Z.
            #
            # The largest actual orbital extent determines the cube.
            # This keeps every discovered planet inside the same physical
            # coordinate scale without exaggerating inclination.
            # ----------------------------------------------------------

            # Current rendered frame extent, derived from the
            # authoritative 3-D planetary positions.
            frame_points = list(positions_3d.values())

            frame_x = [
                point[0]
                for point in frame_points
            ]
            frame_y = [
                point[1]
                for point in frame_points
            ]
            frame_z = [
                point[2]
                for point in frame_points
            ]

            max_extent = max(
                max(abs(value) for value in frame_x),
                max(abs(value) for value in frame_y),
                max(abs(value) for value in frame_z),
                1.0,
            )

            camera_pad = max(
                max_extent * 0.06,
                0.5,
            )

            cube_extent = (
                (max_extent + camera_pad)
                * camera_state["zoom_factor"]
            )

            ax.set_xlim(
                -cube_extent,
                cube_extent,
            )

            ax.set_ylim(
                -cube_extent,
                cube_extent,
            )

            ax.set_zlim(
                -cube_extent,
                cube_extent,
            )

            ax.set_box_aspect(
                (
                    1.0,
                    1.0,
                    1.0,
                )
            )

            ax.set_xlabel("X (AU)")
            ax.set_ylabel("Y (AU)")
            ax.set_zlabel("Z (AU)")

            ax.set_title(
                "ORBITAL ORGANISM — CONTINUOUS 3-D WORLD",
                color=PHOSPHOR, fontsize=13, weight="bold",
            )

            # ----------------------------------------------------------
            # Truthful organism state.
            # ----------------------------------------------------------

            if learner is None:
                learning_state = "OFF"
            elif learner.training_steps > 0:
                learning_state = "LEARNING"
            else:
                learning_state = "READY"

            age_seconds = now - organism_born

            dashboard_ax.clear()
            _style_2d_panel(dashboard_ax)

            dashboard_ax.text(
                0.05, PANEL_TITLE_Y, "ORGANISM",
                color=PHOSPHOR, fontsize=FS_PANEL, weight="bold", va="center",
            )
            dashboard_ax.text(
                0.05, PANEL_TITLE_Y - 0.030,
                f"age {_format_age(age_seconds)} · {learning_state}",
                color=PHOSPHOR, fontsize=FS_CAPTION,
                va="center", alpha=A_SECONDARY,
            )
            _draw_separator(dashboard_ax, PANEL_TITLE_Y - GAP_HEAD_RULE - 0.032)

            # The real signal -- is it learning, is it improving --
            # comes first. Static facts that never change for the life
            # of the process (module list, architecture tagline) are
            # collapsed to one line each and pushed to the bottom
            # instead of delaying the numbers that actually move.
            lines = ["LEARNING"]

            if learner is not None:
                lines.extend([
                    (
                        f"  validation: {learner.validation_loss:.3e}"
                    ),
                    (
                        f"  best: {learner.best_loss:.3e}"
                    ),
                    (
                        f"  successes: {learner.evolution_runs}  "
                        f"rate: {learner.evolution_success_rate:.1%}"
                    ),
                    f"  steps: {learner.training_steps:,}",
                    f"  loss_fn: {learner.active_loss_variant}",
                    f"  features: {learner.active_feature_variant}",
                    f"  activation: {learner.active_activation_variant}",
                    f"  width: {learner.hidden_width}",
                ])
            else:
                lines.append("  engine: disabled")

            budget_state = budget.state()

            lines.extend([
                "",
                "METABOLISM",
                (
                    f"  fidelity: L{budget_state['fidelity_level']} "
                    f"({budget_state['fidelity_name']})"
                ),
                (
                    f"  cpu: {budget_state['cpu_ms']:.1f}/"
                    f"{budget_state['cpu_cap_ms']:.0f}ms"
                ),
                (
                    f"  mem: {budget_state['memory_mb']:.1f}/"
                    f"{budget_state['memory_cap_mb']:.0f}MB"
                ),
                (
                    f"  grown: {budget_state['upgrades_granted']}x  "
                    f"denied: {budget_state['upgrades_denied']}x"
                ),
            ])

            if budget.last_report.get("reason"):
                lines.append(
                    f"  last check: {budget.last_report['reason'][:38]}"
                )

            lines.extend([
                "",
                f"WORLD: {getattr(world, 'label', 'Solar System')}",
                f"  bodies: {len(bodies)}",
                (
                    f"  t = "
                    f"{observer.time_seconds / 86400.0:,.2f}"
                    f" days"
                ),
                (
                    "  speed: REAL-TIME (1x)"
                    if time_state["real_time"] else
                    (
                        f"  speed: {time_state['days_per_second']:.2f} "
                        f"sim-days/s "
                        f"(~{time_state['days_per_second'] * 86400.0:,.0f}x "
                        f"real time)"
                    )
                ),
            ])

            # World-agnostic: prefer Earth when it exists (solar_system
            # mode) since it's the most relatable reference body, but
            # fall back to whatever this world's first real body is
            # instead of silently vanishing in proxima/alpha_centauri
            # mode -- this section used to key off the literal string
            # "Earth" and just disappeared entirely outside the solar
            # system.
            primary_name = "Earth" if "Earth" in bodies else next(iter(bodies), None)
            primary = bodies.get(primary_name) if primary_name else None

            if primary:
                lines.extend([
                    "",
                    primary_name.upper(),
                    f"  r = {primary['radius_au']:.6f} AU",
                    f"  v = {primary['speed_km_s']:.6f} km/s",
                    f"  E = {primary['energy']:.9f}",
                ])

            lines.extend([
                "",
                "COMPUTATION",
                "  local-cpu: coordinator + renderer",
                "  fleet: see right panel",
                "",
                "modules: " + ", ".join(registry.active()),
            ])

            dashboard_ax.text(
                0.05, 0.88, "\n".join(lines),
                color=PHOSPHOR, fontsize=7.5, family="monospace",
                va="top", alpha=0.8,
            )

            # BRAIN moved to the right panel, which had the room for
            # it. This panel is now purely the numeric readout.

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            frame_cpu_ms = (time.perf_counter() - now) * 1000.0
            budget.record_frame_cost(frame_cpu_ms)

            if upgrade_report and upgrade_report.get("checked"):
                print()
                verb = "GREW" if upgrade_report.get("granted") else "held"
                print(f"[Organism] fidelity check ({verb}) -> {upgrade_report['reason']}")

                # Save on EVERY completed check, not just granted ones.
                # Growth needs two checks to happen: the first only
                # establishes a loss baseline, the second compares
                # against it. Saving only on grants meant that baseline
                # lived in memory and died with any non-graceful exit,
                # so a frequently-restarted organism could never reach
                # the second check and would sit at "grown: 0x" forever
                # -- which is exactly what state/organic_budget.json's
                # "loss_at_last_check": null was recording.
                budget.save_state()

            tick_count += 1

            # Wall-clock cadence, not tick count -- ticks are a render-
            # loop implementation detail, not a unit anyone watching
            # this organism actually feels. Age is.
            if age_seconds - last_status_print >= 2.0:
                last_status_print = age_seconds
                status = f"[Organism] age={_format_age(age_seconds)}"
                if learner is not None:
                    status += (
                        f"  loss={learner.validation_loss:.3e}"
                        f"  steps={learner.training_steps:,}"
                        f"  successes={learner.evolution_runs}"
                        f"  L{budget.fidelity_level}"
                    )
                print(f"\r{status}...", end="", flush=True)

            time.sleep(0.03)

    except KeyboardInterrupt:

        print()
        print("Organism stopped.")

    finally:

        for lego in registry.active().values():
            lego.shutdown()

        budget.save_state()

        plt.ioff()
        plt.close(fig)


if __name__ == "__main__":
    main()







