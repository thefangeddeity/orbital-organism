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
    tanzania_ax = fig.add_axes([0.80, PANEL_BOTTOM, 0.18, PANEL_HEIGHT])

    PHOSPHOR = (0.0, 0.85, 1.0)

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

    def _draw_separator(panel_ax, y, alpha=0.22):
        panel_ax.plot(
            [0.05, 0.95], [y, y],
            color=(*PHOSPHOR, alpha), linewidth=0.6,
            transform=panel_ax.transAxes,
        )

    def _style_2d_panel(panel_ax):
        panel_ax.set_facecolor(NAVY_BG)
        for spine in panel_ax.spines.values():
            spine.set_color((*PHOSPHOR, 0.12))
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
    camera_state = {"zoom_factor": 1.0}

    def _on_scroll(event):
        if event.button == "up":
            camera_state["zoom_factor"] *= 0.9
        elif event.button == "down":
            camera_state["zoom_factor"] *= 1.1

        camera_state["zoom_factor"] = max(
            0.05,
            min(20.0, camera_state["zoom_factor"]),
        )

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

            # mplot3d hardcodes '#b0b0b0' in the pane's private _axinfo
            # dict and ignores ax.grid()'s color/alpha kwargs entirely --
            # this is the actual fix, not just a dimmer number.
            try:
                axis._axinfo["grid"]["color"] = (*PHOSPHOR, grid_alpha)
                axis._axinfo["grid"]["linewidth"] = 0.3
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

    def _render_tanzania_panel():
        tanzania_ax.clear()
        _style_2d_panel(tanzania_ax)

        online = tanzania_provider.available()
        # Both states now use the same alpha the rest of the panel's
        # phosphor elements use (0.85-0.9), instead of the status dot
        # being the one fully-opaque element in an otherwise
        # alpha-blended system.
        status_color = (*PHOSPHOR, 0.9) if online else (0.8, 0.25, 0.25, 0.85)

        # Sans-serif for the header, not monospace -- monospace stays
        # reserved for tabular data below, where alignment matters.
        # A heading in a terminal font is what made this read as a
        # retro CRT rather than a designed HUD.
        tanzania_ax.text(
            0.05, 0.95, "TANZANIA",
            color=PHOSPHOR, fontsize=11, weight="bold", va="center",
        )
        # A bordered badge, not bare floating text -- otherwise it has
        # nothing grounding it against the panel edge.
        tanzania_ax.add_patch(
            Rectangle(
                (0.73, 0.935), 0.24, 0.032,
                transform=tanzania_ax.transAxes,
                edgecolor=(*status_color[:3], 0.7), facecolor=(*status_color[:3], 0.12),
                linewidth=1.0,
            )
        )
        tanzania_ax.text(
            0.95, 0.951, "ONLINE" if online else "OFFLINE",
            color=status_color, fontsize=8, weight="bold",
            va="center", ha="right",
        )
        tanzania_ax.text(
            0.05, 0.905, tanzania_provider.host_info,
            color=PHOSPHOR, fontsize=7, va="center", alpha=0.55,
        )
        _draw_separator(tanzania_ax, 0.875)

        job_count, last_task = _tanzania_dispatch_history()

        info_lines = [
            f"role: {tanzania_provider.role or 'n/a'}",
            f"addr: {tanzania_provider.address or 'unconfigured'}",
            "",
            "DISPATCH",
            f"  jobs sent: {job_count}",
        ]

        if last_task:
            info_lines.append(f"  last: {last_task}")
        else:
            info_lines.extend([
                "  (none yet -- run",
                "   tools/dispatch_tanzania.py)",
            ])

        tanzania_ax.text(
            0.06, 0.86, "\n".join(info_lines),
            color=PHOSPHOR, fontsize=7.5, family="monospace",
            va="top", alpha=0.8,
        )

        # Defrag-style block grid, but honest about what it can show:
        # dispatch_tanzania.py runs as a single blocking SSH call from
        # a separate process, with no channel to stream partial
        # progress back mid-run -- there's no "live" to visualize.
        # What IS real: every cell of the last completed sweep's
        # results, colored by how good that combination actually
        # scored. Grounded in real numbers already on disk, not a
        # faked progress animation for work that can't be observed
        # while it's happening.
        _, latest_result = _tanzania_latest_result()

        if latest_result and latest_result.get("results"):
            LOSS_ORDER = ["mse", "mae", "huber", "weighted_mse"]
            ACTIVATION_ORDER = ["relu", "gelu", "tanh"]

            scored = {
                (r["loss_variant"], r["activation_variant"]): r["validation_loss"]
                for r in latest_result["results"]
                if math.isfinite(r["validation_loss"])
            }

            if scored:
                # Log scale: validation losses here span 4 orders of
                # magnitude (0.07 to 1500+ isn't unusual once a bad
                # activation/loss pairing diverges). A linear scale
                # gets dominated by that one outlier and makes every
                # reasonable combination look identically "best" --
                # log spreads the real differences out meaningfully.
                log_scored = {
                    key: math.log(max(value, 1e-12))
                    for key, value in scored.items()
                }
                lo = min(log_scored.values())
                hi = max(log_scored.values())
                span = (hi - lo) or 1.0

                # Enlarged from the first pass (cell_h 0.032 -> 0.06):
                # the grid was floating in a mostly-empty lower panel
                # with no relationship to the resource bar below it.
                # Bigger cells give it real visual weight as the
                # panel's centerpiece instead of reading as an
                # afterthought.
                grid_top = 0.53
                grid_left = 0.06
                cell_w = 0.88 / len(LOSS_ORDER)
                cell_h = 0.06
                gap = 0.008


                tanzania_ax.text(
                    grid_left, grid_top + 0.05, "LAST SWEEP",
                    color=PHOSPHOR, fontsize=6.5, family="monospace",
                    alpha=0.55,
                )

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

                        tanzania_ax.add_patch(
                            Rectangle(
                                (x, y), cell_w - gap, cell_h,
                                color=color,
                            )
                        )

                for col, loss_name in enumerate(LOSS_ORDER):
                    tanzania_ax.text(
                        grid_left + col * cell_w + (cell_w - gap) / 2,
                        grid_top - len(ACTIVATION_ORDER) * (cell_h + gap) - 0.005,
                        loss_name[:4],
                        color=PHOSPHOR, fontsize=5.5, family="monospace",
                        alpha=0.5, ha="center", va="top", rotation=30,
                    )

        # Radial gauge instead of a flat bar, matching the reference
        # HUDs' circular meters. Wedge/Circle patches draw in DATA
        # coordinates -- on this panel's non-square axes (2.88in wide
        # x 7.92in tall) that stretches a true circle into an ellipse,
        # the exact bug already found and fixed once for the status
        # dot. Compensated here by scaling the y-radius by the axes'
        # actual physical aspect ratio so it renders as a true circle.
        gauge_cx, gauge_cy = 0.5, 0.105
        gauge_rx = 0.11
        tanzania_aspect = (9 * PANEL_HEIGHT) / (16 * 0.18)
        gauge_ry = gauge_rx / tanzania_aspect

        theta_bg = np.linspace(0, 2 * np.pi, 120)
        tanzania_ax.plot(
            gauge_cx + gauge_rx * np.cos(theta_bg),
            gauge_cy + gauge_ry * np.sin(theta_bg),
            color=(*PHOSPHOR, 0.15), linewidth=4, solid_capstyle="round",
        )

        frac = max(0.0, min(1.0, tanzania_cap_pct / 100.0))
        theta_fg = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi * frac, 100)
        tanzania_ax.plot(
            gauge_cx + gauge_rx * np.cos(theta_fg),
            gauge_cy + gauge_ry * np.sin(theta_fg),
            color=(*PHOSPHOR, 0.85), linewidth=4, solid_capstyle="round",
        )

        tanzania_ax.text(
            gauge_cx, gauge_cy, f"{tanzania_cap_pct:.0f}%",
            color=PHOSPHOR, fontsize=9, weight="bold",
            ha="center", va="center",
        )
        tanzania_ax.text(
            gauge_cx, gauge_cy - gauge_ry - 0.02, "RESOURCE CAP",
            color=PHOSPHOR, fontsize=6, ha="center", va="top", alpha=0.5,
        )

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
            _render_tanzania_panel()

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

                    ax.text(
                        x + 0.035,
                        y + 0.035,
                        z + 0.035,
                        name,
                        fontsize=8,
                        color=PHOSPHOR,
                    )

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

                ax.text(
                    x + 0.035,
                    y + 0.035,
                    z + 0.035,
                    name,
                    fontsize=8,
                    color=PHOSPHOR,
                )

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
                0.05, 0.96, "ORGANISM",
                color=PHOSPHOR, fontsize=11, weight="bold", va="center",
            )
            dashboard_ax.text(
                0.05, 0.93, f"age {_format_age(age_seconds)} · {learning_state}",
                color=PHOSPHOR, fontsize=7, va="center", alpha=0.55,
            )
            _draw_separator(dashboard_ax, 0.905)

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

            # A graphical fingerprint of the actual network, not
            # decoration -- same idea as SSH randomart: a dense,
            # unique-looking pattern deterministically generated from
            # real data (here, the live weight matrices), that visibly
            # changes as the organism actually learns. Real values
            # through a real colormap, not synthetic art.
            if learner is not None:
                _draw_separator(dashboard_ax, 0.30)
                dashboard_ax.text(
                    0.05, 0.285, "BRAIN",
                    color=PHOSPHOR, fontsize=8, weight="bold", alpha=0.7,
                )

                weight_matrices = [
                    ("W1", learner.w1),
                    ("W2", learner.w2),
                    ("W3", learner.w3),
                ]

                max_abs = max(
                    float(np.max(np.abs(w))) for _, w in weight_matrices
                ) or 1.0
                brain_norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)

                band_top = 0.255
                band_h = 0.058
                band_gap = 0.028

                for i, (label, matrix) in enumerate(weight_matrices):
                    y1 = band_top - i * (band_h + band_gap)
                    y0 = y1 - band_h

                    dashboard_ax.text(
                        0.05, y1 + 0.012,
                        f"{label} ({matrix.shape[0]}x{matrix.shape[1]})",
                        color=PHOSPHOR, fontsize=6, alpha=0.5,
                    )
                    dashboard_ax.imshow(
                        matrix.T,
                        extent=[0.05, 0.95, y0, y1],
                        cmap=BRAIN_CMAP, norm=brain_norm,
                        aspect="auto", interpolation="nearest",
                    )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            frame_cpu_ms = (time.perf_counter() - now) * 1000.0
            budget.record_frame_cost(frame_cpu_ms)

            if upgrade_report and upgrade_report.get("checked"):
                print()
                verb = "GREW" if upgrade_report.get("granted") else "held"
                print(f"[Organism] fidelity check ({verb}) -> {upgrade_report['reason']}")

                if upgrade_report.get("granted"):
                    # Earned growth is saved the instant it's earned, not
                    # deferred to graceful shutdown -- a force-killed
                    # process must not be able to un-earn it.
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







