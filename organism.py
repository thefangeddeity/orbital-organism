from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

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

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("black")

    # Three regions, explicitly positioned in figure-fraction coordinates
    # rather than via subplots_adjust: left ~0-0.19 is the text dashboard
    # (fig.text, no axes needed), middle is the 3-D world, right is the
    # per-worker fleet panel (Tanzania today; Tina/Ariana can each get
    # their own strip the same way once they're real).
    ax = fig.add_axes([0.21, 0.05, 0.56, 0.88], projection="3d")
    tanzania_ax = fig.add_axes([0.80, 0.05, 0.18, 0.88])

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

    PHOSPHOR = (0.0, 1.0, 0.45)

    def _apply_dark_theme(ax):
        # ax.clear() resets these every frame, so this must be re-applied
        # every frame too -- not just once at setup.
        ax.set_facecolor("black")

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
            axis.pane.set_facecolor((0.0, 0.0, 0.0, 1.0))
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

    def _render_tanzania_panel():
        # Honest stub: Tanzania has no real transport yet, so this shows
        # its true current state (offline, nothing dispatched) rather
        # than fabricated activity. The moment dispatch() actually sends
        # work there, this panel is where that shows up live.
        tanzania_ax.clear()
        tanzania_ax.set_facecolor("black")

        for spine in tanzania_ax.spines.values():
            spine.set_color((*PHOSPHOR, 0.2))

        tanzania_ax.set_xticks([])
        tanzania_ax.set_yticks([])
        tanzania_ax.set_xlim(0, 1)
        tanzania_ax.set_ylim(0, 1)

        online = tanzania_provider.available()
        status_color = (0.0, 1.0, 0.3) if online else (0.55, 0.15, 0.15)

        tanzania_ax.add_patch(
            Circle((0.10, 0.95), 0.028, color=status_color, zorder=3)
        )
        tanzania_ax.text(
            0.20, 0.95, "TANZANIA",
            color=PHOSPHOR, fontsize=10, family="monospace",
            weight="bold", va="center",
        )

        info_lines = [
            f"status: {'ONLINE' if online else 'OFFLINE'}",
            f"role: {tanzania_provider.role or 'n/a'}",
            f"addr: {tanzania_provider.address or 'unconfigured'}",
            "",
            "DISPATCH",
            "  jobs sent: 0",
            "  (running local-cpu",
            "   until wired)",
        ]

        tanzania_ax.text(
            0.06, 0.86, "\n".join(info_lines),
            color=PHOSPHOR, fontsize=7.5, family="monospace",
            va="top", alpha=0.8,
        )

        bar_y = 0.10
        tanzania_ax.add_patch(
            Rectangle(
                (0.06, bar_y), 0.88, 0.035,
                edgecolor=(*PHOSPHOR, 0.3), facecolor="none",
            )
        )
        tanzania_ax.add_patch(
            Rectangle(
                (0.06, bar_y), 0.88 * (tanzania_cap_pct / 100.0), 0.035,
                color=(*PHOSPHOR, 0.3),
            )
        )
        tanzania_ax.text(
            0.06, bar_y - 0.03,
            f"resource cap: {tanzania_cap_pct:.0f}%",
            color=PHOSPHOR, fontsize=7, family="monospace",
            va="top", alpha=0.65,
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
                "ORBITAL ORGANISM - CONTINUOUS 3-D WORLD",
                color=PHOSPHOR,
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

            lines = [
                "ORGANISM",
                "",
                f"age: {_format_age(age_seconds)}",
                "architecture: MODULAR",
                f"learning: {learning_state}",
                "",
                "ACTIVE LEGOS",
            ]

            for name in registry.active():
                lines.append(
                    f"  {name}"
                )

            lines.extend([
                "",
                "COMPUTATION",
            ])

            for provider_name, provider in providers.items():
                status = "available" if provider.available() else "disabled"
                lines.append(f"  {provider_name}: {status}")

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
                "LEARNING",
            ])

            if learner is not None:

                lines.extend([
                    "  engine: CPU neural network",
                    (
                        f"  steps: "
                        f"{learner.training_steps:,}"
                    ),
                    (
                        f"  validation: "
                        f"{learner.validation_loss:.3e}"
                    ),
                    (
                        f"  best: "
                        f"{learner.best_loss:.3e}"
                    ),
                    (
                        f"  width: "
                        f"{learner.hidden_width}"
                    ),
                    (
                        f"  loss_fn: "
                        f"{learner.active_loss_variant}"
                    ),
                    (
                        f"  features: "
                        f"{learner.active_feature_variant}"
                    ),
                    (
                        f"  activation: "
                        f"{learner.active_activation_variant}"
                    ),
                    (
                        f"  gen: {learner.evolution_runs}  "
                        f"success: {learner.evolution_success_rate:.1%}"
                    ),
                ])

            else:

                lines.append(
                    "  engine: disabled"
                )

            lines.extend([
                "",
                "WORLD",
                f"  bodies: {len(bodies)}",
                "",
                "SIMULATION",
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

            earth = bodies.get("Earth")

            if earth:

                lines.extend([
                    "",
                    "EARTH",
                    (
                        f"  r = "
                        f"{earth['radius_au']:.6f} AU"
                    ),
                    (
                        f"  v = "
                        f"{earth['speed_km_s']:.6f} km/s"
                    ),
                    (
                        f"  E = "
                        f"{earth['energy']:.9f}"
                    ),
                ])

            lines.extend([
                "",
                "LEGO PRINCIPLE",
                "  capabilities are modules.",
                "  parameters are adjustable.",
                "  providers are replaceable.",
                "  learning persists across runs.",
            ])

            fig.text(
                0.015, 0.94,
                "\n".join(lines),
                verticalalignment="top",
                family="monospace",
                fontsize=8.5,
                bbox={
                    "boxstyle": "round",
                    "alpha": 0.88,
                },
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            frame_cpu_ms = (time.perf_counter() - now) * 1000.0
            budget.record_frame_cost(frame_cpu_ms)

            if upgrade_report and upgrade_report.get("checked"):
                print()
                verb = "GREW" if upgrade_report.get("granted") else "held"
                print(f"[Organism] fidelity check ({verb}) -> {upgrade_report['reason']}")

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
                        f"  gen={learner.evolution_runs}"
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

        plt.ioff()
        plt.close(fig)


if __name__ == "__main__":
    main()







