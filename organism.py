from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

MODULE_DIR = Path(__file__).resolve().parent / "modules"
sys.path.insert(0, str(MODULE_DIR))

from modules.registry import ModuleRegistry


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

    observer.attach_simulator(sim)

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

        learner.attach_simulator(sim)

    print()
    print("ACTIVE LEGOS")

    for name in registry.active():
        print(f"  * {name}")

    print()
    print("COMPUTE PROVIDERS")
    print("  * local-cpu")
    print("  o tanzania      disabled")
    print("  o tina          disabled")

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

    fig = plt.figure(figsize=(11, 8))

    # Dedicated figure margin for the organism dashboard.
    # The dashboard is figure-relative, so it no longer covers
    # the 3-D orbital world.
    fig.subplots_adjust(
        left=0.25,
        right=0.98,
        bottom=0.08,
        top=0.93,
    )
    ax = fig.add_subplot(111, projection="3d")

    # Persistent visual trails belong to the renderer, not the simulator.
    render_trails = {
        body.name: []
        for body in sim.PLANETS
    }

    last = time.perf_counter()

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
                * config["modules"]
                ["orbit_observer"]
                ["parameters"]
                ["simulation_days_per_second"]
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

            if learner is not None:
                learning_result = learner.observe(None)

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

            for body in sim.PLANETS:


                state = bodies.get(body.name)

                if state is None:
                    continue

                orbit = sim.planet_orbit(body)

                path3 = orbit.sample_path_3d(
                    n_points=720
                )

                target_x = state["x_au"]
                target_y = state["y_au"]

                best = min(
                    path3,
                    key=lambda point:
                        (
                            point[0] / sim.AU_KM - target_x
                        ) ** 2
                        +
                        (
                            point[1] / sim.AU_KM - target_y
                        ) ** 2
                )

                x = best[0] / sim.AU_KM
                y = best[1] / sim.AU_KM
                z = best[2] / sim.AU_KM

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

            # ----------------------------------------------------------
            # Orbital planes / paths.
            # ----------------------------------------------------------

            for body in sim.PLANETS:


                orbit = sim.planet_orbit(body)

                points = orbit.sample_path_3d(
                    n_points=360
                )

                xs = [
                    point[0] / sim.AU_KM
                    for point in points
                ]

                ys = [
                    point[1] / sim.AU_KM
                    for point in points
                ]

                zs = [
                    point[2] / sim.AU_KM
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
            # Sun.
            # ----------------------------------------------------------

            ax.scatter(
                [0],
                [0],
                [0],
                s=130,
                marker="o",
            )

            # ----------------------------------------------------------
            # Current planetary bodies.
            # ----------------------------------------------------------

            for name, position in positions_3d.items():

                x, y, z = position

                ax.scatter(
                    [x],
                    [y],
                    [z],
                    s=42,
                    marker="o",
                )

                ax.text(
                    x + 0.035,
                    y + 0.035,
                    z + 0.035,
                    name,
                    fontsize=8,
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

            cube_extent = max_extent + camera_pad

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
                "ORBITAL ORGANISM - CONTINUOUS 3-D WORLD"
            )

            ax.grid(
                True,
                alpha=0.20,
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

            lines = [
                "ORGANISM",
                "",
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
                "  local-cpu: available",
                "  Tanzania: disabled",
                "  Tina: disabled",
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
                        f"{learner.validation_loss:.8f}"
                    ),
                    (
                        f"  best: "
                        f"{learner.best_loss:.8f}"
                    ),
                    (
                        f"  width: "
                        f"{learner.hidden_width}"
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
            tick_count += 1
            if tick_count % 10 == 0:
                print(f'\r[Organism] Tick {tick_count}...', end='', flush=True)

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







