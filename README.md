# Orbital Organism

A self-evolving neural network that learns orbital mechanics — safely
mutates its own loss functions and hyperparameters via transactional
validation, proposes novel loss cores through Gemini, grows its own
rendering fidelity per body ("body-builder"), and offloads exploration
to a compute fleet, all rendered live in a 3D dashboard.

## Requirements

**Core (runs standalone, one machine):**
- Python 3.13+
- `numpy`, `matplotlib`
- The base toolkit this plank sits on: `../../orbital-sandbox.py` (one
  directory up from `prototypes/`)

**Optional, degrades gracefully if absent:**
- `psutil` — powers the idle-aware CPU/memory budget (`modules/
  organic_budget.py`). Without it, the render budget falls back to a
  flat, static cap instead of scaling with real machine load. Install
  with `pip install psutil`.
- A Gemini API key — enables `tools/propose_scratch_tree.py` to
  propose novel loss-function shapes via the real Gemini API instead
  of the local random/best-of-N fallback. Put `GEMINI_API_KEY=...` in
  a `proposer.env` file at the repo root (gitignored, never committed).

**Full experience (fleet dispatch, body-builder, cross-host
validation) — genuinely needs 3 machines:**
- This machine (coordinator + renderer)
- Two more, reachable over SSH, with **passwordless key-based auth**
  configured as host aliases named `tanzania` and `tina` in
  `~/.ssh/config`:

  ```
  Host tanzania
      HostName <its address>
      User <your user>
      IdentityFile ~/.ssh/id_ed25519

  Host tina
      HostName <its address>
      User <your user>
      IdentityFile ~/.ssh/id_ed25519
  ```

  Both need `python3` and `numpy` installed. Nothing else — the actual
  code (`neural_learner.py`, `loss_blocks.py`, `scratch_blocks.py`,
  `lego.py`, `remote_runner.py`) is shipped to them fresh on every
  dispatch, not pre-installed.

If `tanzania`/`tina` aren't reachable, this isn't a hard failure: the
main organism (`organism.py`) still runs, self-evolves, and renders —
fleet-dependent features (mutation-space sweeps, world-texture
generation, body-builder) just have nothing to dispatch to and no-op
until a host comes back online.

## Running it

```
python organism.py
```

Boots the live dashboard: a self-evolving neural net predicting
planetary positions, rendered as a continuous 3D solar system, with
its own generation/success history, loss-function/activation state,
and (if a Gemini key is configured) its own scratch-tree proposal
history, all visible in the left panel.

To grow real per-body texture fidelity (needs `tanzania`/`tina`),
start this as a **separate, long-running process** — it dispatches
over SSH every cycle, which has no place in the render loop's frame
budget:

```
cd tools
python run_body_builder.py --interval 60
```

Round-robins through every body at least Ceres-sized
(`modules/real_systems.py`'s `registers_as_body()` — smaller bodies
don't register as part of the organism at all), proposing one
candidate texture-generation move per cycle, splitting the real
compute across both fleet hosts, and independently re-validating every
accepted candidate on whichever host didn't generate it. Its live
progress shows in `organism.py`'s own dashboard (bottom panel,
re-read every 10s) whenever both processes are running together.

Manual, one-off dispatches (`tools/dispatch_tanzania.py
explore_mutation_space`, `tools/fetch_exoplanet_data.py`, `tools/
propose_scratch_tree.py`) are documented in their own module
docstrings.

## What's real vs. designed-only

Everything under `modules/` and `tools/` referenced above is real,
tested, and running. `TRANSITION_INTERFACE.md`-style cross-domain work
(unifying this project's multi-body orbital regime with a separate
single-body ballistics domain) is a design note in a sibling
experiment, not part of this repo, and not yet built.
