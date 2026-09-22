#!/usr/bin/env python3
"""
Host-parameterized dispatch, generalizing dispatch_tanzania.py's own
pattern (mkdir, scp code + payload, ssh run remote_runner.py) to work
against ANY configured fleet host, not just Tanzania.

dispatch_tanzania.py itself is untouched -- organism.py's own dispatch
(explore_mutation_space, generate_world_textures) keeps using it
exactly as before. This exists for body-builder's own multi-host needs
(splitting generation load across Tanzania/Tina, cross-host validation
of accepted candidates), which need a host CHOSEN PER CALL -- threading
that through every one of dispatch_tanzania.py's existing functions
would have meant changing an already-stable, already-tested file for a
need only body-builder has.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ORGANISM_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = ORGANISM_DIR / "state"
MODULES_DIR = ORGANISM_DIR / "modules"
TOOLS_DIR = Path(__file__).resolve().parent

REMOTE_DIR = "orbital_dispatch"


def _run(cmd: list[str], timeout: float = 60, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kwargs)


def check_reachable(host: str) -> bool:
    result = _run([
        "ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", host, "echo ok",
    ])
    return result.returncode == 0 and "ok" in result.stdout


def deploy_code(host: str) -> None:
    # remote_runner.py imports neural_learner at module load time
    # regardless of which task is invoked, so all five files are
    # needed on any host running ANY task from it, not just the ones
    # that task actually touches. cannon_lib is only imported inside
    # run_train_local_physics(), but shipped unconditionally too --
    # it's small, and conditional shipping based on which task a given
    # call happens to need is real complexity for negligible savings.
    _run(["ssh", host, f"mkdir -p {REMOTE_DIR}/cannon_lib"])
    # scp -r into an already-existing destination nests source-under-
    # destination (cannon_lib/cannon/cannon) rather than replacing it --
    # removing the leaf first lets scp -r create it fresh as the copy
    # target instead.
    _run(["ssh", host, f"rm -rf {REMOTE_DIR}/cannon_lib/cannon"])
    result = _run(
        ["scp", "-r", str(MODULES_DIR / "cannon_lib" / "cannon"), f"{host}:{REMOTE_DIR}/cannon_lib/"],
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"scp cannon_lib to {host} failed: {result.stderr}")

    for name in ("neural_learner.py", "lego.py", "loss_blocks.py", "scratch_blocks.py"):
        result = _run(["scp", str(MODULES_DIR / name), f"{host}:{REMOTE_DIR}/{name}"])
        if result.returncode != 0:
            raise RuntimeError(f"scp {name} to {host} failed: {result.stderr}")

    result = _run([
        "scp", str(TOOLS_DIR / "remote_runner.py"), f"{host}:{REMOTE_DIR}/remote_runner.py",
    ])
    if result.returncode != 0:
        raise RuntimeError(f"scp remote_runner.py to {host} failed: {result.stderr}")


def dispatch(host: str, task: str, payload: dict) -> dict:
    if not check_reachable(host):
        raise RuntimeError(f"{host} is not reachable over SSH right now")

    deploy_code(host)

    local_payload_path = STATE_DIR / f"_fleet_payload_{host}.json"
    local_payload_path.write_text(json.dumps(payload), encoding="utf-8")

    scp_result = _run(["scp", str(local_payload_path), f"{host}:{REMOTE_DIR}/payload.json"])
    if scp_result.returncode != 0:
        raise RuntimeError(f"scp payload to {host} failed: {scp_result.stderr}")

    run_result = subprocess.run(
        ["ssh", host, f"cd {REMOTE_DIR} && python3 remote_runner.py payload.json"],
        capture_output=True, text=True, timeout=120,
    )
    if run_result.returncode != 0:
        raise RuntimeError(f"remote run on {host} failed: {run_result.stderr}")

    return json.loads(run_result.stdout)
