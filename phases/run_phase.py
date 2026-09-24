#!/usr/bin/env python3
"""Programmatic Phase Runner and Adaptive Dependency Engine for ATLAS.

Allows AI research agents and human investigators to inspect, verify,
and execute research phases, trigger automated failure diagnostics,
and cascade dependency invalidations.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Optional


PHASES_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = PHASES_DIR.parent
STATE_FILE = PHASES_DIR / "state.json"


def load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        raise FileNotFoundError(f"State file {STATE_FILE} not found.")
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def print_status_table(state: Dict[str, Any]) -> None:
    print("=" * 80)
    print(f"  ATLAS Research Phases: Status & Dependency Overview")
    print(f"  Target Pod: {state.get('hardware_target', 'Unknown')}")
    print("=" * 80)
    phases = state.get("phases", {})
    print(f"{'ID':<4} | {'Status':<12} | {'Phase Name':<45} | {'Deps'}")
    print("-" * 80)
    for pid_str, p in sorted(phases.items(), key=lambda x: int(x[0])):
        pid = p["id"]
        status = p["status"]
        name = p["name"]
        if len(name) > 43:
            name = name[:40] + "..."
        deps = ",".join(str(d) for d in p.get("dependencies", [])) or "None"
        print(f"{pid:<4} | {status:<12} | {name:<45} | {deps}")
    print("=" * 80)


def verify_phase(phase_id: int, state: Dict[str, Any], verbose: bool = True) -> bool:
    pid_str = str(phase_id)
    phase = state.get("phases", {}).get(pid_str)
    if not phase:
        print(f"Error: Phase {phase_id} not found in state registry.")
        return False

    if verbose:
        print(f"\n[PHASE {phase_id}] Verifying: {phase['name']}")

    known_gap = phase.get("known_gap")
    if known_gap:
        print(f"  [FAIL] Unresolved evidence gap: {known_gap}")
        return False

    # 1. Check phase markdown file exists
    doc_path = REPO_ROOT / phase.get("file", "")
    if not doc_path.exists():
        print(f"  [FAIL] Phase document {doc_path} does not exist.")
        return False
    if verbose:
        print(f"  [OK] Documentation: {doc_path.name}")

    # 2. Check declared artifacts exist
    artifacts = phase.get("artifacts", [])
    missing_artifacts = []
    for art in artifacts:
        art_path = REPO_ROOT / art
        if not art_path.exists():
            missing_artifacts.append(art)
    if missing_artifacts:
        print(f"  [FAIL] Missing declared artifacts: {missing_artifacts}")
        return False
    if verbose:
        print(f"  [OK] Artifacts verified ({len(artifacts)} files).")

    # 3. Check verification command if present
    cmd = phase.get("verification_command")
    if cmd:
        if verbose:
            print(f"  [RUN] Executing verification check: {cmd}")
        env = dict(os.environ)
        venv_bin = REPO_ROOT / ".venv" / "bin"
        if venv_bin.exists():
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        elan_bin = pathlib.Path.home() / ".elan" / "bin"
        if elan_bin.exists():
            env["PATH"] = f"{elan_bin}:{env.get('PATH', '')}"
        if "JAX_PLATFORMS" not in env:
            env["JAX_PLATFORMS"] = "cpu"
        res = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)
        if res.returncode != 0:
            print(f"  [FAIL] Verification command failed (exit code {res.returncode}):")
            print(f"    STDOUT: {res.stdout.strip()}")
            print(f"    STDERR: {res.stderr.strip()}")
            return False
        if verbose:
            print(f"  [OK] Verification command passed.")

    if verbose:
        print(f"  [SUCCESS] Phase {phase_id} verified successfully!")
    return True


def cascade_invalidation(phase_id: int, state: Dict[str, Any]) -> List[int]:
    """Finds all downstream phases that depend directly or indirectly on phase_id."""
    invalidated = set()
    queue = [phase_id]
    phases = state.get("phases", {})

    while queue:
        curr = queue.pop(0)
        for pid_str, p in phases.items():
            pid = int(pid_str)
            if curr in p.get("dependencies", []) and pid not in invalidated:
                invalidated.add(pid)
                queue.append(pid)

    return sorted(list(invalidated))


def main() -> None:
    parser = argparse.ArgumentParser(description="ATLAS Autonomous Research Phase Runner")
    parser.add_argument("--status", action="store_true", help="Print table of all phases and statuses")
    parser.add_argument("--phase", type=int, default=None, help="Specific phase ID to verify or run (1-8)")
    parser.add_argument("--all", action="store_true", help="Verify all phases sequentially")
    parser.add_argument("--cascade", action="store_true", help="Cascade invalidation to downstream dependent phases")
    args = parser.parse_args()

    state = load_state()

    if args.status or (args.phase is None and not args.all):
        print_status_table(state)
        return

    if args.phase is not None:
        if args.cascade:
            downstream = cascade_invalidation(args.phase, state)
            print(f"\n[CASCADE] Modifying Phase {args.phase} invalidates downstream phases: {downstream}")
            for d in downstream:
                state["phases"][str(d)]["status"] = "PENDING"
            save_state(state)
            print(f"[CASCADE] Updated state registry. Downstream phases marked PENDING.")

        success = verify_phase(args.phase, state, verbose=True)
        if success:
            state["phases"][str(args.phase)]["status"] = "COMPLETED"
        else:
            state["phases"][str(args.phase)]["status"] = "NEEDS_CORRECTION"
        save_state(state)
        sys.exit(0 if success else 1)

    if args.all:
        all_ok = True
        for pid in sorted([int(k) for k in state.get("phases", {}).keys()]):
            ok = verify_phase(pid, state, verbose=True)
            if not ok:
                all_ok = False
                state["phases"][str(pid)]["status"] = "NEEDS_CORRECTION"
                save_state(state)
                print(f"\n[HALT] Pipeline failed at Phase {pid}. Autonomous remediation required.")
                sys.exit(1)
            else:
                state["phases"][str(pid)]["status"] = "COMPLETED"
        save_state(state)
        print("\n" + "=" * 80)
        print("  ALL PHASES VERIFIED AND COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        sys.exit(0)


if __name__ == "__main__":
    main()
