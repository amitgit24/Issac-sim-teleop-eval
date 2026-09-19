#!/usr/bin/env python3
"""Map where an arm can pick a cube with fingers down (offline, uses scenes/kinematics.py).

Criterion per table spot and object yaw (cube: grasp yaw = object yaw + k*90 deg, any k works):
  pre-grasp (grasp + PRE) -> straight vertical descent to grasp -> straight lift (+LIFT),
  all at the grasp yaw, IK chained point to point, every joint >= 3% of range from limits and
  no joint jumping > MAX_JUMP rad between consecutive points (would be an IK branch flip).
Prints, per spot, how many of the sampled object yaws are pickable.

Usage: python tools/map_pick_workspace.py [--side right] [--shoulder 0.35] [--gap 0.12] [--grasp-h 0.19]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes"))
import layout as L  # noqa: E402
from kinematics import ArmKinematics, down_rotation  # noqa: E402

TABLE_NEAR_X, TABLE_HEIGHT, SHOULDER_Z = L.TABLE_NEAR_X, L.TABLE_HEIGHT, L.OPENARM_SHOULDER_Z
PRE, LIFT, MAX_JUMP = 0.08, 0.12, 0.3
READY = {"left": L.READY_LEFT_ARM, "right": L.READY_RIGHT_ARM}


def plan_ok(kin, base, x, y, grasp_h, yaw, seed):
    """True if pre-grasp -> descend -> lift is reachable and continuous at this grasp yaw."""
    rot = down_rotation(yaw)
    heights = [grasp_h + PRE * f for f in (1.0, 0.75, 0.5, 0.25, 0.0)] + [grasp_h + LIFT * f for f in (0.5, 1.0)]
    q_prev = None
    for i, dz in enumerate(heights):
        q, pe, re = kin.ik(np.array([x, y, TABLE_HEIGHT + dz]) - base, rot, seed, restarts=4 if i == 0 else 0)
        if pe > 1e-3 or re > 0.5:
            return False
        if q_prev is not None and np.max(np.abs(q - q_prev)) > MAX_JUMP:
            return False
        q_prev, seed = q, q
    return True


def pickable_fraction(kin, base, x, y, grasp_h, obj_yaws, seed):
    n = 0
    for oy in obj_yaws:
        n += any(plan_ok(kin, base, x, y, grasp_h, oy + k * np.pi / 2, seed) for k in range(4))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right")
    ap.add_argument("--shoulder", type=float, default=L.SHOULDER_ABOVE_TABLE, help="shoulder height above tabletop")
    ap.add_argument("--gap", type=float, default=L.EDGE_GAP, help="robot base behind the table near edge")
    ap.add_argument("--grasp-h", type=float, default=0.19, help="ee_base_link height above table at grasp")
    a = ap.parse_args()
    kin = ArmKinematics(a.side)
    base = np.array([TABLE_NEAR_X - a.gap, 0.0, TABLE_HEIGHT + a.shoulder - SHOULDER_Z])
    obj_yaws = np.radians(np.arange(0, 90, 15))
    sign = -1 if a.side == "right" else 1
    ys = sign * np.arange(0.0, 0.46, 0.05)
    print(f"{a.side} arm, shoulder {a.shoulder} m above table, gap {a.gap} m, grasp ee h {a.grasp_h} m. "
          f"Cell = pickable object yaws out of {len(obj_yaws)}. Rows: inset from near edge; cols: world y")
    print("         " + " ".join(f"{y:+.2f}" for y in ys))
    total = 0
    for dx in np.arange(0.05, 0.46, 0.05):
        row = [pickable_fraction(kin, base, TABLE_NEAR_X + dx, y, a.grasp_h, obj_yaws, np.array(READY[a.side])) for y in ys]
        total += sum(r == len(obj_yaws) for r in row)
        print(f"in {dx:.2f}  " + "  ".join(f"{r:>3} " if r else "  . " for r in row), flush=True)
    print(f"FULLY PICKABLE CELLS (all object yaws): {total}")


if __name__ == "__main__":
    main()
