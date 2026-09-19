#!/usr/bin/env python3
"""Solve READY_RIGHT_ARM / READY_LEFT_ARM for scenes/layout.py (docs/DECISIONS.md D7).

Grippers point straight down at (READY_INSET from the near edge, +-READY_Y, READY_ABOVE_TABLE).
Wrist yaw is free: every yaw in 5 deg steps is solved and the one whose closest joint is farthest
from its limit wins. The left arm is solved independently (not assumed to be a mirror) and the
mirror relation is reported as a sanity check.

Usage: /data/isaac/isaacsim/bin/python tools/solve_ready_pose.py [--inset X] [--y Y] [--height H]
(defaults: READY_INSET / READY_Y / READY_ABOVE_TABLE from layout.py)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes"))
import layout as L  # noqa: E402
from kinematics import ArmKinematics, down_rotation  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--inset", type=float, default=L.READY_INSET)
ap.add_argument("--y", type=float, default=L.READY_Y)
ap.add_argument("--height", type=float, default=L.READY_ABOVE_TABLE)
a = ap.parse_args()
results = {}
for side, sign in (("right", -1.0), ("left", 1.0)):
    kin = ArmKinematics(side)
    target = np.array([L.TABLE_NEAR_X + a.inset, sign * a.y, L.TABLE_HEIGHT + a.height]) - np.array(L.ROBOT_POS)
    best = None
    for yaw in np.radians(np.arange(-180, 180, 5)):
        q, pe, re = kin.ik(target, down_rotation(yaw), (kin.lower + kin.upper) / 2, restarts=8)
        if pe > 1e-4 or re > 0.05:
            continue
        margin = np.min(np.minimum(q - kin.lower, kin.upper - q) / (kin.upper - kin.lower))
        if best is None or margin > best[0]:
            best = (margin, yaw, q, pe, re)
    if best is None:
        raise SystemExit(f"{side}: ready target unreachable")
    margin, yaw, q, pe, re = best
    results[side] = q
    print(f"{side}: yaw {np.degrees(yaw):+.0f} deg, err {pe * 1000:.3f} mm / {re:.3f} deg, min joint margin {margin:.1%}")
    print(f"READY_{side.upper()}_ARM = ({', '.join(f'{v:.4f}' for v in q)})")
print("mirror check (right + left for joints 1,2,3,5,6,7 and right - left for joint 4, ~0 if mirrored):",
      np.round(np.r_[results["right"][:3] + results["left"][:3], results["right"][3] - results["left"][3], results["right"][4:] + results["left"][4:]], 3))
