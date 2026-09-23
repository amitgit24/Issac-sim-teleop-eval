#!/usr/bin/env python3
"""Search handover points (robot root frame) where the full two-arm handover plan is feasible.

Runs scenes/handover_planner.plan_handover (the real planner) for a soup can at the pick-zone center
over a grid of handover x (table inset), can y, and right gripper height above the table.
Usage: /data/isaac/isaacsim/bin/python tools/search_handover_point.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes"))
import layout as L  # noqa: E402
from handover_planner import HandoverPoint, plan_handover  # noqa: E402
from kinematics import FINGER_Q_MAX, ArmKinematics  # noqa: E402
from pick_planner import GraspSpec  # noqa: E402

base = np.array(L.ROBOT_POS)
table_z = L.TABLE_HEIGHT - base[2]
can = GraspSpec(width=0.068, yaw_symmetry=np.pi / 6, height=0.102)  # = pick_task soup_can
obj = np.array([L.PICK_CENTER[0], L.PICK_CENTER[1], L.TABLE_HEIGHT + 0.051]) - base
kr, kl = ArmKinematics("right"), ArmKinematics("left")
qr0, ql0 = np.array(L.READY_RIGHT_ARM), np.array(L.READY_LEFT_ARM)
ok_pts = []
for inset in (0.12, 0.20, 0.28):
    for y in (-0.10, -0.04, 0.02, 0.08):
        for h in (0.30, 0.36, 0.42):  # right gripper origin height above the table
            hp = HandoverPoint(x=L.TABLE_NEAR_X + inset - base[0], y=y, z=L.TABLE_HEIGHT + h - base[2])
            segs, why = plan_handover(kr, kl, qr0, -FINGER_Q_MAX, ql0, FINGER_Q_MAX, obj, table_z, can, hp)
            print(f"inset {inset:.2f} y {y:+.2f} height {h:.2f}: {why}", flush=True)
            if segs is not None:
                ok_pts.append((inset, y, h))
print("FEASIBLE:", ok_pts)
