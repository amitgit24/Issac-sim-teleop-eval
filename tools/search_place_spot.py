#!/usr/bin/env python3
"""Where can each container sit so the right arm can pick at the pick zone and place into it?
Offline (real planners). Prints feasible (inset, y) spots per container for a held object.
Usage: python tools/search_place_spot.py [object: soup_can|cube|mug_a2]
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes"))
import inventory as INV  # noqa: E402
import layout as L  # noqa: E402
from kinematics import FINGER_Q_MAX, ArmKinematics  # noqa: E402
from pick_planner import GraspSpec  # noqa: E402
from place_planner import plan_pick_place  # noqa: E402

HELD = {"soup_can": GraspSpec(width=0.068, yaw_symmetry=math.pi / 6, height=0.102),
        "cube": GraspSpec(width=0.05, yaw_symmetry=math.pi / 2, height=0.05),
        "mug_a2": GraspSpec(width=0.092, yaw_symmetry=math.pi, height=0.091)}
held = sys.argv[1] if len(sys.argv) > 1 else "soup_can"
spec = HELD[held]
kin = ArmKinematics("right")
base = np.array(L.ROBOT_POS)
tz = L.TABLE_HEIGHT - base[2]
obj = np.array([L.PICK_CENTER[0], L.PICK_CENTER[1], L.TABLE_HEIGHT + spec.height / 2]) - base
for it in INV.with_role("place_target"):
    sx, sy, sz = it.size_m
    r = 0.5 * math.hypot(sx, sy)
    ok = []
    reasons = {}
    for inset in np.arange(0.06, 0.41, 0.04):
        for y in np.arange(-0.04, -0.69, -0.04):
            x = L.TABLE_NEAR_X + inset
            if math.hypot(x - L.PICK_CENTER[0], y - L.PICK_CENTER[1]) < r + 0.09:
                continue  # container would overlap the pick zone
            if abs(y) + r > L.TABLE_TOP_SIZE[1] / 2 - 0.02 or inset - r < 0.02:
                continue  # off the table
            segs, why = plan_pick_place(kin, np.array(L.READY_RIGHT_ARM), -FINGER_Q_MAX, obj, np.array([0.0, 1.0]), tz, spec,
                                        (x - base[0], y), tz + sz, min(0.03, 0.5 * sz))
            reasons[why] = reasons.get(why, 0) + 1
            if segs is not None:
                ok.append((round(float(inset), 2), round(float(y), 2)))
    print(f"{it.name:13s} size {tuple(round(v, 3) for v in it.size_m)}: {len(ok)} feasible spots {ok}  reasons {reasons}", flush=True)
