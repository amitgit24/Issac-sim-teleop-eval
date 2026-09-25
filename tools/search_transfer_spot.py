#!/usr/bin/env python3
"""Where can the bin sit so the full pick -> handover -> left place chain is feasible? (offline)

Runs transfer_planner.plan_handover_place for several objects over bin positions on the left side of
the table. Spots where the bin would block the pick zone or the left arm's low approach to the
handover point (a corridor from the handover point toward +y) are skipped.
Usage: python tools/search_transfer_spot.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes"))
import inventory as INV  # noqa: E402
import layout as L  # noqa: E402
from handover_planner import HandoverPoint  # noqa: E402
from kinematics import FINGER_Q_MAX, ArmKinematics  # noqa: E402
from pick_planner import GraspSpec  # noqa: E402
from transfer_planner import plan_handover_place  # noqa: E402

base = np.array(L.ROBOT_POS)
tz = L.TABLE_HEIGHT - base[2]
kr, kl = ArmKinematics("right"), ArmKinematics("left")
bin_ = INV.BY_NAME["klt_bin"]
bx, by, bh = bin_.size_m
hp = HandoverPoint(x=L.TABLE_NEAR_X + 0.12 - base[0], y=-0.04, z=L.TABLE_HEIGHT + 0.36 - base[2])
hp_world = (L.TABLE_NEAR_X + 0.12, -0.04)
OBJECTS = {
    "soup_can": GraspSpec(width=0.068, yaw_symmetry=math.pi / 6, height=0.102),
    "mustard": GraspSpec(width=0.058, yaw_symmetry=math.pi, height=0.191),
    "glass_short": GraspSpec(width=0.081, yaw_symmetry=math.pi / 6, height=0.0953),
    "mug_c1": GraspSpec(width=0.089, yaw_symmetry=math.pi, height=0.1072, yaw_tolerance=0.61),
}


def blocked(x, y):
    # bin footprint (axis-aligned) vs the pick zone and the left approach corridor
    if abs(x - L.PICK_CENTER[0]) < bx / 2 + 0.10 and abs(y - L.PICK_CENTER[1]) < by / 2 + 0.10:
        return True
    corridor = (hp_world[0] - 0.14, hp_world[0] + 0.14, hp_world[1], hp_world[1] + 0.50)
    return not (x + bx / 2 < corridor[0] or x - bx / 2 > corridor[1] or y + by / 2 < corridor[2] or y - by / 2 > corridor[3])


results = {}
for inset in np.arange(0.12, 0.46, 0.04):
    for y in np.arange(0.12, 0.64, 0.04):
        x = L.TABLE_NEAR_X + inset
        if blocked(x, y) or y + by / 2 > L.TABLE_TOP_SIZE[1] / 2 - 0.02 or inset - bx / 2 < 0.02 or inset + bx / 2 > L.TABLE_TOP_SIZE[0] - 0.02:
            continue
        ok = []
        for name, spec in OBJECTS.items():
            obj = np.array([L.PICK_CENTER[0], L.PICK_CENTER[1], L.TABLE_HEIGHT + spec.height / 2]) - base
            segs, why = plan_handover_place(kr, kl, np.array(L.READY_RIGHT_ARM), -FINGER_Q_MAX, np.array(L.READY_LEFT_ARM), FINGER_Q_MAX,
                                            obj, tz, spec, hp, (x - base[0], y - base[1]), tz + bh)
            ok.append(segs is not None)
            if segs is None:
                results.setdefault(why, 0)
                results[why] += 1
        print(f"bin inset {inset:.2f} y {y:+.2f}: {sum(ok)}/{len(ok)} objects {'ALL' if all(ok) else ''}", flush=True)
print("failure reasons:", results)
