#!/usr/bin/env python3
"""Find a teleop home pose from which the right arm can move CONTINUOUSLY (step-by-step IK, like
live teleop) to grasp height anywhere in the pick zone, after first turning the gripper yaw.

The 0.40 m ready pose sits on an IK branch that cannot descend to the table continuously
(docs/MISTAKES.md M35), so teleop gets its own lower home, placed beside (not above) the pick zone
so tall objects never start under the claws. Offline; prints candidates and the best joints.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes"))
import layout as L  # noqa: E402
from kinematics import ArmKinematics, down_rotation  # noqa: E402
from pick_planner import LOW_REACH_SEED_RIGHT, _seeds  # noqa: E402

kin = ArmKinematics("right")
base = np.array(L.ROBOT_POS)
tz = L.TABLE_HEIGHT - base[2]
GRASP_EE = 0.19  # gripper origin height above the table for a low grasp (cube)


MID = float(next((a for a in sys.argv[1:] if not a.startswith("--")), 0.0))  # IK joint-centering weight (teleop)


def follow(q, targets, R_seq):
    for t, R in zip(targets, R_seq):
        qn, pe, re = kin.ik(t, R, q, restarts=0, mid_weight=MID)
        if pe > 5e-3 or re > 2.0 or np.max(np.abs(qn - q)) > 0.3:
            return None
        q = qn
    return q


def reachable(q_home, yaw_home, x, y, dyaw):
    p0 = kin.fk(q_home)[0]
    yaws = np.linspace(yaw_home, yaw_home + dyaw, 12)
    q = follow(q_home, [p0] * 12, [down_rotation(a) for a in yaws])  # turn at home height
    if q is None:
        return False
    R = down_rotation(yaw_home + dyaw)
    horiz = [p0 + (np.array([x, y, p0[2]]) - p0) * s for s in np.linspace(0, 1, 25)]
    down = [np.array([x, y, z]) for z in np.linspace(p0[2], tz + GRASP_EE, 50)]
    q = follow(q, horiz + down, [R] * 75)
    if q is None:
        return False
    up = [np.array([x, y, z]) for z in np.linspace(tz + GRASP_EE, tz + GRASP_EE + 0.15, 20)]
    return follow(q, up, [R] * 20) is not None


cells = [(L.TABLE_NEAR_X + dx - base[0], y) for dx in (0.10, 0.15, 0.20) for y in (-0.35, -0.30, -0.25)]
turns = np.radians((0, 45, -45, 90))
best = None
HOMES = [(0.40, 0.15, -0.25), (0.28, 0.10, -0.18)] if "--fixed" in sys.argv else [(h, i, y) for h in (0.28, 0.32) for i in (0.10, 0.15) for y in (-0.12, -0.18)]
for h, inset, yh in HOMES:
    if True:
        if True:
            p = np.array([L.TABLE_NEAR_X + inset, yh, L.TABLE_HEIGHT + h]) - base
            for yaw in np.radians(np.arange(0, 360, 30)):
                for seed in (np.array(L.READY_RIGHT_ARM), LOW_REACH_SEED_RIGHT, *_seeds(kin, LOW_REACH_SEED_RIGHT, n_random=3)[2:]):
                    q, pe, re = kin.ik(p, down_rotation(yaw), seed, restarts=0)
                    if pe > 1e-3 or re > 0.5:
                        continue
                    n = sum(reachable(q, yaw, x, y, d) for x, y in cells for d in turns)
                    m = np.min(np.minimum(q - kin.lower, kin.upper - q) / (kin.upper - kin.lower))
                    score = (n, m)
                    if best is None or score > best[0]:
                        best = (score, h, inset, yh, np.degrees(yaw), q)
                        print(f"h {h:.2f} inset {inset:.2f} y {yh:+.2f} yaw {np.degrees(yaw):4.0f}: {n}/{len(cells) * len(turns)} "
                              f"continuous reaches, margin {m:.1%}", flush=True)
                    break
(n, m), h, inset, yh, yaw, q = best
print(f"BEST: height {h} inset {inset} y {yh} yaw {yaw:.0f} -> {n}/{len(cells) * len(turns)}, margin {m:.1%}")
print(f"TELEOP_HOME_RIGHT = ({', '.join(f'{v:.4f}' for v in q)})")
