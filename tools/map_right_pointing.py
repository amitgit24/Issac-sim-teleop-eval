#!/usr/bin/env python3
"""Which finger directions can the right gripper reach near the handover area? (offline)

Fingers direction = Rz(az) applied to (0, cos el, -sin el): az=0 points at the left arm (+y),
el>0 tilts the fingers downward. Roll = rotation of the claws about the finger axis (0 = claws
above/below the can). Prints, per (az, el), the claw rolls reachable at several points.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes"))
import layout as L  # noqa: E402
from kinematics import ArmKinematics  # noqa: E402
from pick_planner import _seeds  # noqa: E402

kin = ArmKinematics("right")
base = np.array(L.ROBOT_POS)
q0 = np.array(L.READY_RIGHT_ARM)


def gripper_rot(az, el, roll):
    # base: fingers (-z_g) along +y, claws (y_g) along +z -> R_x(90); then roll about the finger
    # axis, tilt down about x, turn about world z.
    R0 = Rotation.from_euler("x", 90, degrees=True)
    return (Rotation.from_euler("z", az, degrees=True) * Rotation.from_euler("x", -el, degrees=True)
            * R0 * Rotation.from_euler("z", roll, degrees=True)).as_matrix()


pts = [(inset, y, h) for inset in (0.15, 0.25) for y in (-0.20, -0.10) for h in (0.25, 0.35)]
for az in (0, 20, 40, 60):
    for el in (0, 20, 40, 60):
        row = []
        for inset, y, h in pts:
            p = np.array([L.TABLE_NEAR_X + inset, y, L.TABLE_HEIGHT + h]) - base
            rolls = []
            for roll in range(-90, 91, 30):
                R = gripper_rot(az, el, roll)
                ok = False
                for seed in _seeds(kin, q0, n_random=6):
                    q, pe, re = kin.ik(p, R, seed, restarts=0)
                    if pe < 1e-3 and re < 0.5:
                        ok = True
                        break
                if ok:
                    rolls.append(roll)
            row.append(rolls)
        print(f"az {az:+3d} el {el:2d}: " + " | ".join(f"in{i:.2f} y{y:+.2f} h{h:.2f}:{r}" for (i, y, h), r in zip(pts, row)), flush=True)
