#!/usr/bin/env python3
"""Offline check of the teleop controller (pure numpy, no Isaac): from the ready pose, a simulated
operator turns the gripper, moves over each pick-zone cell and descends to grasp height, then lifts.
Counts how many of these reach tasks the controller completes (with automatic reconfiguration).
Usage: python tests/check_teleop_reach.py [--min-success N]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes"))
import layout as L  # noqa: E402
from kinematics import FINGER_Q_MAX, ArmKinematics  # noqa: E402
from teleop_controller import ArmState, TeleopController, Workspace  # noqa: E402

DT = 1.0 / 60.0
base = np.array(L.ROBOT_POS)
tz = L.TABLE_HEIGHT - base[2]
ws = Workspace(lo=np.array([L.TABLE_NEAR_X - 0.05 - base[0], -0.70, tz + 0.05]),
               hi=np.array([L.TABLE_NEAR_X + 0.914 - 0.05 - base[0], 0.70, tz + 0.60]), table_z=tz)
GRASP_EE = 0.19


def make():
    kin = {s: ArmKinematics(s) for s in ("right", "left")}
    home = {"right": np.array(L.READY_RIGHT_ARM), "left": np.array(L.READY_LEFT_ARM)}
    arms = {s: ArmState(kin=kin[s], q=home[s].copy()) for s in kin}
    for s, a in arms.items():
        a.finger = a.finger_goal = (1 if s == "left" else -1) * FINGER_Q_MAX
    return TeleopController(arms=arms, workspace=ws, active="right", home_q=home)


def drive_to(c, goal, yaw_turn, max_s=25.0):
    """Proportional 'operator': turn first, then move toward goal (root frame). True if reached."""
    a = c.arms["right"]
    turned = 0.0
    for _ in range(int(max_s / DT)):
        if c.reconfiguring:
            c.step(np.zeros(3), np.zeros(3), DT)
            continue
        if abs(yaw_turn - turned) > 0.01:
            w = np.clip((yaw_turn - turned) / 0.2, -1, 1)
            before = a.rot.copy()
            c.step(np.zeros(3), np.array([0, 0, w]), DT)
            dz = np.arctan2(a.rot[1, 0], a.rot[0, 0]) - np.arctan2(before[1, 0], before[0, 0])
            turned += (dz + np.pi) % (2 * np.pi) - np.pi
            continue
        err = goal - a.pos
        if np.linalg.norm(err) < 0.004:
            return True
        c.step(np.clip(err / 0.03, -1, 1), np.zeros(3), DT)
    return False


def feasible(c, x, y, turn):
    """Does ANY IK solution exist for the grasp and lift poses at this yaw? (not a continuity test)"""
    from pick_planner import _seeds

    a = c.arms["right"]
    rot = np.array([[np.cos(turn), -np.sin(turn), 0], [np.sin(turn), np.cos(turn), 0], [0, 0, 1]]) @ a.rot
    for z in (tz + GRASP_EE, tz + GRASP_EE + 0.12):
        if not any(a.kin.ik(np.array([x, y, z]), rot, s, restarts=0)[1] < 1e-3 for s in _seeds(a.kin, a.q, n_random=8)):
            return False
    return True


def main():
    ok = total = possible = ok_possible = 0
    for dx in (0.10, 0.15, 0.20):
        for y in (-0.35, -0.30, -0.25):
            for turn in np.radians((0, 45, -45, 90)):
                c = make()
                x = L.TABLE_NEAR_X + dx - base[0]
                can = feasible(c, x, y, turn)
                p0 = c.arms["right"].pos
                done = drive_to(c, np.array([x, y, p0[2]]), turn) and drive_to(c, np.array([x, y, tz + GRASP_EE]), 0.0) \
                    and drive_to(c, np.array([x, y, tz + GRASP_EE + 0.12]), 0.0)
                ok += done
                total += 1
                possible += can
                ok_possible += done and can
                if can and not done:
                    print(f"  missed a feasible task: inset {dx:.2f} y {y:+.2f} turn {np.degrees(turn):+.0f}", flush=True)
    print(f"TELEOP REACH: {ok}/{total} tasks completed; {possible} of the {total} are physically reachable at all; "
          f"completed {ok_possible}/{possible} of those")
    need = int(sys.argv[sys.argv.index("--min-success") + 1]) if "--min-success" in sys.argv else 0
    if ok < need:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
