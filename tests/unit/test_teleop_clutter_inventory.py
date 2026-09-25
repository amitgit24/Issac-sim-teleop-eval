import math
from pathlib import Path

import numpy as np

import clutter as CL
import inventory as INV
import layout as L
from kinematics import FINGER_Q_MAX, ArmKinematics, tip_depth
from teleop_controller import ArmState, TeleopController, Workspace

ROOT = Path(__file__).resolve().parents[2]
BASE = np.array(L.ROBOT_POS)
TZ = L.TABLE_HEIGHT - BASE[2]
DT = 1.0 / 60.0


def controller():
    kin = {s: ArmKinematics(s) for s in ("left", "right")}
    home = {"right": np.array(L.READY_RIGHT_ARM), "left": np.array(L.READY_LEFT_ARM)}
    arms = {s: ArmState(kin=kin[s], q=home[s].copy()) for s in kin}
    for s, a in arms.items():
        a.finger = a.finger_goal = (1 if s == "left" else -1) * FINGER_Q_MAX
    ws = Workspace(lo=np.array([L.TABLE_NEAR_X - 0.05 - BASE[0], -0.7, TZ + 0.05]),
                   hi=np.array([L.TABLE_NEAR_X + 0.86 - BASE[0], 0.7, TZ + 0.6]), table_z=TZ)
    return TeleopController(arms=arms, workspace=ws, home_q=home)


def test_teleop_never_commands_fingertips_below_table():
    c = controller()
    for _ in range(400):  # hold "down" for 6.7 s
        c.step(np.array([0, 0, -1.0]), np.zeros(3), DT)
        a = c.arms["right"]
        p, r = a.kin.fk(a.q)
        assert p[2] - tip_depth(a.finger) * r[2, 2] >= TZ - 1e-3


def test_teleop_moves_active_arm_only():
    c = controller()
    left0 = c.arms["left"].q.copy()
    p0 = c.arms["right"].pos.copy()
    for _ in range(30):
        c.step(np.array([1.0, 0, 0]), np.zeros(3), DT)
    assert c.arms["right"].pos[0] > p0[0] + 0.05
    assert np.allclose(c.arms["left"].q, left0)


def test_gripper_toggle_ramps():
    c = controller()
    c.toggle_gripper()
    c.step(np.zeros(3), np.zeros(3), DT)
    f = c.arms["right"].finger
    assert -FINGER_Q_MAX < f < 0  # moving toward closed, rate-limited


def test_clutter_layout_is_non_overlapping_and_on_the_table():
    rng = np.random.default_rng(3)
    items = CL.choose_items(rng, 8)
    layout = CL.sample_layout(rng, items)
    placed = [(it, s) for it, s in layout if s is not None]
    assert len(placed) >= 6
    for i, (a, sa) in enumerate(placed):
        assert abs(sa[1]) + CL.footprint(a) <= L.TABLE_TOP_SIZE[1] / 2
        for z in CL.default_exclusions():
            assert math.hypot(sa[0] - z.x, sa[1] - z.y) >= CL.footprint(a) + z.r - 1e-9
        for b, sb in placed[i + 1:]:
            assert math.hypot(sa[0] - sb[0], sa[1] - sb[1]) >= CL.footprint(a) + CL.footprint(b) - 1e-9


def test_inventory_is_verified_and_consistent():
    assert len(INV.ITEMS) == 21
    for it in INV.ITEMS:
        assert it.verified, it.name
        assert len(it.size_m) == 3 and min(it.size_m) > 0
        if "://" not in it.usd:
            assert (ROOT / it.usd).exists(), it.usd
        if "pick" in it.roles and it.grasp_width:
            assert it.grasp_width + 0.04 <= 0.139 + 1e-6  # fits the gripper's opening with margin
    assert {"place_target"} <= {r for it in INV.ITEMS for r in it.roles}
