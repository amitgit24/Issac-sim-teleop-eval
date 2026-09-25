import numpy as np
import pytest

import layout as L
from kinematics import FINGER_Q_MAX, GAP_PER_RAD, ArmKinematics, down_rotation, finger_q_for_gap, tip_depth

BASE = np.array(L.ROBOT_POS)


@pytest.fixture(scope="module")
def arms():
    return {s: ArmKinematics(s) for s in ("left", "right")}


def test_chains_have_seven_joints(arms):
    for kin in arms.values():
        assert len(kin.joint_names) == 7
        assert np.all(kin.lower < kin.upper)


@pytest.mark.parametrize("side,sign", [("right", -1.0), ("left", 1.0)])
def test_ready_pose_puts_gripper_on_its_target(arms, side, sign):
    q = np.array(L.READY_RIGHT_ARM if side == "right" else L.READY_LEFT_ARM)
    p, r = arms[side].fk(q)
    target = np.array([L.TABLE_NEAR_X + L.READY_INSET, sign * L.READY_Y, L.TABLE_HEIGHT + L.READY_ABOVE_TABLE]) - BASE
    assert np.linalg.norm(p - target) < 1e-4
    assert r[2, 2] > 0.9999  # fingers point straight down


def test_arms_are_mirror_images(arms):
    qr = np.array(L.READY_RIGHT_ARM)
    ql = qr * np.array([-1, -1, -1, 1, -1, -1, -1])
    pr, _ = arms["right"].fk(qr)
    pl, _ = arms["left"].fk(ql)
    assert np.allclose(pr * [1, -1, 1], pl, atol=1e-6)


def test_ik_round_trip(arms):
    kin = arms["right"]
    rng = np.random.default_rng(0)
    solved = 0
    for _ in range(10):
        q_true = kin.lower + (0.15 + 0.7 * rng.random(7)) * (kin.upper - kin.lower)
        p, r = kin.fk(q_true)
        q, pe, re = kin.ik(p, r, q_true + 0.05 * rng.standard_normal(7))
        if pe < 1e-3 and re < 0.5:
            solved += 1
            p2, _ = kin.fk(q)
            assert np.linalg.norm(p2 - p) < 1e-3
    assert solved >= 8


def test_gripper_model_is_consistent():
    # q=0 is CLOSED; the gap grows linearly to 13.9 cm at |q| = pi/4 (measured in sim, docs M15)
    assert finger_q_for_gap("right", 0.0) == 0.0
    assert finger_q_for_gap("right", 1.0) == pytest.approx(-FINGER_Q_MAX)
    assert finger_q_for_gap("left", 0.05) == pytest.approx(0.05 / GAP_PER_RAD)
    assert tip_depth(0.0) > tip_depth(FINGER_Q_MAX)  # fingertips rise as the claws open


def test_down_rotation_points_fingers_down():
    for yaw in np.linspace(-np.pi, np.pi, 7):
        assert down_rotation(yaw)[2, 2] == pytest.approx(1.0)
