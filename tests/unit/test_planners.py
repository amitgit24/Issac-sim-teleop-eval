import math

import numpy as np
import pytest

import layout as L
from handover_planner import HandoverPoint, plan_handover
from kinematics import FINGER_Q_MAX, ArmKinematics
from pick_planner import GraspSpec, grasp_direction, plan_pick
from place_planner import plan_pick_place

BASE = np.array(L.ROBOT_POS)
TABLE_Z = L.TABLE_HEIGHT - BASE[2]
CUBE = GraspSpec(width=0.05, yaw_symmetry=math.pi / 2, height=0.05)
CAN = GraspSpec(width=0.068, yaw_symmetry=math.pi / 6, height=0.102)
DT = 1.0 / 60.0


@pytest.fixture(scope="module")
def kin():
    return {s: ArmKinematics(s) for s in ("left", "right")}


def obj_at_pick_center(spec):
    return np.array([L.PICK_CENTER[0], L.PICK_CENTER[1], L.TABLE_HEIGHT + spec.height / 2]) - BASE


def assert_smooth(arm_q_rows, max_step=0.06):
    q = np.concatenate(arm_q_rows)
    assert np.max(np.abs(np.diff(q, axis=0))) < max_step  # rad per 1/60 s control step


def test_pick_plan_is_complete_and_smooth(kin):
    plan = plan_pick(kin["right"], np.array(L.READY_RIGHT_ARM), -FINGER_Q_MAX, obj_at_pick_center(CUBE),
                     np.array([0.0, 1.0]), TABLE_Z, CUBE, dt=DT)
    assert plan is not None
    assert [s.name for s in plan.segments] == ["approach", "descend", "close", "settle", "lift", "hold"]
    assert_smooth([s.arm_q for s in plan.segments])
    # the grasp puts the fingertips 3.5 cm below the object's top, clear of the table
    assert plan.segments[2].finger_q[-1] == 0.0  # force-limited close commands fully closed


def test_pick_descent_is_vertical(kin):
    k = kin["right"]
    plan = plan_pick(k, np.array(L.READY_RIGHT_ARM), -FINGER_Q_MAX, obj_at_pick_center(CAN), np.array([0.0, 1.0]), TABLE_Z, CAN, dt=DT)
    xy = np.array([k.fk(q)[0][:2] for q in plan.segments[1].arm_q])
    assert np.max(np.linalg.norm(xy - xy[0], axis=1)) < 2e-3  # no sideways swing into the object


def test_unreachable_object_gives_no_plan(kin):
    far = np.array([L.TABLE_NEAR_X + 0.85, 0.6, L.TABLE_HEIGHT + 0.025]) - BASE
    assert plan_pick(kin["right"], np.array(L.READY_RIGHT_ARM), -FINGER_Q_MAX, far, np.array([0.0, 1.0]), TABLE_Z, CUBE, dt=DT) is None


def test_grasp_direction_follows_object_yaw():
    yaw = 0.7
    q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    assert np.allclose(grasp_direction(q, (1.0, 0.0, 0.0)), [math.cos(yaw), math.sin(yaw)], atol=1e-9)


def test_pick_place_into_bin(kin):
    tx, ty = L.TABLE_NEAR_X + 0.26 - BASE[0], -0.04
    segs, why = plan_pick_place(kin["right"], np.array(L.READY_RIGHT_ARM), -FINGER_Q_MAX, obj_at_pick_center(CAN),
                                np.array([0.0, 1.0]), TABLE_Z, CAN, (tx, ty), TABLE_Z + 0.1464, 0.07, dt=DT)
    assert why == "ok"
    names = [s.name for s in segs]
    assert names[-6:] == ["lower", "release", "settle", "retreat", "return", "hold"]
    assert_smooth([s.arm_q for s in segs], max_step=0.08)


def test_handover_plan(kin):
    hp = HandoverPoint(x=L.TABLE_NEAR_X + 0.12 - BASE[0], y=-0.04, z=L.TABLE_HEIGHT + 0.36 - BASE[2])
    segs, why = plan_handover(kin["right"], kin["left"], np.array(L.READY_RIGHT_ARM), -FINGER_Q_MAX,
                              np.array(L.READY_LEFT_ARM), FINGER_Q_MAX, obj_at_pick_center(CAN), TABLE_Z, CAN, hp, dt=DT)
    assert why == "ok"
    names = [s[0] for s in segs]
    assert names.index("left_close") < names.index("right_release") < names.index("left_backoff")
