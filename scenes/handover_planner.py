"""Right-to-left handover of an upright cylinder (soup can), planned offline in joint space. Pure numpy.

Design (docs/DECISIONS.md D19), after two failed variants (M31-M34):
  1. right picks the can by its top 3.5 cm (the proven pick) and carries it, still hanging upright,
     to the handover point;
  2. left comes in HORIZONTALLY from its own side (fingers pointing -y, claws closing along x) and
     wraps the can's lower body: fingertips go past the can's axis, so the claws cradle it and the
     center of mass is inside the grip. Right's claws hold only the top 3.5 cm; left's pads
     (+-2.9 cm tall) sit entirely below them;
  3. left closes (force-limited), right opens and lifts straight up;
  4. left backs away along +y and up, holding the can.
Frames: robot root frame (identity rotation), as everywhere in the planners.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from kinematics import FINGER_POCKET_DEPTH, down_rotation, finger_q_for_gap, tip_depth
from pick_planner import APPROACH_CHECK_POINTS, Segment, _cartesian_line, _joint_move, _min_jerk, _seeds, plan_pick

RIGHT_CLOSED_ON_CAN = -0.355  # measured finger value on the 6.8 cm can (tip depth ~0.164 m)
LEFT_TIPS_PAST_AXIS = 0.01  # left fingertips go this far past the can's axis (wrap, not pinch)
LEFT_BELOW_RIGHT = 0.01  # clearance between left's pads and the bottom of right's claws
PAD_HALF_WIDTH = 0.029  # claw pads span +-2.9 cm across the closing plane (measured)
# Claw rolls for the left gripper pointing -y (mirror of tools/map_right_pointing.py: horizontal
# pointing is reachable with the claws rolled 60-90 deg; 90 = claws closing along world x).
LEFT_ROLLS = (90.0, 60.0, -90.0, -60.0)
RIGHT_CAN_BOTTOM_CLEARANCE = 0.03  # can bottom stays this far above the table while carried


@dataclass
class HandoverPoint:
    """Handover location (root frame): can center x, y and the right gripper origin height z."""

    x: float
    y: float
    z: float


def left_rotation(roll_deg: float) -> np.ndarray:
    """Left gripper with its fingers pointing -y (toward the right arm), claws rolled about them."""
    return (Rotation.from_euler("x", -90, degrees=True) * Rotation.from_euler("z", roll_deg, degrees=True)).as_matrix()


def can_geometry(spec, hp):
    """(can center z, z of right's fingertips) when right holds the can with its origin at hp."""
    tips_z = hp.z - tip_depth(RIGHT_CLOSED_ON_CAN)
    top = tips_z + FINGER_POCKET_DEPTH
    return top - spec.height / 2.0, tips_z


def _joint_move_clear(kin, q0, q1, min_z):
    """Joint-space move whose gripper origin stays above min_z (root frame) at every sample."""
    for s in _min_jerk(APPROACH_CHECK_POINTS):
        if kin.fk(q0 + (q1 - q0) * s)[0][2] < min_z:
            return False
    return True


HANDLE_AWAY_MIN = 0.5  # handle must point at least this much toward -y (away from the incoming left hand)


def plan_handover(kin_r, kin_l, q_r0, f_r0, q_l0, f_l0, obj_pos, table_z, spec, hp, dt=1.0 / 60.0, grasp_dir=None,
                  obj_quat=None, handle_axis_local=None):
    """Full two-arm plan. Returns (segments, reason). Segment = (name, qr, fr, ql, fl) per control step."""
    sec = lambda t: max(2, int(round(t / dt)))  # noqa: E731

    # 1. right: the proven top pick
    # grasp_dir: the object's grasp axis (pick_planner.grasp_direction); any direction works for a cylinder
    gd = np.array([0.0, 1.0]) if grasp_dir is None else np.asarray(grasp_dir, float)
    pick = plan_pick(kin_r, q_r0, f_r0, obj_pos, gd, table_z, spec, dt=dt, durations={"hold": 0.2})
    if pick is None:
        return None, "right pick not reachable"
    right_segs = [s for s in pick.segments if s.name != "hold"]
    q_lift, f_closed = right_segs[-1].arm_q[-1], right_segs[-1].finger_q[-1]
    # Objects with a handle (mugs): the handle must not point into the left hand's approach (+y side)
    # or along its closing axis, or the claws hit it and push the object (docs/MISTAKES.md M47). Track
    # the handle in the right gripper's frame (fixed once grasped) and constrain the handover yaw.
    handle_g = None
    if handle_axis_local is not None and obj_quat is not None:
        w, x, y, z = obj_quat
        h_world = Rotation.from_quat([x, y, z, w]).apply(np.asarray(handle_axis_local, float))
        _, r_grasp = kin_r.fk(right_segs[1].arm_q[-1])  # end of the descent = grasp pose
        handle_g = r_grasp.T @ h_world

    # 2. right carries the upright can to the handover point (fingers stay down; yaw is free)
    ee_min_z = table_z + RIGHT_CAN_BOTTOM_CLEARANCE + spec.height - FINGER_POCKET_DEPTH + tip_depth(RIGHT_CLOSED_ON_CAN)
    if hp.z < ee_min_z:
        return None, "handover point too low for the hanging can"
    target = np.array([hp.x, hp.y, hp.z])
    q_hr, best = None, None
    for yaw in np.radians(np.arange(0, 360, 30)):
        for seed in [q_lift, *_seeds(kin_r, q_lift)[1:]]:
            if handle_g is not None and (down_rotation(yaw) @ handle_g)[1] > -HANDLE_AWAY_MIN:
                break  # this yaw would turn the handle toward the left hand
            q, pe, re = kin_r.ik(target, down_rotation(yaw), seed, restarts=0)
            if pe < 1e-3 and re < 0.5 and _joint_move_clear(kin_r, q_lift, q, ee_min_z - 0.01):
                travel = np.sum(np.abs(q - q_lift))
                if best is None or travel < best:
                    q_hr, best = q, travel
                break
    if q_hr is None:
        return None, "right handover pose not reachable" + (" with the handle turned away" if handle_g is not None else "")
    n = sec(2.5)
    right_segs.append(Segment("transfer", _joint_move(q_lift, q_hr, n), np.full(n, f_closed)))

    # 3. left: horizontal wrap of the can's lower body, approaching along -y
    can_z, right_tips_z = can_geometry(spec, hp)
    radius = spec.width / 2.0
    pad_center_z = min(can_z, right_tips_z - LEFT_BELOW_RIGHT - PAD_HALF_WIDTH)
    if pad_center_z - PAD_HALF_WIDTH < can_z - spec.height / 2.0 - 0.02:
        return None, "no room for the left pads below right's claws"
    q_open_l = finger_q_for_gap("left", spec.width + 0.04)
    tips = np.array([hp.x, hp.y - LEFT_TIPS_PAST_AXIS, pad_center_z])
    grasp_l = tips + [0.0, tip_depth(q_open_l), 0.0]
    pre_l = grasp_l + [0.0, radius + 0.06, 0.0]
    left = None
    for roll in LEFT_ROLLS:
        R = left_rotation(roll)
        for seed in _seeds(kin_l, q_l0):
            q_pre, pe, re = kin_l.ik(pre_l, R, seed, restarts=0)
            if pe > 1e-3 or re > 0.5:
                continue
            approach = _cartesian_line(kin_l, pre_l, grasp_l, R, q_pre, sec(1.2))
            if approach is None:
                continue
            travel = np.sum(np.abs(q_pre - np.asarray(q_l0, float)))
            if left is None or travel < left[0]:
                left = (travel, R, q_pre, approach)
    if left is None:
        return None, "left wrap not reachable"
    _, R_l, q_pre_l, approach_l = left
    q_grasp_l = approach_l[-1]

    segments = []
    q_l0 = np.asarray(q_l0, float)
    for s in right_segs:  # left waits at ready during the pick and carry
        n = len(s.arm_q)
        segments.append(("right_" + s.name, s.arm_q, s.finger_q, np.repeat(q_l0[None], n, 0), np.full(n, f_l0)))
    hold_r = lambda n: (np.repeat(q_hr[None], n, 0), np.full(n, f_closed))  # noqa: E731
    n = sec(2.0)
    segments.append(("left_to_pre", *hold_r(n), _joint_move(q_l0, q_pre_l, n), f_l0 + (q_open_l - f_l0) * _min_jerk(n)))
    segments.append(("left_approach", *hold_r(len(approach_l)), approach_l, np.full(len(approach_l), q_open_l)))
    n = sec(0.8)
    segments.append(("left_close", *hold_r(n), np.repeat(q_grasp_l[None], n, 0), q_open_l * (1 - _min_jerk(n))))
    n = sec(0.4)
    segments.append(("left_settle", *hold_r(n), np.repeat(q_grasp_l[None], n, 0), np.zeros(n)))

    # 4. right opens and lifts straight up out of the way
    q_r_open = finger_q_for_gap("right", spec.width + 0.04)
    n = sec(0.6)
    segments.append(("right_release", np.repeat(q_hr[None], n, 0), f_closed + (q_r_open - f_closed) * _min_jerk(n),
                     np.repeat(q_grasp_l[None], n, 0), np.zeros(n)))
    p_hr, r_hr = kin_r.fk(q_hr)
    up = _cartesian_line(kin_r, p_hr, p_hr + [0.0, 0.0, 0.10], r_hr, q_hr, sec(1.0))
    if up is None:
        return None, "right lift-off not continuous"
    segments.append(("right_liftoff", up, np.full(len(up), q_r_open), np.repeat(q_grasp_l[None], len(up), 0), np.zeros(len(up))))

    # 5. left backs away (+y) and up, holding the can
    p_gl, _ = kin_l.fk(q_grasp_l)
    back = _cartesian_line(kin_l, p_gl, p_gl + [0.0, 0.06, 0.04], R_l, q_grasp_l, sec(1.2))
    if back is None:
        return None, "left back-off not continuous"
    q_up = up[-1]
    segments.append(("left_backoff", np.repeat(q_up[None], len(back), 0), np.full(len(back), q_r_open), back, np.zeros(len(back))))
    n = sec(1.0)
    segments.append(("left_hold", np.repeat(q_up[None], n, 0), np.full(n, q_r_open), np.repeat(back[-1][None], n, 0), np.zeros(n)))
    return segments, "ok"
