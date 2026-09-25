"""Pick -> handover -> place: right picks, hands the object to the left, left drops it into a bin.

Pure numpy (root frame). Built on handover_planner (proven right->left handover, D19); after the
left has backed off holding the object, the left arm:
  carry     joint-space move to above the bin, object bottom >= CARRY_CLEARANCE above the rim. The
            gripper may turn about the vertical axis on the way (the object stays upright): keeping
            the exact handover orientation (fingers pointing -y) is only reachable near the robot's
            center line, where the handover itself happens (docs/MISTAKES.md M45);
  lower     straight down until the object's bottom is RELEASE_ABOVE_RIM above the rim;
  release   open the claws: the object drops the last few cm into the bin. (The claws hold the
            object's lower body from the side, so lowering the object INTO the bin would drive the
            claws into the bin wall; releasing just above the rim avoids that.)
  back_out  straight back along the gripper's own axis (away from the object), then up;
  return    joint-space move to the ready pose, fingertips kept above the rim.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from handover_planner import can_geometry, plan_handover
from kinematics import finger_q_for_gap
from pick_planner import APPROACH_CHECK_POINTS, _cartesian_line, _joint_move, _min_jerk, _seeds

CARRY_CLEARANCE = 0.05  # object bottom above the rim while carried
RELEASE_ABOVE_RIM = 0.015  # object bottom above the rim at release (it drops the rest)
BACK_OUT = 0.10  # straight retreat along the gripper axis after release
YAW_TURNS_DEG = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0)  # wrist turns allowed while carrying
CARRY_SECONDS = 4.0


def plan_handover_place(kin_r, kin_l, q_r0, f_r0, q_l0, f_l0, obj_pos, table_z, spec, hp,
                        bin_xy, bin_rim_z, dt=1.0 / 60.0, grasp_dir=None, obj_quat=None, handle_axis_local=None):
    """Returns (segments, reason); segment = (name, qr, fr, ql, fl) per control step (root frame)."""
    sec = lambda t: max(2, int(round(t / dt)))  # noqa: E731
    segs, why = plan_handover(kin_r, kin_l, q_r0, f_r0, q_l0, f_l0, obj_pos, table_z, spec, hp, dt=dt, grasp_dir=grasp_dir,
                              obj_quat=obj_quat, handle_axis_local=handle_axis_local)
    if segs is None:
        return None, "handover: " + why
    segs = [s for s in segs if s[0] != "left_hold"]
    names = [s[0] for s in segs]
    q_grasp_l = segs[names.index("left_close")][3][-1]
    q_back = segs[-1][3][-1]
    q_r_park, f_r_open = segs[-1][1][-1], segs[-1][2][-1]
    # where the object sits relative to the left gripper (fixed while held; world axes, root frame)
    p_grasp, R_l = kin_l.fk(q_grasp_l)
    can_z, _ = can_geometry(spec, hp)
    offset = np.array([hp.x, hp.y, can_z]) - p_grasp  # object center minus gripper origin
    half_h = spec.height / 2.0
    p_back, _ = kin_l.fk(q_back)

    best = None
    for turn in YAW_TURNS_DEG:
        Rz = Rotation.from_euler("z", turn, degrees=True).as_matrix()
        R = Rz @ R_l
        off = Rz @ offset  # the held object turns with the gripper
        ee_release = np.array([bin_xy[0], bin_xy[1], bin_rim_z + RELEASE_ABOVE_RIM + half_h]) - off
        ee_above = ee_release + [0.0, 0.0, CARRY_CLEARANCE - RELEASE_ABOVE_RIM]
        carry_min_z = min(p_back[2], ee_above[2]) - 0.005  # the held object never dips below start/goal height
        back_dir = R[:, 2] * np.array([1.0, 1.0, 0.0])  # gripper +z = away from the fingertips, horizontal part
        back_dir = back_dir / max(np.linalg.norm(back_dir), 1e-9)
        for seed in [q_back, *_seeds(kin_l, q_back)[1:]]:
            q_above, pe, re = kin_l.ik(ee_above, R, seed, restarts=0)
            if pe > 1e-3 or re > 0.5:
                continue
            if not all(kin_l.fk(q_back + (q_above - q_back) * s)[0][2] >= carry_min_z for s in _min_jerk(APPROACH_CHECK_POINTS)):
                continue
            lower = _cartesian_line(kin_l, ee_above, ee_release, R, q_above, sec(0.8))
            if lower is None:
                continue
            back = _cartesian_line(kin_l, ee_release, ee_release + back_dir * BACK_OUT + [0.0, 0.0, 0.03], R, lower[-1], sec(1.0))
            if back is None:
                continue
            travel = np.sum(np.abs(q_above - q_back)) + 0.5 * abs(np.radians(turn))
            if best is None or travel < best[0]:
                best = (travel, q_above, lower, back)
            break
    if best is None:
        return None, "left place pose not reachable"
    _, q_above, lower, back = best
    q_home = np.asarray(q_l0, float)
    rim_clear = bin_rim_z + 0.03
    if not all(kin_l.fk(back[-1] + (q_home - back[-1]) * s)[0][2] >= rim_clear for s in _min_jerk(APPROACH_CHECK_POINTS)):
        return None, "left return dips toward the bin"

    park = lambda n: (np.repeat(q_r_park[None], n, 0), np.full(n, f_r_open))  # noqa: E731  right arm waits lifted
    f_hold = 0.0  # left claws commanded closed (force-limited) while holding
    f_open = finger_q_for_gap("left", spec.width + 0.04)
    # 4 s, not 2: a fast joint-space carry with a wrist turn shook the short mug out of the 3 N*m side
    # grip (M46); the gentler carry keeps the swing small.
    n = sec(CARRY_SECONDS)
    segs.append(("left_carry", *park(n), _joint_move(q_back, q_above, n), np.full(n, f_hold)))
    segs.append(("left_lower", *park(len(lower)), lower, np.full(len(lower), f_hold)))
    q_rel = lower[-1]
    n = sec(0.6)
    segs.append(("left_release", *park(n), np.repeat(q_rel[None], n, 0), f_hold + (f_open - f_hold) * _min_jerk(n)))
    n = sec(0.5)
    segs.append(("left_settle", *park(n), np.repeat(q_rel[None], n, 0), np.full(n, f_open)))
    segs.append(("left_back_out", *park(len(back)), back, np.full(len(back), f_open)))
    n = sec(2.0)
    segs.append(("left_return", *park(n), _joint_move(back[-1], q_home, n), np.full(n, f_open)))
    n = sec(2.5)  # let the object come to rest in the bin
    segs.append(("hold", *park(n), np.repeat(q_home[None], n, 0), np.full(n, f_open)))
    return segs, "ok"
