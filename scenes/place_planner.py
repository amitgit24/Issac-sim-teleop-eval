"""Pick-and-place planner: the proven top pick, then carry the object over a container and release it.

Pure numpy (root frame), built on pick_planner. After the pick's lift:
  transfer  joint-space move to above the target, the hanging object's bottom kept >= CLEARANCE above
            the container rim (and the tabletop) the whole way;
  lower     straight descent until the object's bottom is `release_depth` below the rim (bins/bowls)
            or just above the surface (flat trays);
  release   open the fingers; settle;
  retreat   straight up 4 cm (clear of the released object), then a joint-space move back to the ready
            pose that keeps the gripper above the carry height; hold.
Geometry: the claws pinch the object's top 3.5 cm, so the object's bottom hangs
(height - FINGER_POCKET_DEPTH) below the fingertips.
"""

import numpy as np

from kinematics import FINGER_POCKET_DEPTH, down_rotation, finger_q_for_gap, tip_depth
from pick_planner import APPROACH_CHECK_POINTS, Segment, _cartesian_line, _joint_move, _min_jerk, _seeds, plan_pick

CLEARANCE = 0.04  # object bottom above the rim / tabletop while carried
RIGHT_CLOSED_DEPTH_Q = -0.30  # typical closed finger value on 5-9 cm objects, for tip depth


def object_bottom_below_ee(spec, finger_q=RIGHT_CLOSED_DEPTH_Q) -> float:
    """Vertical distance from the gripper origin down to the held object's bottom."""
    return tip_depth(finger_q) + spec.height - FINGER_POCKET_DEPTH


def plan_pick_place(kin, q_start, finger_start, obj_pos, grasp_dir, table_z, spec,
                    target_xy, target_rim_z, release_depth, dt=1.0 / 60.0, q_home=None):
    """Returns (PickPlan-like segments list, reason). target_* in the root frame; rim z = container top."""
    sec = lambda t: max(2, int(round(t / dt)))  # noqa: E731
    pick = plan_pick(kin, q_start, finger_start, obj_pos, grasp_dir, table_z, spec, dt=dt, durations={"hold": 0.2})
    if pick is None:
        return None, "pick not reachable"
    segs = [s for s in pick.segments if s.name != "hold"]
    q_lift, f_closed = segs[-1].arm_q[-1], segs[-1].finger_q[-1]
    hang = object_bottom_below_ee(spec)
    carry_z = max(target_rim_z, table_z) + CLEARANCE + hang  # gripper height while carrying over the rim
    release_z = target_rim_z - release_depth + hang
    p_lift, r_lift = kin.fk(q_lift)
    yaw_lift = float(np.arctan2(r_lift[1, 0], r_lift[0, 0]))

    best = None
    for dyaw in np.radians((0, 30, -30, 60, -60, 90, -90, 180)):
        rot = down_rotation(yaw_lift + dyaw)
        above = np.array([target_xy[0], target_xy[1], max(carry_z, p_lift[2])])
        for seed in [q_lift, *_seeds(kin, q_lift)[1:]]:
            q_above, pe, re = kin.ik(above, rot, seed, restarts=0)
            if pe > 1e-3 or re > 0.5:
                continue
            # the carried object's bottom must stay above the rim/tabletop along the joint move
            ok = all(kin.fk(q_lift + (q_above - q_lift) * s)[0][2] - hang >= min(carry_z, p_lift[2]) - hang - 0.005
                     for s in _min_jerk(APPROACH_CHECK_POINTS))
            if not ok:
                continue
            lower = _cartesian_line(kin, above, np.array([target_xy[0], target_xy[1], release_z]), rot, q_above, sec(1.0))
            if lower is None:
                continue
            travel = np.sum(np.abs(q_above - q_lift))
            if best is None or travel < best[0]:
                best = (travel, q_above, lower, rot)
            break
    if best is None:
        return None, "place pose not reachable"
    _, q_above, lower, rot = best
    n = sec(2.0)
    segs.append(Segment("transfer", _joint_move(q_lift, q_above, n), np.full(n, f_closed)))
    segs.append(Segment("lower", lower, np.full(len(lower), f_closed)))
    q_rel = lower[-1]
    q_open = finger_q_for_gap(kin.side, spec.width + 0.04)
    n = sec(0.6)
    segs.append(Segment("release", np.repeat(q_rel[None], n, 0), f_closed + (q_open - f_closed) * _min_jerk(n)))
    n = sec(0.4)
    segs.append(Segment("settle", np.repeat(q_rel[None], n, 0), np.full(n, q_open)))
    p_rel, _ = kin.fk(q_rel)
    up = _cartesian_line(kin, p_rel, p_rel + [0.0, 0.0, 0.04], rot, q_rel, sec(0.5))
    if up is None:
        return None, "retreat not continuous"
    segs.append(Segment("retreat", up, np.full(len(up), q_open)))
    q_home = np.asarray(q_start if q_home is None else q_home, float)
    min_z = p_rel[2] + 0.03
    if not all(kin.fk(up[-1] + (q_home - up[-1]) * s)[0][2] >= min_z for s in _min_jerk(APPROACH_CHECK_POINTS)[1:]):
        return None, "return-to-ready dips toward the placed object"
    n = sec(2.0)
    segs.append(Segment("return", _joint_move(up[-1], q_home, n), np.full(n, q_open)))
    n = sec(2.5)  # a can dropped on its side needs ~2 s to stop rolling in the bin
    segs.append(Segment("hold", np.repeat(q_home[None], n, 0), np.full(n, q_open)))
    return segs, "ok"
