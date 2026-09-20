"""Position-based pick planner: plans a whole pick as a joint-space trajectory BEFORE moving.

Pure numpy (uses kinematics.py). All poses are in the robot root frame.

Design (docs/DECISIONS.md D9): the openarm demonstrator re-solved differential IK online every step
and drifted. Here every waypoint is solved exactly offline, the descent and lift are dense
straight-line Cartesian paths (IK chained point to point, so the arm cannot swing sideways and
knock the object), and the arm simply tracks joint targets (gravity-free, error ~1e-4 rad).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from kinematics import FINGER_POCKET_DEPTH, ArmKinematics, down_rotation, finger_q_for_gap, tip_depth

MAX_JOINT_JUMP = 0.3  # rad between consecutive dense path points; more = IK branch flip
PATH_STEP = 0.01  # m between dense Cartesian points
# Extra IK seeds for the pre-grasp. The ready pose sits on an IK branch that cannot descend to low
# grasps continuously; this configuration (the earlier 0.30 m ready pose, right arm) can. Mirrored for
# the left arm. Random restarts cover the rest (docs/MISTAKES.md M30).
LOW_REACH_SEED_RIGHT = np.array([0.2435, 1.5520, -0.8246, 2.1654, -0.0702, 0.5342, 1.0278])
APPROACH_CHECK_POINTS = 25


@dataclass
class GraspSpec:
    """How to grasp one object type (sizes in meters)."""

    width: float  # object size across the fingers near its top (where the claws pinch)
    yaw_symmetry: float  # grasp yaw repeats every this many radians (cube: pi/2, bottle: pi, cylinder: small)
    height: float  # object height; the pre-grasp keeps the fingertips above its top
    open_margin: float = 0.04  # fingers open this much wider than the object before descending
    min_tip_height: float = 0.01  # never put the fingertips lower than this above the table

    @property
    def grasp_height(self) -> float:
        """Fingertip height above the table: the object's top sits FINGER_POCKET_DEPTH above the tips."""
        return max(self.min_tip_height, self.height - FINGER_POCKET_DEPTH)


@dataclass
class Segment:
    name: str
    arm_q: np.ndarray  # (N, 7) joint targets, one row per control step
    finger_q: np.ndarray  # (N,) finger joint target per control step


@dataclass
class PickPlan:
    yaw: float
    segments: list[Segment] = field(default_factory=list)
    grasp_pos: np.ndarray | None = None


def _min_jerk(n: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _joint_move(q0, q1, n):
    return q0 + (q1 - q0) * _min_jerk(n)[:, None]


def _cartesian_line(kin, p0, p1, rot, q_seed, n_steps):
    """Dense IK along a straight line, time-parametrized with min-jerk. None if not continuous."""
    n_pts = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / PATH_STEP)) + 1)
    qs, q = [], q_seed
    for s in np.linspace(0.0, 1.0, n_pts):
        q_new, pe, re = kin.ik(p0 + (p1 - p0) * s, rot, q, restarts=0)
        if pe > 1e-3 or re > 0.5 or np.max(np.abs(q_new - q)) > MAX_JOINT_JUMP:
            return None
        qs.append(q_new)
        q = q_new
    qs = np.array(qs)
    # resample the dense path onto n_steps control steps with min-jerk timing
    u = _min_jerk(n_steps) * (len(qs) - 1)
    i0 = np.floor(u).astype(int).clip(0, len(qs) - 2)
    w = (u - i0)[:, None]
    return qs[i0] * (1 - w) + qs[i0 + 1] * w


def _seeds(kin, q_start, n_random=4, rng_seed=0):
    low = LOW_REACH_SEED_RIGHT if kin.side == "right" else LOW_REACH_SEED_RIGHT * np.array([-1, -1, -1, 1, -1, -1, -1])
    rng = np.random.default_rng(rng_seed)
    span = kin.upper - kin.lower
    rand = [kin.lower + (0.1 + 0.8 * rng.random(7)) * span for _ in range(n_random)]
    return [np.asarray(q_start, float), low, *rand]


def _approach_clear(kin, q0, q1, min_tip_z_root, finger_q) -> bool:
    """Fingertips stay above min_tip_z_root along the joint-space approach (except the end point)."""
    depth = tip_depth(finger_q)
    for s in _min_jerk(APPROACH_CHECK_POINTS)[:-1]:
        p, r = kin.fk(q0 + (q1 - q0) * s)
        if p[2] - depth * r[2, 2] < min_tip_z_root:
            return False
    return True


def plan_pick(
    kin: ArmKinematics,
    q_start: np.ndarray,
    finger_start: float,
    obj_pos: np.ndarray,
    grasp_dir: np.ndarray,
    table_z: float,
    spec: GraspSpec,
    pre_height: float = 0.10,
    lift_height: float = 0.12,
    dt: float = 1.0 / 60.0,
    durations: dict | None = None,
) -> PickPlan | None:
    """Plan approach -> descend -> close -> lift -> hold for an object at obj_pos (root frame).

    grasp_dir: world-horizontal direction the fingers must close along (from the object's grasp axis).
    Tries every symmetric grasp yaw and keeps the feasible plan with the least joint travel.
    Returns None if no yaw gives a reachable, continuous plan.
    """
    d = {"approach": 1.5, "descend": 1.0, "close": 0.8, "settle": 0.3, "lift": 1.2, "hold": 1.0}
    d.update(durations or {})
    steps = {k: max(2, int(round(v / dt))) for k, v in d.items()}
    side = kin.side
    q_open = finger_q_for_gap(side, spec.width + spec.open_margin)
    # Command fully closed; the finger actuator's effort cap sets the grip force. (A target just
    # past the object's width barely squeezed: the claws touch above the tips, M22.)
    q_closed = 0.0
    grasp_ee_z = table_z + spec.grasp_height + tip_depth(q_open)
    grasp_pos = np.array([obj_pos[0], obj_pos[1], grasp_ee_z])
    # fingertips clear the object's top by 3 cm on the way in
    pre_height = max(pre_height, spec.height - spec.grasp_height + 0.03)
    pre_pos = grasp_pos + [0, 0, pre_height]
    lift_pos = grasp_pos + [0, 0, lift_height]

    # finger-opening axis is the gripper's y: (-sin yaw, cos yaw) must be parallel to grasp_dir
    yaw0 = float(np.arctan2(-grasp_dir[0], grasp_dir[1]))
    n_yaws = max(1, int(round(2 * np.pi / spec.yaw_symmetry)))
    best = None
    obj_top = table_z + spec.height
    for k in range(n_yaws):
        yaw = yaw0 + k * spec.yaw_symmetry
        rot = down_rotation(yaw)
        for seed in _seeds(kin, q_start):
            q_pre, pe, re = kin.ik(pre_pos, rot, seed, restarts=0)
            if pe > 1e-3 or re > 0.5:
                continue
            # the approach must not sweep the fingertips through the object (1 cm below its top at most)
            if not _approach_clear(kin, np.asarray(q_start, float), q_pre, obj_top - 0.01, q_open):
                continue
            descend = _cartesian_line(kin, pre_pos, grasp_pos, rot, q_pre, steps["descend"])
            if descend is None:
                continue
            lift = _cartesian_line(kin, grasp_pos, lift_pos, rot, descend[-1], steps["lift"])
            if lift is None:
                continue
            travel = np.sum(np.abs(q_pre - q_start))
            if best is None or travel < best[0]:
                best = (travel, yaw, q_pre, descend, lift)
    if best is None:
        return None
    _, yaw, q_pre, descend, lift = best
    q_grasp = descend[-1]
    plan = PickPlan(yaw=yaw, grasp_pos=grasp_pos)
    n = steps["approach"]
    plan.segments.append(Segment("approach", _joint_move(q_start, q_pre, n), finger_start + (q_open - finger_start) * _min_jerk(n)))
    plan.segments.append(Segment("descend", descend, np.full(len(descend), q_open)))
    n = steps["close"]
    f_close = q_open + (q_closed - q_open) * _min_jerk(n)
    plan.segments.append(Segment("close", np.repeat(q_grasp[None], n, 0), f_close))
    n = steps["settle"]
    plan.segments.append(Segment("settle", np.repeat(q_grasp[None], n, 0), np.full(n, q_closed)))
    plan.segments.append(Segment("lift", lift, np.full(len(lift), q_closed)))
    n = steps["hold"]
    plan.segments.append(Segment("hold", np.repeat(lift[-1][None], n, 0), np.full(n, q_closed)))
    return plan


def grasp_direction(obj_quat_wxyz, axis_local) -> np.ndarray:
    """World-horizontal unit direction of the object's grasp axis (root frame)."""
    w, x, y, z = obj_quat_wxyz
    d = Rotation.from_quat([x, y, z, w]).apply(np.asarray(axis_local, float))
    d = d[:2] / max(np.linalg.norm(d[:2]), 1e-9)
    return d
