"""Teleop controller: integrates operator twists into an end-effector target and solves IK. Pure numpy.

Per control step for the ACTIVE arm: target position += linear * speed * dt (world axes; the robot
root has identity rotation, so root == world axes), target rotation = Rdelta(world rpy) * target.
The target is clamped to a safe box (fingertips above the tabletop, within the table area) and IK is
solved from the previous joint solution. If IK cannot reach the target (> 5 mm / 2 deg), the arm
holds its last reachable pose and the target snaps back to it, so pushing into a limit never jumps.
The inactive arm holds its last joint targets.

Reconfiguration: the arm's reachable space is split across IK branches (joint6 is limited to
+-45 deg), so step-by-step IK can get stuck at a branch boundary (docs/MISTAKES.md M35). When a step
is blocked, the controller searches other IK solutions for the requested pose and, if the joint-space
transition to the closest one keeps the fingertips >= RECONFIG_TIP_CLEARANCE above the tabletop,
moves there smoothly (operator input paused, status shows RECONFIGURING). Otherwise it holds.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from kinematics import FINGER_Q_MAX, ArmKinematics, tip_depth
from pick_planner import _min_jerk, _seeds

MAX_LINEAR = 0.15  # m/s at full deflection
MAX_ANGULAR = 0.8  # rad/s at full deflection
PRECISION_SCALE = 0.3
GRIPPER_RATE = 1.5  # rad/s finger speed when opening/closing
IK_POS_TOL, IK_ROT_TOL = 5e-3, 2.0  # accept a step's IK within these (m, deg)
RECONFIG_JOINT_SPEED = 1.2  # rad/s for the largest joint during a reconfiguration
RECONFIG_MIN_TIME = 0.8  # s
RECONFIG_TIP_CLEARANCE = 0.03  # m above the tabletop along the whole transition
RECONFIG_CHECK_POINTS = 30


@dataclass
class ArmState:
    kin: ArmKinematics
    q: np.ndarray  # commanded joint targets
    pos: np.ndarray = None  # EE target (root frame)
    rot: np.ndarray = None
    finger: float = 0.0  # commanded finger joint
    finger_goal: float = 0.0
    blocked: bool = False  # last step hit reach/workspace limits
    reconfig: list = field(default_factory=list)  # pending joint targets of a reconfiguration

    def __post_init__(self):
        if self.pos is None:
            self.pos, self.rot = self.kin.fk(self.q)


@dataclass
class Workspace:
    """Safe box for the EE target (root frame) and the tabletop height the fingertips must clear."""

    lo: np.ndarray
    hi: np.ndarray
    table_z: float
    tip_clearance: float = 0.005


@dataclass
class TeleopController:
    arms: dict  # side -> ArmState
    workspace: Workspace
    active: str = "right"
    precision: bool = False
    home_q: dict = field(default_factory=dict)  # side -> ready joints

    def switch_arm(self):
        self.active = "left" if self.active == "right" else "right"

    def toggle_gripper(self, side=None):
        a = self.arms[side or self.active]
        sign = 1.0 if a.kin.side == "left" else -1.0
        # open <-> closed (0). Closing commands q=0; the force-limited actuator sets the squeeze.
        a.finger_goal = 0.0 if abs(a.finger_goal) > 0.1 else sign * FINGER_Q_MAX

    def go_home(self, side=None):
        side = side or self.active
        a = self.arms[side]
        a.q = np.array(self.home_q[side], float)
        a.pos, a.rot = a.kin.fk(a.q)

    def _clamp(self, a: ArmState, pos, rot):
        pos = np.clip(pos, self.workspace.lo, self.workspace.hi)
        # fingertip point (along the gripper -z axis) must stay above the tabletop
        tips_z = pos[2] - tip_depth(a.finger) * rot[2, 2]
        floor = self.workspace.table_z + self.workspace.tip_clearance
        if tips_z < floor:
            pos = pos + np.array([0.0, 0.0, floor - tips_z])
        return pos

    @property
    def reconfiguring(self) -> bool:
        return bool(self.arms[self.active].reconfig)

    def _plan_reconfig(self, a: ArmState, pos, rot, dt):
        """Closest other IK solution for (pos, rot) with a safe joint-space transition, or None."""
        best = None
        for seed in _seeds(a.kin, a.q, n_random=6)[1:]:
            q, pe, re = a.kin.ik(pos, rot, seed, restarts=0)
            if pe > 1e-3 or re > 0.5:
                continue
            dist = np.max(np.abs(q - a.q))
            if best is not None and dist >= best[0]:
                continue
            safe = True
            for s in np.linspace(0.0, 1.0, RECONFIG_CHECK_POINTS):
                p, r = a.kin.fk(a.q + (q - a.q) * s)
                if p[2] - tip_depth(a.finger) * r[2, 2] < self.workspace.table_z + RECONFIG_TIP_CLEARANCE:
                    safe = False
                    break
            if safe:
                best = (dist, q)
        if best is None:
            return None
        dist, q = best
        n = max(int(round(RECONFIG_MIN_TIME / dt)), int(np.ceil(dist / RECONFIG_JOINT_SPEED / dt)))
        return list(a.q + (q - a.q) * _min_jerk(n)[:, None])

    def step(self, linear, angular, dt):
        """Advance the active arm by one control step. Returns {side: (q, finger)} targets."""
        a = self.arms[self.active]
        if a.reconfig:  # finish a reconfiguration before taking new input
            a.q = a.reconfig.pop(0)
            if not a.reconfig:
                a.pos, a.rot = a.kin.fk(a.q)
            return self._finish_step(dt)
        scale = PRECISION_SCALE if self.precision else 1.0
        pos = a.pos + np.asarray(linear, float) * MAX_LINEAR * scale * dt
        dr = Rotation.from_rotvec(np.asarray(angular, float) * MAX_ANGULAR * scale * dt).as_matrix()
        rot = dr @ a.rot
        pos = self._clamp(a, pos, rot)
        moved = np.any(linear) or np.any(angular)
        a.blocked = False
        if moved:
            q, pe, re = a.kin.ik(pos, rot, a.q, restarts=0)
            if pe < IK_POS_TOL and re < IK_ROT_TOL:
                a.q, a.pos, a.rot = q, pos, rot
            else:  # branch boundary or truly unreachable
                path = self._plan_reconfig(a, pos, rot, dt)
                if path is None:
                    a.blocked = True  # hold the last reachable pose
                else:
                    a.reconfig = path
        return self._finish_step(dt)

    def _finish_step(self, dt):
        # gripper ramps toward its goal for every arm
        for arm in self.arms.values():
            d = arm.finger_goal - arm.finger
            arm.finger += np.clip(d, -GRIPPER_RATE * dt, GRIPPER_RATE * dt)
        return {s: (arm.q.copy(), float(arm.finger)) for s, arm in self.arms.items()}
