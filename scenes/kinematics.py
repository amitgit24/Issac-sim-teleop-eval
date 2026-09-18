"""OpenArm arm kinematics from the URDF: forward kinematics + bounded numerical IK (numpy/scipy only).

Frames: everything is in the robot ROOT frame (openarm_body_link0 = the articulation root, which the
scene places at ROBOT_POS with identity rotation), for the gripper body `openarm_<side>_ee_base_link`.
No Isaac imports, so it works in any python with numpy + scipy and can be unit-checked offline.

Why offline IK instead of Isaac Lab's DifferentialIKController: the openarm project's online
differential IK drifted (~2 cm / 20 deg residual, joints pinned at limits). Solving each waypoint
exactly and tracking joint targets with a gravity-free arm is deterministic (docs/DECISIONS.md).
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# Bimanual OpenArm v2.0 URDF (generated from the vendored openarm_description, third_party/).
URDF_PATH = Path(__file__).resolve().parents[1] / "assets" / "openarm" / "urdf" / "openarm_bimanual.urdf"

# Measured in sim (docs/PROCESS_LOG.md step 4): finger q is 0 = CLOSED (tips touching). Opening grows
# ~linearly to 13.9 cm at |q| = 0.7854 along the gripper's local y-axis. Right arm opens with negative q,
# left arm with positive q. Fingertips sit 0.152 m (fully open) .. 0.171 m (closed) below ee_base_link.
# The claws only form a pocket ~4 cm deep: measured inner gap (fully open) is 13.8-14.5 cm up to 2 cm
# above the tips, 11.9 at 3 cm, 8.7 at 4 cm, 6.7 at 5 cm, then < 2.5 cm (knuckles; the housing
# bottom is 4.1 cm above the tips). An object must not reach higher than this above the fingertips,
# or the knuckles hit its top and push it over (docs/MISTAKES.md M23).
FINGER_POCKET_DEPTH = 0.035
FINGER_Q_MAX = 0.7854
GAP_PER_RAD = 0.1392 / 0.7854
TIP_DEPTH_OPEN = 0.152
TIP_DEPTH_CLOSED = 0.171


def finger_q_for_gap(side: str, gap: float) -> float:
    """Finger joint value giving a fingertip gap of `gap` meters (clamped to the joint range)."""
    q = float(np.clip(gap / GAP_PER_RAD, 0.0, FINGER_Q_MAX))
    return q if side == "left" else -q


def tip_depth(finger_q: float) -> float:
    """Fingertip depth below ee_base_link for a finger joint value (linear fit of the measurements)."""
    frac = abs(finger_q) / FINGER_Q_MAX
    return TIP_DEPTH_CLOSED + (TIP_DEPTH_OPEN - TIP_DEPTH_CLOSED) * frac


@dataclass
class _Joint:
    xyz: np.ndarray
    rot: np.ndarray  # fixed origin rotation
    axis: np.ndarray | None  # None for fixed joints


class ArmKinematics:
    """7-DOF chain root -> openarm_<side>_ee_base_link."""

    def __init__(self, side: str, urdf_path: Path = URDF_PATH):
        if side not in ("left", "right"):
            raise ValueError(side)
        self.side = side
        root = ET.parse(urdf_path).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], f"openarm_{side}_ee_base_link"
        while link in by_child and by_child[link].find("parent").get("link") != "world":
            chain.append(by_child[link])
            link = by_child[link].find("parent").get("link")
        chain.reverse()
        self.joints, self.joint_names, lower, upper = [], [], [], []
        for j in chain:
            o = j.find("origin")
            xyz = np.array([float(v) for v in o.get("xyz").split()])
            rot = Rotation.from_euler("xyz", [float(v) for v in o.get("rpy").split()]).as_matrix()
            axis = None
            if j.get("type") == "revolute":
                axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
                self.joint_names.append(j.get("name"))
                lower.append(float(j.find("limit").get("lower")))
                upper.append(float(j.find("limit").get("upper")))
            self.joints.append(_Joint(xyz, rot, axis))
        self.lower, self.upper = np.array(lower), np.array(upper)
        if len(self.joint_names) != 7:
            raise RuntimeError(f"expected 7 arm joints for {side}, got {self.joint_names}")

    def fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Gripper (position, 3x3 rotation) in the robot root frame."""
        pos, rot, k = np.zeros(3), np.eye(3), 0
        for j in self.joints:
            pos = pos + rot @ j.xyz
            rot = rot @ j.rot
            if j.axis is not None:
                rot = rot @ Rotation.from_rotvec(j.axis * q[k]).as_matrix()
                k += 1
        return pos, rot

    def ik(self, pos_target, rot_target, q_seed, margin: float = 0.03, restarts: int = 20, rng_seed: int = 0):
        """Joint values reaching (pos_target, rot_target), kept `margin` (fraction of range) inside limits.

        Starts from q_seed (keeps consecutive waypoints on the same IK branch); falls back to random
        restarts only if that fails. Returns (q, position error m, rotation error deg).
        """
        span = self.upper - self.lower
        lo, hi = self.lower + margin * span, self.upper - margin * span
        pos_target, rot_target = np.asarray(pos_target, float), np.asarray(rot_target, float)

        def residual(q):
            p, r = self.fk(q)
            rot_err = Rotation.from_matrix(rot_target.T @ r).as_rotvec()
            # 1 mm position ~ 0.5 deg orientation; small pull toward the seed avoids branch jumps.
            return np.concatenate([10.0 * (p - pos_target), 0.6 * rot_err, 0.01 * (q - q_seed)])

        def errors(q):
            p, r = self.fk(q)
            return np.linalg.norm(p - pos_target), np.degrees(np.linalg.norm(Rotation.from_matrix(rot_target.T @ r).as_rotvec()))

        rng = np.random.default_rng(rng_seed)
        best = None
        for attempt in range(restarts + 1):
            q0 = np.clip(q_seed, lo, hi) if attempt == 0 else lo + rng.random(7) * (hi - lo)
            q = least_squares(residual, q0, bounds=(lo, hi)).x
            pe, re = errors(q)
            if best is None or pe + re * 1e-3 < best[1] + best[2] * 1e-3:
                best = (q, pe, re)
            if pe < 1e-3 and re < 0.5:
                break
        return best


def down_rotation(yaw: float) -> np.ndarray:
    """Gripper rotation with fingers pointing straight down (gripper z-axis = world +z, as measured
    at the ready pose) and the finger-opening axis (gripper y) rotated by `yaw` about world z."""
    return Rotation.from_euler("z", yaw).as_matrix()
