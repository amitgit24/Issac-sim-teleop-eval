"""Position-based pick task: object catalog, scene (object + cameras), reset, success, observations.

Library module: import env_common (which sets the asset root) before this, after SimulationApp.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

import env_common as E
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, subtract_frame_transforms

from kinematics import ArmKinematics
from pick_planner import GraspSpec

FPS_SIM = 60  # physics / control rate
RECORD_EVERY = 2  # record and render every 2nd step -> 30 Hz data and video
CAM_W, CAM_H = 640, 480

# High friction with "max" combining, so the grip does not depend on the robot asset's own
# (unknown) finger friction.
GRIP_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.2, dynamic_friction=1.0, restitution=0.0, friction_combine_mode="max"
)


@dataclass
class ObjectSpec:
    name: str
    spawn: object  # an isaaclab spawner cfg
    rest_z: float  # object root height above the tabletop when resting
    grasp: GraspSpec
    prompt: str
    # object-local axis the fingers close along (rotated into the world at plan time)
    grasp_axis_local: tuple = (1.0, 0.0, 0.0)
    # orientation that makes the object stand upright (wxyz); the random yaw is applied on top
    base_rot: tuple = (1.0, 0.0, 0.0, 0.0)


# YCB "Axis_Aligned" assets are Y-up (lying on their side in Isaac's Z-up world, measured):
# a +90 deg turn about x stands them upright (local +y -> world +z, local +z -> world -y).
YCB_UPRIGHT = (0.70710678, 0.70710678, 0.0, 0.0)
YCB_DIR = f"{E.ASSET_ROOT}/Isaac/Props/YCB/Axis_Aligned_Physics"


def ycb_spawn(file: str, mass: float):
    return sim_utils.UsdFileCfg(
        usd_path=f"{YCB_DIR}/{file}",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=4),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
    )


OBJECTS = {
    "cube": ObjectSpec(
        name="cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=4),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            physics_material=GRIP_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.12, 0.12)),
        ),
        rest_z=0.025,
        grasp=GraspSpec(width=0.05, yaw_symmetry=math.pi / 2, height=0.05),
        prompt="pick up the red cube",
    ),
    # Tomato soup can: cylinder 6.8 cm diameter x 10.2 cm tall (measured). Any yaw works.
    "soup_can": ObjectSpec(
        name="soup_can",
        spawn=ycb_spawn("005_tomato_soup_can.usd", 0.35),
        rest_z=0.051,
        grasp=GraspSpec(width=0.068, yaw_symmetry=math.pi / 6, height=0.102),
        prompt="pick up the soup can",
        base_rot=YCB_UPRIGHT,
    ),
    # Mustard bottle: 9.6 x 5.8 x 19.1 cm (measured, local x, z, y). Grasp across the 5.8 cm side
    # (local z); the claws pinch the top 3.5 cm (cap/shoulder), the only part their pocket fits.
    "mustard": ObjectSpec(
        name="mustard",
        spawn=ycb_spawn("006_mustard_bottle.usd", 0.30),
        rest_z=0.0955,
        grasp=GraspSpec(width=0.058, yaw_symmetry=math.pi, height=0.191),
        prompt="pick up the mustard bottle",
        grasp_axis_local=(0.0, 0.0, 1.0),
        base_rot=YCB_UPRIGHT,
    ),
    # YCB mug: body 9.3 cm diameter (local z), 8.1 cm tall (local y), handle along local x (11.7 cm
    # overall). Visual-only upstream, so tools/make_physics_prop.py wraps it with rigid body + mass +
    # convex-decomposition collision. Fingers close across the body, perpendicular to the handle;
    # opening to 13.3 cm fits the gripper's 13.9 cm maximum.
    "mug": ObjectSpec(
        name="mug",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(E.PROJECT_ROOT / "assets" / "objects" / "ycb_mug_physics.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=4),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
        ),
        rest_z=0.0405,
        grasp=GraspSpec(width=0.093, yaw_symmetry=math.pi, height=0.081),
        prompt="pick up the mug",
        grasp_axis_local=(0.0, 0.0, 1.0),
        base_rot=YCB_UPRIGHT,
    ),
}


WRIST_CAM_POS = (-0.10, 0.0, -0.08)  # gripper frame; below the housing, outside it on the -x side
WRIST_CAM_AIM = (0.0, 0.0, -0.16)  # fingertip point the optical axis passes through


def wrist_camera_cfg(side: str) -> CameraCfg:
    """Camera fixed to the gripper body, looking at the space between the fingertips.

    Gripper frame: fingers extend along -z (tips 0.15-0.17 m down) and open along y; the oval
    housing fills the space above z=-0.07, so mounts above it only see the housing (verified by
    rendering candidates, docs/PROCESS_LOG.md step 4). ROS convention: the camera looks along its
    +z. q = Rx(180) (look along gripper -z) then a tilt about y toward the aim point.
    """
    px, _, pz = WRIST_CAM_POS
    tilt = math.atan2(px - WRIST_CAM_AIM[0], pz - WRIST_CAM_AIM[2])
    rot = (0.0, math.cos(tilt / 2), 0.0, -math.sin(tilt / 2))
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/openarm_{side}_ee_base_link/wrist_cam",
        update_period=0,
        height=CAM_H,
        width=CAM_W,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=10.0, clipping_range=(0.01, 5.0)),
        offset=CameraCfg.OffsetCfg(pos=WRIST_CAM_POS, rot=rot, convention="ros"),
    )


def make_scene_cfg(obj: ObjectSpec):
    @configclass
    class PickSceneCfg(E.RobotEnvSceneCfg):
        object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=obj.spawn,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(E.PICK_CENTER[0], E.PICK_CENTER[1], E.TABLE_HEIGHT + obj.rest_z), rot=obj.base_rot
            ),
        )
        cam_high = CameraCfg(
            prim_path="{ENV_REGEX_NS}/cam_high",
            update_period=0,
            height=CAM_H,
            width=CAM_W,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
            # set by set_high_camera() after reset
        )
        cam_right_wrist = wrist_camera_cfg("right")
        cam_left_wrist = wrist_camera_cfg("left")

    return PickSceneCfg(num_envs=1, env_spacing=4.0)


HIGH_CAM_EYE = (0.45, -0.95, 1.55)
HIGH_CAM_TARGET = (-0.25, -0.12, 0.80)
CAMERAS = ("cam_high", "cam_right_wrist", "cam_left_wrist")


def set_high_camera(scene):
    dev = scene["cam_high"].device
    scene["cam_high"].set_world_poses_from_view(
        eyes=torch.tensor([HIGH_CAM_EYE], device=dev), targets=torch.tensor([HIGH_CAM_TARGET], device=dev)
    )


class Arms:
    """Joint index bookkeeping + FK for both arms."""

    def __init__(self, robot):
        self.kin = {s: ArmKinematics(s) for s in ("left", "right")}
        self.arm_ids = {s: [robot.joint_names.index(n) for n in self.kin[s].joint_names] for s in self.kin}
        self.finger_ids = {s: robot.find_joints(f"openarm_{s}_finger_joint.*")[0] for s in self.kin}
        self.ee_idx = {s: robot.find_bodies(f"openarm_{s}_ee_base_link")[0][0] for s in self.kin}


def reset_episode(sim, scene, arms: Arms, obj: ObjectSpec, rng: np.random.Generator):
    """Robot to the ready pose (state AND targets, M17/openarm M5 bug 3), object at a jittered pose."""
    robot = scene["robot"]
    q = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    dx, dy = rng.uniform(-E.PICK_JITTER, E.PICK_JITTER, 2)
    yaw = rng.uniform(-math.pi, math.pi)
    root = scene["object"].data.default_root_state.clone()
    root[:, 0] = E.PICK_CENTER[0] + dx
    root[:, 1] = E.PICK_CENTER[1] + dy
    root[:, 2] = E.TABLE_HEIGHT + obj.rest_z
    yaw_t = torch.tensor([yaw], device=root.device)
    yaw_q = quat_from_euler_xyz(torch.zeros_like(yaw_t), torch.zeros_like(yaw_t), yaw_t)
    root[:, 3:7] = quat_mul(yaw_q, torch.tensor([obj.base_rot], device=root.device, dtype=root.dtype))
    root[:, 7:] = 0.0
    scene["object"].write_root_state_to_sim(root)
    scene.write_data_to_sim()
    # let the object settle onto the table with the robot holding still (not recorded)
    for _ in range(10):
        robot.set_joint_position_target(q)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())
    return {"object_xy_offset": [float(dx), float(dy)], "object_yaw": float(yaw)}


def object_pose_root(scene):
    """Object (position, quaternion wxyz) in the robot root frame, as numpy."""
    robot, o = scene["robot"], scene["object"]
    p, q = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, o.data.root_pos_w, o.data.root_quat_w)
    return p[0].cpu().numpy(), q[0].cpu().numpy()


def measured_state(robot, arms: Arms) -> dict:
    """Per arm: 7 joint positions, finger position, and ee pose (xyz + quat wxyz, root frame)."""
    out = {}
    jp = robot.data.joint_pos[0]
    for s in ("left", "right"):
        p, q = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w,
            robot.data.body_pos_w[:, arms.ee_idx[s]], robot.data.body_quat_w[:, arms.ee_idx[s]],
        )
        out[s] = {
            "joint_pos": jp[arms.arm_ids[s]].cpu().numpy(),
            "gripper": float(jp[arms.finger_ids[s][0]].item()),
            "ee_pose": np.concatenate([p[0].cpu().numpy(), q[0].cpu().numpy()]),
        }
    return out


def fk_pose(kin: ArmKinematics, q) -> np.ndarray:
    """xyz + quat wxyz of the commanded joint target (exact, validated FK)."""
    from scipy.spatial.transform import Rotation

    p, r = kin.fk(q)
    x, y, z, w = Rotation.from_matrix(r).as_quat()
    return np.array([*p, w, x, y, z])


def check_success(scene, arms: Arms, obj: ObjectSpec, side: str, min_lift: float = 0.08) -> dict:
    """Lifted AND held: object raised >= min_lift above its resting height, still between the fingers."""
    robot, o = scene["robot"], scene["object"]
    obj_p = o.data.root_pos_w[0].cpu().numpy()
    ee_p = robot.data.body_pos_w[0, arms.ee_idx[side]].cpu().numpy()
    lift = obj_p[2] - (E.TABLE_HEIGHT + obj.rest_z)
    lateral = float(np.linalg.norm(obj_p[:2] - ee_p[:2]))
    ok = bool(lift >= min_lift and lateral < 0.03)
    return {"success": ok, "lift": float(lift), "lateral_offset": lateral}
