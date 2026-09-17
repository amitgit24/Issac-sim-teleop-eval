"""Shared Robot_env definitions: asset paths, scene layout constants, scene config, and checks.

Pure library module. The importing script must create its isaacsim.SimulationApp BEFORE importing
this module (isaaclab imports require it), and must import this module BEFORE any other isaaclab
module (it sets the asset root that isaaclab reads at import time). It never creates an app.
"""

import os
import sys
import traceback
from pathlib import Path

import carb

# A plain SimulationApp (unlike Isaac Lab's AppLauncher) leaves the cloud asset root unset,
# which makes ISAAC_NUCLEUS_DIR resolve to "None/Isaac". Must be set before importing isaaclab.
ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
_settings = carb.settings.get_settings()
if not _settings.get("/persistent/isaac/asset_root/cloud"):
    _settings.set_string("/persistent/isaac/asset_root/cloud", ASSET_ROOT)

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from kinematics import FINGER_Q_MAX, TIP_DEPTH_CLOSED  # noqa: E402
from layout import (  # noqa: E402,F401  (re-exported for scripts)
    EDGE_GAP,
    PEDESTAL_SIZE,
    PICK_CENTER,
    PICK_JITTER,
    READY_ABOVE_TABLE,
    READY_INSET,
    READY_LEFT_ARM,
    READY_RIGHT_ARM,
    READY_Y,
    ROBOT_POS,
    TABLE_HEIGHT,
    TABLE_NEAR_X,
    TABLE_POS,
    TABLE_TOP_SIZE,
)

# Root of NVIDIA's non-Isaac asset packs (ArchVis furniture etc.), sibling of the Isaac root.
NVIDIA_ASSETS_DIR = ASSET_ROOT.rsplit("/Isaac/", 1)[0]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"

# Wooden dining table from the NVIDIA ArchVis furniture library (same asset server).
# Authored in centimeters; Isaac Lab's spawner converts it to meters on load, giving
# 0.91 x 1.53 x 0.76 m with its origin on the floor, so it rests on the ground at z=0.
# The asset is visual-only. Do NOT apply physics APIs to its meshes: editing prims inside
# the reference re-triggers the cm->m correction on sim.reset() and shrinks it another 100x.
# Collision comes from the invisible table_top_collider box below instead.
TABLE_USD = f"{NVIDIA_ASSETS_DIR}/ArchVis/Residential/Furniture/DiningSets/EastRural/EastRural_Table.usd"
TABLE_ROT = (1.0, 0.0, 0.0, 0.0)  # (w, x, y, z)
TABLE_PRIM = "/World/envs/env_0/Table"

# Bimanual OpenArm v2.0 (URDF + meshes vendored in assets/openarm/urdf and third_party/). Fixed base, shoulders at
# z=+0.698 above its base, arms at y=+-0.155, faces +x, ~0.5 m reach, arms hang down at zero pose.
# Converted by tools/convert_openarm_urdf.py with convex-DECOMPOSITION colliders and the finger
# mimic kept. The importer defaults (convex hulls, mimic dropped) fill in the claw fingers and
# break grasping (docs/MISTAKES.md M24, M26).
OPENARM_USD = PROJECT_ROOT / "assets" / "openarm" / "openarm_bimanual.usd"
# Ready pose (layout.py): grippers down and fully open. Finger q=0 is CLOSED on this gripper
# (measured, docs/MISTAKES.md M15); right opens with negative q, left with positive.
# N*m cap on each finger joint = grip force limit (~30 N at the tips). 10 N*m (~100 N) made PhysX let
# one finger push straight through a 0.35 kg can while the other held it (docs/MISTAKES.md M25).
FINGER_EFFORT = 3.0
READY_JOINT_POS = {
    **{f"openarm_left_joint{i + 1}": q for i, q in enumerate(READY_LEFT_ARM)},
    **{f"openarm_right_joint{i + 1}": q for i, q in enumerate(READY_RIGHT_ARM)},
    "openarm_left_finger_joint.*": FINGER_Q_MAX,
    "openarm_right_finger_joint.*": -FINGER_Q_MAX,
}
ROBOT_PRIM = "/World/envs/env_0/Robot"

# Viewer / snapshot camera: front-right 3/4 view of the robot across the table.
VIEW_EYE = (1.7, 1.6, 1.8)
VIEW_TARGET = (-0.25, 0.0, 0.85)

# Sphere light hanging above the scene center.
CENTER_LIGHT_POS = (0.0, 0.0, 2.5)


@configclass
class RobotEnvSceneCfg(InteractiveSceneCfg):
    """Ground plane, a center light, a library table, and OpenArm on a pedestal."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    center_light = AssetBaseCfg(
        prim_path="/World/CenterLight",
        spawn=sim_utils.SphereLightCfg(intensity=150000.0, radius=0.2, color=(1.0, 0.97, 0.92)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=CENTER_LIGHT_POS),
    )

    # Weak fill so surfaces facing away from the center light are not pitch black.
    fill_light = AssetBaseCfg(
        prim_path="/World/FillLight",
        spawn=sim_utils.DomeLightCfg(intensity=800.0, color=(0.9, 0.9, 0.95)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=TABLE_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS, rot=TABLE_ROT),
    )

    # Invisible static slab flush with the tabletop, so objects can rest on the table.
    table_top_collider = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableTopCollider",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_TOP_SIZE,
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TABLE_POS[0], TABLE_POS[1], TABLE_HEIGHT - TABLE_TOP_SIZE[2] / 2.0)
        ),
    )

    pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        spawn=sim_utils.CuboidCfg(
            size=PEDESTAL_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.28), roughness=0.6),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(ROBOT_POS[0], ROBOT_POS[1], PEDESTAL_SIZE[2] / 2.0)),
    )

    # Articulation settings from the earlier OpenArm prototype, except
    # gravity is disabled on the robot links (an ideally gravity-compensated arm, as in Isaac Lab's
    # Franka config). With gravity on, stiffness 100 lets the bent ready pose sag ~0.05 rad at
    # joints 1-4; the openarm project never saw this because its home pose hangs straight down.
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(OPENARM_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            # More solver iterations than openarm's 8/2: two fingers squeezing a light object is a
            # stiff contact problem, and too few iterations let a finger sink into the object (M25).
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_POS,
            joint_pos=READY_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arms": ImplicitActuatorCfg(
                joint_names_expr=["openarm_.*_joint[1-7]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=20.0,
                stiffness=100.0,
                damping=2.0,
            ),
            # Force-limited gripper, like a real one: the grasp commands the fingers fully closed
            # and the effort cap sets the squeeze, independent of where the claws touch the object
            # (docs/MISTAKES.md M22).
            # Only finger_joint1 is driven: finger_joint2 is a PhysX mimic of it (M26), and a drive
            # on joint2 would fight the mimic constraint.
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["openarm_.*_finger_joint1"],
                effort_limit_sim=FINGER_EFFORT,
                velocity_limit_sim=5.0,
                stiffness=100.0,
                damping=2.0,
            ),
        },
    )


def print_world_bounds(prim_path: str):
    """Print and return (min, max) of a prim's world-space bounding box."""
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"prim not found: {prim_path}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    lo, hi = box.GetMin(), box.GetMax()
    print(f"BOUNDS {prim_path}: min=({lo[0]:.3f}, {lo[1]:.3f}, {lo[2]:.3f}) max=({hi[0]:.3f}, {hi[1]:.3f}, {hi[2]:.3f})")
    return lo, hi


def check_ready_pose(robot, home) -> None:
    """Verify the robot settled in the ready pose: grippers at target, pointing down, clear of the table."""
    import torch

    max_err = torch.max(torch.abs(robot.data.joint_pos - home)).item()
    print(f"MAX JOINT ERROR FROM READY POSE: {max_err:.4f} rad")
    if max_err > 0.02:
        raise RuntimeError(f"robot did not hold the ready pose (max joint error {max_err:.4f} rad)")

    for side, sign in (("left", 1.0), ("right", -1.0)):
        idx = robot.find_bodies(f"openarm_{side}_ee_base_link")[0][0]
        pos = robot.data.body_pos_w[0, idx]
        w, x, y, z = robot.data.body_quat_w[0, idx].tolist()
        # World z-component of the gripper's z-axis (1.0 = fingers pointing straight down).
        up = 1.0 - 2.0 * (x * x + y * y)
        tilt = torch.rad2deg(torch.acos(torch.clamp(torch.tensor(up), -1.0, 1.0))).item()
        target = torch.tensor(
            [TABLE_NEAR_X + READY_INSET, sign * READY_Y, TABLE_HEIGHT + READY_ABOVE_TABLE],
            device=pos.device,
        )
        err = torch.linalg.norm(pos - target).item()
        px, py, pz = pos.tolist()
        # Physics body poses, not USD bounds: with GPU physics the USD transforms are not updated.
        tip_z = pz - TIP_DEPTH_CLOSED  # deepest the fingertips hang (measured), so a lower bound
        print(
            f"{side.upper()} GRIPPER: pos=({px:.3f}, {py:.3f}, {pz:.3f}) err={err * 1000:.1f}mm "
            f"tilt={tilt:.1f}deg fingertips >= {tip_z - TABLE_HEIGHT:.3f}m above table"
        )
        if err > 0.01 or tilt > 3.0 or tip_z < TABLE_HEIGHT + 0.05:
            raise RuntimeError(f"{side} gripper not at the expected ready pose")


def harden_finger_mimics() -> None:
    """Make each finger_joint2 -> finger_joint1 mimic a RIGID constraint, like the real linkage.

    The URDF importer creates the mimic as a soft spring (naturalFrequency 25, dampingRatio 0.005)
    that tracks in free motion but yields under contact: the object then pushes one claw open while
    the other closes (docs/MISTAKES.md M27). naturalFrequency <= 0 makes PhysX treat the mimic as
    a hard constraint. Call after InteractiveScene(...) and before sim.reset().
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for side in ("left", "right"):
        prim = stage.GetPrimAtPath(f"{ROBOT_PRIM}/joints/openarm_{side}_finger_joint2")
        attr = prim.GetAttribute("physxMimicJoint:rotX:naturalFrequency")
        if not attr.IsValid():
            raise RuntimeError(f"{prim.GetPath()} has no mimic joint; re-run tools/convert_openarm_urdf.py")
        attr.Set(0.0)
        prim.GetAttribute("physxMimicJoint:rotX:dampingRatio").Set(0.0)


def check_table_placement() -> None:
    """Fail if the table asset did not load at the expected size/placement (see docs/MISTAKES.md M8)."""
    lo, hi = print_world_bounds(TABLE_PRIM)
    if abs(lo[2]) > 0.01 or abs(hi[2] - TABLE_HEIGHT) > 0.01:
        raise RuntimeError(f"table not at expected placement: z range {lo[2]:.3f}..{hi[2]:.3f}, expected 0..{TABLE_HEIGHT}")


def enable_headless_cameras() -> None:
    """Required before SimulationContext for camera rendering with a plain SimulationApp."""
    settings = carb.settings.get_settings()
    settings.set_bool("/isaaclab/cameras_enabled", True)
    settings.set_bool("/isaaclab/render/offscreen", True)


def run_main(main, simulation_app) -> None:
    """Run main() and exit with a real status code (SimulationApp.close() always exits 0, M2)."""
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if exit_code:
            os._exit(exit_code)
        if simulation_app.is_running():
            simulation_app.close(skip_cleanup=True)
    os._exit(exit_code)
