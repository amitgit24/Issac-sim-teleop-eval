#!/usr/bin/env python3
"""Robot_env scene: ground plane, center light, library table, and the bimanual OpenArm robot.

Checks the base scene: table placement and the robot holding its ready pose.

Usage:
    /data/isaac/isaacsim/bin/python scenes/robot_env_scene.py              # GUI, runs until closed
    /data/isaac/isaacsim/bin/python scenes/robot_env_scene.py --headless --steps 120 \
        --snapshot artifacts/scene_snapshot.png                             # headless check + image
"""

import argparse

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--headless", action="store_true", help="Run without the GUI.")
parser.add_argument("--steps", type=int, default=0, help="Number of sim steps to run (0 = until window closed).")
parser.add_argument("--snapshot", type=str, default="", help="Save an RGB image of the scene to this path.")
args = parser.parse_args()

from isaacsim import SimulationApp

# This must precede every carb, isaaclab, or Omniverse import.
simulation_app = SimulationApp({"headless": args.headless})

# env_common must be imported before any other isaaclab module: it sets the asset root,
# which isaaclab reads once at import time (docs/MISTAKES.md M1).
from env_common import (  # isort: skip
    OPENARM_USD,
    PROJECT_ROOT,
    ROBOT_POS,
    TABLE_PRIM,
    VIEW_EYE,
    VIEW_TARGET,
    RobotEnvSceneCfg,
    check_ready_pose,
    check_table_placement,
    enable_headless_cameras,
    harden_finger_mimics,
    print_world_bounds,
    run_main,
)

from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene


def main() -> None:
    if not OPENARM_USD.is_file():
        raise FileNotFoundError(
            f"OpenArm USD not found: {OPENARM_USD}. Run tools/convert_openarm_urdf.py first."
        )
    if args.snapshot:
        enable_headless_cameras()

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=VIEW_EYE, target=VIEW_TARGET)

    scene = InteractiveScene(RobotEnvSceneCfg(num_envs=1, env_spacing=4.0))
    harden_finger_mimics()

    camera = None
    if args.snapshot:
        from isaaclab.sensors import Camera, CameraCfg

        camera = Camera(
            CameraCfg(
                prim_path="/World/SnapshotCamera",
                width=1280,
                height=720,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 50.0)),
            )
        )

    sim.reset()
    scene.reset()
    robot = scene["robot"]
    # scene.reset() does not move the robot to its default pose (docs/MISTAKES.md M17).
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    check_table_placement()
    print_world_bounds(f"{TABLE_PRIM}/TableTop")

    if robot.num_joints <= 0:
        raise RuntimeError("OpenArm articulation initialized with no joints")
    print(f"OPENARM READY: {robot.num_joints} joints, base at {ROBOT_POS}")
    print("ROBOT_ENV SCENE READY: ground + center light + table + OpenArm")

    if camera is not None:
        camera.set_world_poses_from_view(
            eyes=torch.tensor([VIEW_EYE], device=sim.device),
            targets=torch.tensor([VIEW_TARGET], device=sim.device),
        )

    home = robot.data.default_joint_pos.clone()  # = READY_JOINT_POS
    step = 0
    while simulation_app.is_running():
        robot.set_joint_position_target(home)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_cfg.dt)
        step += 1
        if not torch.isfinite(robot.data.joint_pos).all().item():
            raise FloatingPointError(f"non-finite OpenArm joint position at step {step}")
        if args.steps and step >= args.steps:
            break

    check_ready_pose(robot, home)

    if camera is not None:
        from PIL import Image

        camera.update(sim_cfg.dt)
        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        out = Path(args.snapshot)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(out)
        print(f"SNAPSHOT SAVED: {out} shape={rgb.shape} mean={rgb.mean():.1f}")



if __name__ == "__main__":
    run_main(main, simulation_app)
