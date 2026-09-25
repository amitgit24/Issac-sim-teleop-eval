#!/usr/bin/env python3
"""Record a 360-degree camera orbit of the furnished room (presentation video).

Scene: full room preset, the robot at its ready pose, a soup can in the pick zone and N clutter
items on the table. The camera circles the table once. Output: artifacts/showcase/room_tour.mp4.
Usage: $PY tools/record_room_tour.py [--seconds 12] [--clutter 6] [--room full]
"""
import argparse
import math
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=12.0)
ap.add_argument("--clutter", type=int, default=6)
ap.add_argument("--room", default="full")
ap.add_argument("--seed", type=int, default=5)
args = ap.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes"))
import env_common as E  # noqa: E402

E.enable_headless_cameras()
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

import clutter as CL  # noqa: E402
import pick_task as T  # noqa: E402
from room import add_room  # noqa: E402

FPS = 30


def main():
    obj = T.OBJECTS["soup_can"]
    cfg = T.make_scene_cfg(obj)
    for cam in T.CAMERAS:
        setattr(cfg, cam, None)
    cfg.showcase = T.showcase_camera_cfg()
    add_room(cfg, args.room)
    rng = np.random.default_rng(args.seed)
    items = CL.choose_items(rng, args.clutter, exclude_names=("soup_can",))
    CL.add_clutter_to_cfg(cfg, items)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))
    scene = InteractiveScene(cfg)
    E.harden_finger_mimics()
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    arms = T.Arms(robot)
    T.reset_episode(sim, scene, arms, obj, rng)
    CL.place_clutter(scene, items, rng)
    for _ in range(40):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(1 / 60)
    cam = scene["showcase"]
    center = np.array([-0.15, -0.05, 0.80])
    frames = []
    n = int(args.seconds * FPS)
    for i in range(n):
        th = math.radians(-20.0 + 360.0 * i / n)
        r = 1.6 + 0.1 * math.sin(2 * th)  # gentle in/out; stays inside the side tables (y=+-1.99) and bookcases
        eye = center + np.array([r * math.cos(th), r * math.sin(th), 0.85 + 0.12 * math.cos(th)])
        cam.set_world_poses_from_view(eyes=torch.tensor([eye], dtype=torch.float32, device=cam.device),
                                      targets=torch.tensor([center], dtype=torch.float32, device=cam.device))
        for _ in range(2):  # 60 Hz physics, 30 fps video
            robot.set_joint_position_target(robot.data.default_joint_pos)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(1 / 60)
        frames.append(cam.data.output["rgb"][0, ..., :3].cpu().numpy())
    out = E.ARTIFACTS / "showcase" / "room_tour.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames[2:], fps=FPS, quality=9, macro_block_size=1)
    print(f"ROOM TOUR SAVED: {out} ({len(frames) - 2} frames)", flush=True)


if __name__ == "__main__":
    E.run_main(main, simulation_app)
