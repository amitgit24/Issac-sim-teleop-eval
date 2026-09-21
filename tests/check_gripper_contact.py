#!/usr/bin/env python3
"""Gripper contact test: close each hand on a FIXED 6 cm box placed between its fingertips.

Both fingers must stop at the box (joint well away from 0 = closed). A finger reaching ~0 has
passed through the box, i.e. its collision is not working (docs/MISTAKES.md M25).
Usage: /data/isaac/isaacsim/bin/python tests/check_gripper_contact.py [--usd PATH]
"""
import argparse
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--usd", default="")
args = ap.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes"))
import env_common as E  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from kinematics import TIP_DEPTH_OPEN  # noqa: E402

BOX = 0.06


def box_cfg(name, y):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, BOX, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, y, 0.0)),
    )


@configclass
class Cfg(E.RobotEnvSceneCfg):
    box_left = box_cfg("BoxLeft", 0.5)
    box_right = box_cfg("BoxRight", -0.5)


def main():
    cfg = Cfg(num_envs=1, env_spacing=4.0)
    if args.usd:
        cfg.robot.spawn.usd_path = args.usd
    print(f"ROBOT USD: {cfg.robot.spawn.usd_path}", flush=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device="cuda:0"))
    scene = InteractiveScene(cfg)
    E.harden_finger_mimics()
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    q = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    for _ in range(3):
        robot.set_joint_position_target(q); scene.write_data_to_sim(); sim.step(render=False); scene.update(1 / 60)
    ok = True
    for side in ("left", "right"):
        ee = robot.find_bodies(f"openarm_{side}_ee_base_link")[0][0]
        w, x, y, z = robot.data.body_quat_w[0, ee].tolist()
        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat([x, y, z, w]).as_matrix()
        # box 2 cm above the fingertips, long side along the finger-opening axis (gripper y)
        c = robot.data.body_pos_w[0, ee].cpu().numpy() + R @ np.array([0, 0, -TIP_DEPTH_OPEN + 0.02])
        qb = Rotation.from_matrix(R).as_quat()
        st = scene[f"box_{side}"].data.default_root_state.clone()
        st[0, :3] = torch.tensor(c)
        st[0, 3:7] = torch.tensor([qb[3], qb[0], qb[1], qb[2]])
        scene[f"box_{side}"].write_root_state_to_sim(st)
    fids = {s: robot.find_joints(f"openarm_{s}_finger_joint.*")[0] for s in ("left", "right")}
    names = {s: [robot.joint_names[i] for i in fids[s]] for s in fids}
    q_close = q.clone()
    for s in fids:
        q_close[0, fids[s]] = 0.0
    for _ in range(120):
        robot.set_joint_position_target(q_close); scene.write_data_to_sim(); sim.step(render=False); scene.update(1 / 60)
    open_gap_q = 0.06 / (0.1392 / 0.7854)  # finger |q| where a 6 cm gap would stop the tips
    for s in fids:
        vals = robot.data.joint_pos[0, fids[s]].tolist()
        for n, v in zip(names[s], vals):
            blocked = abs(v) > 0.15
            ok &= blocked
            print(f"FINGER {n}: {v:+.3f} rad -> {'STOPPED BY BOX' if blocked else 'PASSED THROUGH (no contact)'}", flush=True)
    print(f"(a finger stopping at the 6 cm box expects |q| around {open_gap_q:.2f} or more)")
    print("GRIPPER CONTACT CHECK " + ("PASSED" if ok else "FAILED"), flush=True)
    if not ok:
        raise RuntimeError("a finger passed through the fixed box")


if __name__ == "__main__":
    E.run_main(main, simulation_app)
