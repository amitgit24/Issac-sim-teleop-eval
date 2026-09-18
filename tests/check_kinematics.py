#!/usr/bin/env python3
"""Check scenes/kinematics.py against the simulator: FK on random configs, and IK round trips.

Usage: /data/isaac/isaacsim/bin/python tests/check_kinematics.py   (exit 1 on any failure)
"""
import sys
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes"))

import env_common as E  # noqa: E402  (first isaaclab-related import, sets the asset root)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

from kinematics import ArmKinematics, down_rotation  # noqa: E402

POS_TOL = 1e-3  # m
ROT_TOL = 0.5  # deg


def sim_ee_pose(robot, side):
    idx = robot.find_bodies(f"openarm_{side}_ee_base_link")[0][0]
    p, q = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, robot.data.body_pos_w[:, idx], robot.data.body_quat_w[:, idx]
    )
    w, x, y, z = q[0].tolist()
    return p[0].cpu().numpy(), Rotation.from_quat([x, y, z, w]).as_matrix()


def settle(sim, scene, robot, q_full, steps=3):
    robot.write_joint_state_to_sim(q_full, torch.zeros_like(q_full))
    for _ in range(steps):
        robot.set_joint_position_target(q_full)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device="cuda:0"))
    scene = InteractiveScene(E.RobotEnvSceneCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    rng = np.random.default_rng(0)
    worst = [0.0, 0.0]
    for side in ("left", "right"):
        kin = ArmKinematics(side)
        ids = [robot.joint_names.index(n) for n in kin.joint_names]
        # FK on random configurations
        for _ in range(20):
            q = kin.lower + rng.random(7) * (kin.upper - kin.lower)
            full = robot.data.default_joint_pos.clone()
            full[0, ids] = torch.tensor(q, dtype=full.dtype, device=full.device)
            settle(sim, scene, robot, full)
            ps, rs = sim_ee_pose(robot, side)
            pf, rf = kin.fk(robot.data.joint_pos[0, ids].cpu().numpy())
            pe = np.linalg.norm(ps - pf)
            re = np.degrees(np.linalg.norm(Rotation.from_matrix(rf.T @ rs).as_rotvec()))
            worst = [max(worst[0], pe), max(worst[1], re)]
        print(f"FK {side}: 20 random configs, worst pos err {worst[0] * 1000:.3f} mm, rot err {worst[1]:.3f} deg")
        if worst[0] > POS_TOL or worst[1] > ROT_TOL:
            raise RuntimeError(f"FK mismatch on {side} arm")
        # IK round trip: fingers-down poses over the table. Heights keep the fingertips (0.15-0.17 m
        # below the gripper frame) above the tabletop; yaws are offsets from the ready pose's wrist yaw.
        seed = robot.data.default_joint_pos[0, ids].cpu().numpy()
        _, r_ready = kin.fk(seed)
        ready_yaw = np.arctan2(-r_ready[0, 1], r_ready[1, 1])
        sign = 1.0 if side == "left" else -1.0
        reached = 0
        for dx, dy, dz, dyaw in ((0.20, 0.10, 0.20, 0.0), (0.30, 0.0, 0.18, np.pi / 6), (0.15, 0.25, 0.28, -np.pi / 6)):
            yaw = ready_yaw + sign * dyaw
            base = np.array(E.ROBOT_POS)
            target = np.array([E.TABLE_POS[0] - E.TABLE_TOP_SIZE[0] / 2 + dx, sign * dy, E.TABLE_HEIGHT + dz]) - base
            q, pe, re = kin.ik(target, down_rotation(yaw), seed)
            full = robot.data.default_joint_pos.clone()
            full[0, ids] = torch.tensor(q, dtype=full.dtype, device=full.device)
            settle(sim, scene, robot, full, steps=30)
            ps, rs = sim_ee_pose(robot, side)
            # The sim must land where the solver says it does (reachability is mapped separately).
            pf, rf = kin.fk(q)
            spe = np.linalg.norm(ps - pf)
            sre = np.degrees(np.linalg.norm(Rotation.from_matrix(rf.T @ rs).as_rotvec()))
            print(
                f"IK {side} target {np.round(target, 3)} yaw {np.degrees(yaw):+.0f} (ready {np.degrees(ready_yaw):+.0f}): "
                f"solver-to-target {pe * 1000:.2f} mm/{re:.2f} deg ({'reached' if pe < 1e-3 and re < 0.5 else 'UNREACHABLE'}), "
                f"sim-to-solver {spe * 1000:.2f} mm/{sre:.2f} deg"
            )
            if spe > 2e-3 or sre > 1.0:
                raise RuntimeError(f"sim did not reach the IK solution on {side} arm")
            reached += pe < 1e-3 and re < 0.5
        if reached == 0:
            raise RuntimeError(f"no IK test target reachable on {side} arm")
    print("KINEMATICS CHECK PASSED")


if __name__ == "__main__":
    E.run_main(main, simulation_app)
