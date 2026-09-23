#!/usr/bin/env python3
"""Right-to-left handover episodes (soup can), recorded like run_pick.py (npz + 3 camera MP4s).

Plan: scenes/handover_planner.py (docs/DECISIONS.md D19). Success = after the final hold the can is
raised >= 5 cm above its resting height, within 6 cm of the LEFT fingertip point, and >= 8 cm from
the right gripper's fingertips.

Usage:
  /data/isaac/isaacsim/bin/python scenes/run_handover.py --episodes 3 [--seed 0] [--no-video] [--trace]
"""

import argparse

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run-name", default="")
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--keep-failed", action="store_true")
parser.add_argument("--trace", action="store_true")
parser.add_argument("--inset", type=float, default=0.12, help="handover point: can x as table inset (m)")
parser.add_argument("--y", type=float, default=-0.04, help="handover point: can y (m, root frame)")
parser.add_argument("--height", type=float, default=0.36, help="handover point: right gripper height above table (m)")
args = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import env_common as E  # noqa: E402  (first isaaclab-related import: sets the asset root)

if not args.no_video:
    E.enable_headless_cameras()

import json  # noqa: E402
import time  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

import pick_task as T  # noqa: E402
from handover_planner import HandoverPoint, plan_handover  # noqa: E402
from kinematics import tip_depth  # noqa: E402

OBJECT = "soup_can"
KEYS = ("joint_pos", "gripper", "ee_pose")


def fingertips(robot, arms, side):
    """World position of a gripper's fingertip point (origin + tip depth along the finger direction)."""
    from scipy.spatial.transform import Rotation

    e = robot.data.body_pos_w[0, arms.ee_idx[side]].cpu().numpy()
    w, x, y, z = robot.data.body_quat_w[0, arms.ee_idx[side]].tolist()
    q = float(robot.data.joint_pos[0, arms.finger_ids[side][0]])
    return e - Rotation.from_quat([x, y, z, w]).as_matrix()[:, 2] * tip_depth(q)


def main():
    obj = T.OBJECTS[OBJECT]
    cfg = T.make_scene_cfg(obj)
    if args.no_video:
        for cam in T.CAMERAS:
            setattr(cfg, cam, None)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / T.FPS_SIM, device="cuda:0"))
    sim.set_camera_view(eye=T.HIGH_CAM_EYE, target=T.HIGH_CAM_TARGET)
    scene = InteractiveScene(cfg)
    E.harden_finger_mimics()
    sim.reset()
    scene.reset()
    if not args.no_video:
        T.set_high_camera(scene)
    robot = scene["robot"]
    arms = T.Arms(robot)
    dt = sim.get_physics_dt()
    base = np.array(E.ROBOT_POS)
    hp = HandoverPoint(x=E.TABLE_NEAR_X + args.inset - base[0], y=args.y, z=E.TABLE_HEIGHT + args.height - base[2])
    run = args.run_name or f"handover_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = E.ARTIFACTS / "episodes" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary = []

    for ep in range(args.episodes):
        info = T.reset_episode(sim, scene, arms, obj, rng)
        if not args.no_video:
            for _ in range(3):
                sim.render()
            scene.update(dt)
        obj_p, _ = T.object_pose_root(scene)
        st = T.measured_state(robot, arms)
        segments, why = plan_handover(
            arms.kin["right"], arms.kin["left"], st["right"]["joint_pos"], st["right"]["gripper"],
            st["left"]["joint_pos"], st["left"]["gripper"], obj_p, E.TABLE_HEIGHT - base[2], obj.grasp, hp, dt=dt,
        )
        rec = {"episode": ep, **info, "object_pos_root": obj_p.tolist()}
        if segments is None:
            rec.update(success=False, failure=why)
            print(f"EPISODE {ep}: NO PLAN ({why})", flush=True)
            summary.append(rec)
            continue

        target = robot.data.default_joint_pos.clone()
        frames = {c: [] for c in T.CAMERAS} if not args.no_video else None
        data = {f"{k}_{s}_{f}": [] for k in ("obs", "act") for s in ("right", "left") for f in KEYS}
        data["phase"] = []
        step = 0
        for seg_i, (name, qr, fr, ql, fl) in enumerate(segments):
            for i in range(len(qr)):
                if step % T.RECORD_EVERY == 0:
                    st = T.measured_state(robot, arms)
                    for s in ("right", "left"):
                        for f in KEYS:
                            data[f"obs_{s}_{f}"].append(st[s][f])
                    for s, q, f in (("right", qr[i], fr[i]), ("left", ql[i], fl[i])):
                        data[f"act_{s}_joint_pos"].append(q)
                        data[f"act_{s}_gripper"].append(f)
                        data[f"act_{s}_ee_pose"].append(T.fk_pose(arms.kin[s], q))
                    data["phase"].append(seg_i)
                    if frames is not None:
                        for c in T.CAMERAS:
                            frames[c].append(scene[c].data.output["rgb"][0, ..., :3].cpu().numpy())
                for s, q, f in (("right", qr[i], fr[i]), ("left", ql[i], fl[i])):
                    target[0, arms.arm_ids[s]] = torch.tensor(q, dtype=target.dtype, device=target.device)
                    target[0, arms.finger_ids[s]] = float(f)
                robot.set_joint_position_target(target)
                scene.write_data_to_sim()
                sim.step(render=frames is not None and (step + 1) % T.RECORD_EVERY == 0)
                scene.update(dt)
                step += 1
                if args.trace and step % 12 == 0:
                    op = scene["object"].data.root_pos_w[0].cpu().numpy()
                    fq = {s: robot.data.joint_pos[0, arms.finger_ids[s]].tolist() for s in ("right", "left")}
                    print(f"TRACE ep{ep} {name:14s} step {step:4d}: obj=({op[0]:+.3f},{op[1]:+.3f},{op[2] - E.TABLE_HEIGHT:.3f}) "
                          f"obj-left_tips={np.linalg.norm(op - fingertips(robot, arms, 'left')):.3f} right_f=({fq['right'][0]:+.2f},{fq['right'][1]:+.2f}) "
                          f"left_f=({fq['left'][0]:+.2f},{fq['left'][1]:+.2f})", flush=True)
                if not torch.isfinite(robot.data.joint_pos).all():
                    raise FloatingPointError(f"non-finite joint state, episode {ep} step {step}")

        # success: the LEFT gripper holds the raised can, clear of the right gripper
        op = scene["object"].data.root_pos_w[0].cpu().numpy()
        lift = float(op[2] - (E.TABLE_HEIGHT + obj.rest_z))
        left_dist = float(np.linalg.norm(op - fingertips(robot, arms, "left")))
        right_dist = float(np.linalg.norm(op - fingertips(robot, arms, "right")))
        ok = lift >= 0.05 and left_dist < 0.06 and right_dist >= 0.08
        rec.update(success=bool(ok), lift=lift, obj_to_left_tips=left_dist, obj_to_right_tips=right_dist, steps=step,
                   phases=[s[0] for s in segments])
        print(f"EPISODE {ep}: {'SUCCESS' if ok else 'FAIL'} lift={lift:.3f}m obj-left_tips={left_dist:.3f}m obj-right_tips={right_dist:.3f}m", flush=True)
        if ok or args.keep_failed:
            np.savez_compressed(out_dir / f"episode_{ep:04d}.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in data.items()},
                                success=ok, prompt="hand the soup can from the right hand to the left hand",
                                phase_names=np.array([s[0] for s in segments]))
            if frames is not None:
                for c in T.CAMERAS:
                    imageio.mimwrite(out_dir / f"episode_{ep:04d}_{c}.mp4", frames[c], fps=T.FPS_SIM // T.RECORD_EVERY,
                                     quality=8, macro_block_size=1)
            rec["saved"] = True
        summary.append(rec)

    n_ok = sum(r.get("success", False) for r in summary)
    (out_dir / "summary.json").write_text(json.dumps({"task": "handover", "object": OBJECT, "seed": args.seed,
                                                      "handover_point": vars(hp), "episodes": summary,
                                                      "success_rate": n_ok / max(1, len(summary))}, indent=2))
    print(f"HANDOVER RUN DONE: {n_ok}/{len(summary)} success -> {out_dir}", flush=True)


if __name__ == "__main__":
    E.run_main(main, simulation_app)
