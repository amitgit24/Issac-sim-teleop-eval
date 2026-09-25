#!/usr/bin/env python3
"""Run scripted position-based pick episodes with the right arm; record data and videos.

Per episode, in artifacts/episodes/<run>/:
  episode_XXXX.npz   obs_* (measured) and act_* (commanded) for both arms at 30 Hz:
                     *_joint_pos (T,7), *_gripper (T,), *_ee_pose (T,7 = xyz + quat wxyz, robot root frame)
  episode_XXXX_<camera>.mp4   cam_high, cam_right_wrist, cam_left_wrist (30 fps)
  summary.json       per-episode success, lift, plan yaw, object pose, failure reason

Usage:
  /data/isaac/isaacsim/bin/python scenes/run_pick.py --object cube --episodes 5 [--seed 0] [--no-video]
"""

import argparse

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--object", default="cube")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run-name", default="")
parser.add_argument("--no-video", action="store_true", help="skip cameras (fast planner/physics check)")
parser.add_argument("--keep-failed", action="store_true", help="also keep data of failed episodes")
parser.add_argument("--trace", action="store_true", help="print object tilt/offset and fingertip height every 6 steps")
parser.add_argument("--room", choices=("none", "walls", "full"), default="none")
parser.add_argument("--clutter", type=int, default=0, help="number of inventory clutter items on the table")
args = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import env_common as E  # noqa: E402  (first isaaclab-related import: sets the asset root)
from room import add_room  # noqa: E402

if not args.no_video:
    E.enable_headless_cameras()

import json  # noqa: E402
import time  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

import clutter as CL  # noqa: E402
import pick_task as T  # noqa: E402
from kinematics import tip_depth  # noqa: E402
from pick_planner import grasp_direction, plan_pick  # noqa: E402

SIDE = "right"


def trace_tilt(q_wxyz, base_wxyz) -> float:
    """Angle (deg) between the object's current up-axis and world up (0 = standing as spawned)."""
    from scipy.spatial.transform import Rotation

    w, x, y, z = q_wxyz
    bw, bx, by, bz = base_wxyz
    up_local = Rotation.from_quat([bx, by, bz, bw]).inv().apply([0, 0, 1.0])  # local axis that starts up
    up_now = Rotation.from_quat([x, y, z, w]).apply(up_local)
    return float(np.degrees(np.arccos(np.clip(up_now[2], -1, 1))))


def main():
    obj = T.OBJECTS[args.object]
    cfg = T.make_scene_cfg(obj)
    add_room(cfg, args.room)
    clutter_rng = np.random.default_rng(args.seed + 1000)
    clutter_items = CL.choose_items(clutter_rng, args.clutter, exclude_names=(args.object,)) if args.clutter else []
    CL.add_clutter_to_cfg(cfg, clutter_items)
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
    kin = arms.kin[SIDE]
    run = args.run_name or f"{args.object}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = E.ARTIFACTS / "episodes" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    dt = sim.get_physics_dt()
    summary = []

    for ep in range(args.episodes):
        info = T.reset_episode(sim, scene, arms, obj, rng)
        if clutter_items:
            info["clutter_placed"] = CL.place_clutter(scene, clutter_items, clutter_rng)
            for _ in range(20):  # let the clutter settle before planning
                robot.set_joint_position_target(robot.data.default_joint_pos)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(dt)
        if not args.no_video:
            # RTX output lags the physics state by a frame or two: render a few times so the first
            # recorded frame shows this episode, not a blank or stale buffer.
            for _ in range(3):
                sim.render()
            scene.update(dt)
        obj_p, obj_q = T.object_pose_root(scene)
        st = T.measured_state(robot, arms)
        plan = plan_pick(
            kin, st[SIDE]["joint_pos"], st[SIDE]["gripper"], obj_p, grasp_direction(obj_q, obj.grasp_axis_local),
            E.TABLE_HEIGHT - E.ROBOT_POS[2], obj.grasp, dt=dt,
        )
        rec = {"episode": ep, **info, "object_pos_root": obj_p.tolist()}
        if plan is None:
            rec.update(success=False, failure="no feasible plan")
            print(f"EPISODE {ep}: NO PLAN (object at {np.round(obj_p, 3)})", flush=True)
            summary.append(rec)
            continue
        rec["plan_yaw_deg"] = float(np.degrees(plan.yaw))

        # hold targets for both arms; the left arm stays at its ready pose
        target = robot.data.default_joint_pos.clone()
        left_hold = target[0, arms.arm_ids["left"]].cpu().numpy()
        left_grip = float(target[0, arms.finger_ids["left"][0]].item())
        frames = {c: [] for c in T.CAMERAS} if not args.no_video else None
        data = {k: [] for k in (
            "obs_right_joint_pos", "obs_right_gripper", "obs_right_ee_pose",
            "obs_left_joint_pos", "obs_left_gripper", "obs_left_ee_pose",
            "act_right_joint_pos", "act_right_gripper", "act_right_ee_pose",
            "act_left_joint_pos", "act_left_gripper", "act_left_ee_pose", "phase")}
        phase_names = [s.name for s in plan.segments]
        max_track_err, step = 0.0, 0
        for seg_i, seg in enumerate(plan.segments):
            for arm_q, f_q in zip(seg.arm_q, seg.finger_q):
                record = step % T.RECORD_EVERY == 0
                if record:  # observation BEFORE applying this step's action
                    st = T.measured_state(robot, arms)
                    for s in ("right", "left"):
                        data[f"obs_{s}_joint_pos"].append(st[s]["joint_pos"])
                        data[f"obs_{s}_gripper"].append(st[s]["gripper"])
                        data[f"obs_{s}_ee_pose"].append(st[s]["ee_pose"])
                    data["act_right_joint_pos"].append(arm_q)
                    data["act_right_gripper"].append(f_q)
                    data["act_right_ee_pose"].append(T.fk_pose(kin, arm_q))
                    data["act_left_joint_pos"].append(left_hold)
                    data["act_left_gripper"].append(left_grip)
                    data["act_left_ee_pose"].append(T.fk_pose(arms.kin["left"], left_hold))
                    data["phase"].append(seg_i)
                    if frames is not None:
                        for c in T.CAMERAS:
                            frames[c].append(scene[c].data.output["rgb"][0, ..., :3].cpu().numpy())
                t = torch.tensor(arm_q, dtype=target.dtype, device=target.device)
                target[0, arms.arm_ids[SIDE]] = t
                target[0, arms.finger_ids[SIDE]] = float(f_q)
                robot.set_joint_position_target(target)
                scene.write_data_to_sim()
                sim.step(render=frames is not None and (step + 1) % T.RECORD_EVERY == 0)
                scene.update(dt)
                step += 1
                if seg.name in ("approach", "descend", "lift"):
                    err = torch.max(torch.abs(robot.data.joint_pos[0, arms.arm_ids[SIDE]] - t)).item()
                    max_track_err = max(max_track_err, err)
                if args.trace and step % 6 == 0:
                    o = scene["object"].data
                    w, x, y, z = o.root_quat_w[0].tolist()
                    tilt = trace_tilt((w, x, y, z), obj.base_rot)
                    ee = robot.data.body_pos_w[0, arms.ee_idx[SIDE]].cpu().numpy()
                    op = o.root_pos_w[0].cpu().numpy()
                    fq = robot.data.joint_pos[0, arms.finger_ids[SIDE]].tolist()
                    print(f"TRACE ep{ep} {seg.name:8s} step {step:4d}: tips_z={ee[2] - tip_depth(fq[0]) - E.TABLE_HEIGHT:+.3f} "
                          f"obj_z={op[2] - E.TABLE_HEIGHT:.3f} obj_tilt={tilt:5.1f}deg obj_xy_off=({op[0] - ee[0]:+.3f},{op[1] - ee[1]:+.3f}) "
                          f"fingers=({fq[0]:+.3f},{fq[1]:+.3f})", flush=True)
                if not torch.isfinite(robot.data.joint_pos).all():
                    raise FloatingPointError(f"non-finite joint state, episode {ep} step {step}")
            if seg.name == "settle":
                rec["finger_after_close"] = float(robot.data.joint_pos[0, arms.finger_ids[SIDE][0]].item())
        res = T.check_success(scene, arms, obj, SIDE)
        rec.update(res, steps=step, max_arm_tracking_err_rad=max_track_err, phases=phase_names)
        print(
            f"EPISODE {ep}: {'SUCCESS' if res['success'] else 'FAIL'} lift={res['lift']:.3f}m lateral={res['lateral_offset']:.3f}m "
            f"yaw={rec['plan_yaw_deg']:.0f} finger_after_close={rec.get('finger_after_close', float('nan')):.3f} "
            f"track_err={max_track_err:.4f}rad",
            flush=True,
        )
        if res["success"] or args.keep_failed:
            np.savez_compressed(out_dir / f"episode_{ep:04d}.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in data.items()},
                                success=res["success"], prompt=obj.prompt)
            if frames is not None:
                for c in T.CAMERAS:
                    imageio.mimwrite(out_dir / f"episode_{ep:04d}_{c}.mp4", frames[c], fps=T.FPS_SIM // T.RECORD_EVERY, quality=8, macro_block_size=1)
            rec["saved"] = True
        summary.append(rec)

    n_ok = sum(r.get("success", False) for r in summary)
    (out_dir / "summary.json").write_text(json.dumps({"object": args.object, "seed": args.seed, "episodes": summary,
                                                      "success_rate": n_ok / max(1, len(summary))}, indent=2))
    print(f"PICK RUN DONE: {n_ok}/{len(summary)} success -> {out_dir}", flush=True)


if __name__ == "__main__":
    E.run_main(main, simulation_app)
