#!/usr/bin/env python3
"""Pick -> handover -> place episodes: right picks, passes to left, and left drops into a bin.

Plan: scenes/transfer_planner.py. The container is a fixed (kinematic), verified place target from
the inventory. Success = after the final hold the object's center is inside the container footprint,
below its rim + 2 cm, settled (moved < 1 cm during the last second), and >= 6 cm from both hands'
fingertip points. Recorded like run_handover.py (D15 npz + 3 camera MP4s).

Usage:
  /data/isaac/isaacsim/bin/python scenes/run_handover_place.py --object soup_can --container klt_bin --episodes 3
  add --no-video for a fast check, --clutter N for N distractors, --room {none,walls,full}
"""

import argparse

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--object", default="soup_can", help="any upright pick_task object tall enough for the side-wrap")
parser.add_argument("--container", default="klt_bin", help="a verified inventory place_target")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--run-name", default="")
parser.add_argument("--showcase", action="store_true", help="also record a 1280x720 orbiting presentation video per episode")
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--keep-failed", action="store_true")
parser.add_argument("--trace", action="store_true")
parser.add_argument("--inset", type=float, default=0.12, help="handover point: object x as table inset (m)")
parser.add_argument("--y", type=float, default=-0.04, help="handover point: object y (m, root frame)")
parser.add_argument("--height", type=float, default=0.36, help="handover point: right gripper height above table (m)")
parser.add_argument("--bin-inset", type=float, default=0.40, help="bin center x as table inset (m)")
parser.add_argument("--bin-y", type=float, default=0.24, help="bin center y (m)")
parser.add_argument("--clutter", type=int, default=0, help="number of clutter items on the table")
parser.add_argument("--room", choices=("none", "walls", "full"), default="none")
args = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import env_common as E  # noqa: E402  (first isaaclab-related import: sets the asset root)

if not args.no_video:
    E.enable_headless_cameras()

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

import clutter as CL  # noqa: E402
import inventory as INV  # noqa: E402
import pick_task as T  # noqa: E402
from handover_planner import HandoverPoint  # noqa: E402
from kinematics import tip_depth  # noqa: E402
from pick_planner import grasp_direction  # noqa: E402
from room import add_room  # noqa: E402
from transfer_planner import plan_handover_place  # noqa: E402

KEYS = ("joint_pos", "gripper", "ee_pose")


def fingertips(robot, arms, side):
    """World position of a gripper's fingertip point (origin + tip depth along the finger direction)."""
    from scipy.spatial.transform import Rotation

    e = robot.data.body_pos_w[0, arms.ee_idx[side]].cpu().numpy()
    w, x, y, z = robot.data.body_quat_w[0, arms.ee_idx[side]].tolist()
    q = float(robot.data.joint_pos[0, arms.finger_ids[side][0]])
    return e - Rotation.from_quat([x, y, z, w]).as_matrix()[:, 2] * tip_depth(q)


def main():
    obj = T.OBJECTS[args.object]
    container = INV.BY_NAME[args.container]
    if "place_target" not in container.roles or not container.verified:
        raise SystemExit(f"{args.container} is not a verified place target in the inventory")

    cfg = T.make_scene_cfg(obj)
    add_room(cfg, args.room)
    bin_x, bin_y = E.TABLE_NEAR_X + args.bin_inset, args.bin_y
    spawn = container.spawn_cfg()
    spawn.rigid_props.kinematic_enabled = True
    cfg.container = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Container",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(bin_x, bin_y, E.TABLE_HEIGHT + container.rest_root_z + 0.001), rot=container.rest_rot
        ),
    )

    rng = np.random.default_rng(args.seed)
    clutter_items = CL.choose_items(rng, args.clutter, exclude_names=(args.container,)) if args.clutter else []
    CL.add_clutter_to_cfg(cfg, clutter_items)
    handover_x = E.TABLE_NEAR_X + args.inset
    exclusions = [
        CL.default_exclusions()[0],
        CL.Zone(handover_x, args.y, 0.16),
        CL.Zone(bin_x, bin_y, 0.5 * math.hypot(*container.size_m[:2]) + 0.05),
    ]
    if args.no_video:
        for cam in T.CAMERAS:
            setattr(cfg, cam, None)
    if args.showcase:
        cfg.showcase = T.showcase_camera_cfg()
        handover_target = np.array((handover_x, args.y, E.TABLE_HEIGHT + args.height - 0.20))
        bin_target = np.array((bin_x, bin_y, E.TABLE_HEIGHT + container.size_m[2] / 2.0))
        T.SHOWCASE_TARGET = tuple((handover_target + bin_target) / 2.0)
        T.SHOWCASE_RADIUS, T.SHOWCASE_HEIGHT = 1.25, E.TABLE_HEIGHT + args.height + 0.05

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
    table_z_root = E.TABLE_HEIGHT - base[2]
    bin_xy_root = (bin_x - base[0], bin_y - base[1])
    bin_rim_z_root = table_z_root + container.size_m[2]
    hp = HandoverPoint(x=handover_x - base[0], y=args.y - base[1], z=E.TABLE_HEIGHT + args.height - base[2])
    run = args.run_name or f"handover_place_{args.object}_in_{args.container}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = E.ARTIFACTS / "episodes" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for ep in range(args.episodes):
        info = T.reset_episode(sim, scene, arms, obj, rng)
        n_clutter = CL.place_clutter(scene, clutter_items, rng, exclusions) if clutter_items else 0
        for _ in range(20):
            robot.set_joint_position_target(robot.data.default_joint_pos)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
        if not args.no_video:
            for _ in range(3):
                sim.render()
            scene.update(dt)
        obj_p, obj_q = T.object_pose_root(scene)
        st = T.measured_state(robot, arms)
        segments, why = plan_handover_place(
            arms.kin["right"], arms.kin["left"], st["right"]["joint_pos"], st["right"]["gripper"],
            st["left"]["joint_pos"], st["left"]["gripper"], obj_p, table_z_root, obj.grasp, hp,
            bin_xy_root, bin_rim_z_root, dt=dt, grasp_dir=grasp_direction(obj_q, obj.grasp_axis_local),
            obj_quat=obj_q, handle_axis_local=obj.handle_axis_local,
        )
        rec = {"episode": ep, **info, "clutter_placed": n_clutter, "object_pos_root": obj_p.tolist()}
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
        total_steps = sum(len(seg[1]) for seg in segments)
        pos_1s_before_end = None
        show_frames, show_total = [], total_steps
        for seg_i, (name, qr, fr, ql, fl) in enumerate(segments):
            for i in range(len(qr)):
                if step % T.RECORD_EVERY == 0:
                    st = T.measured_state(robot, arms)
                    for side in ("right", "left"):
                        for field in KEYS:
                            data[f"obs_{side}_{field}"].append(st[side][field])
                    for side, q, f in (("right", qr[i], fr[i]), ("left", ql[i], fl[i])):
                        data[f"act_{side}_joint_pos"].append(q)
                        data[f"act_{side}_gripper"].append(f)
                        data[f"act_{side}_ee_pose"].append(T.fk_pose(arms.kin[side], q))
                    data["phase"].append(seg_i)
                    if frames is not None:
                        for camera in T.CAMERAS:
                            frames[camera].append(scene[camera].data.output["rgb"][0, ..., :3].cpu().numpy())
                    if args.showcase:
                        T.set_showcase_view(scene, step / max(1, show_total))
                        show_frames.append(scene["showcase"].data.output["rgb"][0, ..., :3].cpu().numpy())
                for side, q, f in (("right", qr[i], fr[i]), ("left", ql[i], fl[i])):
                    target[0, arms.arm_ids[side]] = torch.tensor(q, dtype=target.dtype, device=target.device)
                    target[0, arms.finger_ids[side]] = float(f)
                robot.set_joint_position_target(target)
                scene.write_data_to_sim()
                sim.step(render=frames is not None and (step + 1) % T.RECORD_EVERY == 0)
                scene.update(dt)
                step += 1
                if step == total_steps - int(round(1.0 / dt)):
                    pos_1s_before_end = scene["object"].data.root_pos_w[0].cpu().numpy().copy()
                if args.trace and step % 12 == 0:
                    op = scene["object"].data.root_pos_w[0].cpu().numpy()
                    fq = {side: robot.data.joint_pos[0, arms.finger_ids[side]].tolist() for side in ("right", "left")}
                    print(
                        f"TRACE ep{ep} {name:14s} step {step:4d}: "
                        f"obj=({op[0]:+.3f},{op[1]:+.3f},{op[2] - E.TABLE_HEIGHT:.3f}) "
                        f"obj-left_tips={np.linalg.norm(op - fingertips(robot, arms, 'left')):.3f} "
                        f"obj-right_tips={np.linalg.norm(op - fingertips(robot, arms, 'right')):.3f} "
                        f"right_f=({fq['right'][0]:+.2f},{fq['right'][1]:+.2f}) "
                        f"left_f=({fq['left'][0]:+.2f},{fq['left'][1]:+.2f})",
                        flush=True,
                    )
                if not torch.isfinite(robot.data.joint_pos).all():
                    raise FloatingPointError(f"non-finite joint state, episode {ep} step {step}")

        op = scene["object"].data.root_pos_w[0].cpu().numpy()
        dx, dy = float(op[0] - bin_x), float(op[1] - bin_y)
        moved_last_s = float(np.linalg.norm(op - pos_1s_before_end)) if pos_1s_before_end is not None else float("nan")
        inside = abs(dx) < container.size_m[0] / 2.0 and abs(dy) < container.size_m[1] / 2.0
        below_rim = op[2] < E.TABLE_HEIGHT + container.size_m[2] + 0.02
        left_dist = float(np.linalg.norm(op - fingertips(robot, arms, "left")))
        right_dist = float(np.linalg.norm(op - fingertips(robot, arms, "right")))
        left_clear, right_clear = left_dist >= 0.06, right_dist >= 0.06
        ok = bool(inside and below_rim and moved_last_s < 0.01 and left_clear and right_clear)
        rec.update(
            success=ok, inside=bool(inside), below_rim=bool(below_rim), moved_last_s=moved_last_s,
            left_clear=bool(left_clear), right_clear=bool(right_clear), obj_to_left_tips=left_dist,
            obj_to_right_tips=right_dist, object_offset_from_container=[dx, dy], steps=step,
            phases=[seg[0] for seg in segments],
        )
        print(
            f"EPISODE {ep}: {'SUCCESS' if ok else 'FAIL'} inside={inside} below_rim={below_rim} "
            f"moved_last_1s={moved_last_s * 100:.1f}cm left_clear={left_clear} right_clear={right_clear} "
            f"offset=({dx:+.3f},{dy:+.3f})",
            flush=True,
        )
        if ok or args.keep_failed:
            object_name = args.object.replace("_", " ")
            container_name = args.container.replace("_", " ")
            np.savez_compressed(
                out_dir / f"episode_{ep:04d}.npz",
                **{k: np.asarray(v, dtype=np.float32) for k, v in data.items()},
                success=ok,
                prompt=f"hand the {object_name} to the left hand and put it in the {container_name}",
                phase_names=np.array([seg[0] for seg in segments]),
            )
            if frames is not None:
                for camera in T.CAMERAS:
                    imageio.mimwrite(
                        out_dir / f"episode_{ep:04d}_{camera}.mp4", frames[camera],
                        fps=T.FPS_SIM // T.RECORD_EVERY, quality=8, macro_block_size=1,
                    )
            if args.showcase and show_frames:
                imageio.mimwrite(
                    out_dir / f"episode_{ep:04d}_showcase.mp4", show_frames,
                    fps=T.FPS_SIM // T.RECORD_EVERY, quality=9, macro_block_size=1,
                )
            rec["saved"] = True
        summary.append(rec)

    n_ok = sum(rec.get("success", False) for rec in summary)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "task": "handover_place", "object": args.object, "container": args.container,
                "seed": args.seed, "handover_point": vars(hp), "bin_center": [bin_x, bin_y],
                "clutter": [item.name for item in clutter_items], "episodes": summary,
                "success_rate": n_ok / max(1, len(summary)),
            },
            indent=2,
        )
    )
    print(f"HANDOVER-PLACE RUN DONE: {n_ok}/{len(summary)} success -> {out_dir}", flush=True)


if __name__ == "__main__":
    E.run_main(main, simulation_app)
