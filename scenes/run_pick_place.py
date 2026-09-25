#!/usr/bin/env python3
"""Pick-and-place episodes: right arm picks an object at the pick zone and places it in a container.

Plan: scenes/place_planner.py. The container is a fixed (kinematic) place target from the inventory,
at a spot found feasible by tools/search_place_spot.py. Recorded like run_pick.py (npz + 3 camera MP4s,
docs/DECISIONS.md D15). Success = after the final hold the object's center is inside the container's
footprint, below its rim + 2 cm, settled (moved < 1 cm during the last second), and >= 6 cm from the
right fingertips. (An instantaneous speed limit was noisy for a can lying in the ribbed bin.)

Usage:
  /data/isaac/isaacsim/bin/python scenes/run_pick_place.py --object soup_can --container klt_bin --episodes 5
  add --no-video for a fast check, --clutter N for N distractors, --room {none,walls,full}
"""

import argparse

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--object", default="soup_can")
parser.add_argument("--container", default="klt_bin")
parser.add_argument("--target-inset", type=float, default=0.26, help="container center, table inset (m)")
parser.add_argument("--target-y", type=float, default=-0.04, help="container center y (m)")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--clutter", type=int, default=0, help="number of clutter items on the table")
parser.add_argument("--room", default="none")
parser.add_argument("--run-name", default="")
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--keep-failed", action="store_true")
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
from kinematics import tip_depth  # noqa: E402
from pick_planner import grasp_direction  # noqa: E402
from place_planner import plan_pick_place  # noqa: E402
from room import add_room  # noqa: E402

SIDE = "right"
KEYS = ("joint_pos", "gripper", "ee_pose")


def main():
    obj = T.OBJECTS[args.object]
    box = INV.BY_NAME[args.container]
    if "place_target" not in box.roles or not box.verified:
        raise SystemExit(f"{args.container} is not a verified place target in the inventory")
    cfg = T.make_scene_cfg(obj)
    add_room(cfg, args.room)
    tx, ty = E.TABLE_NEAR_X + args.target_inset, args.target_y
    spawn = box.spawn_cfg()
    spawn.rigid_props.kinematic_enabled = True  # a fixed place target
    cfg.container = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Container", spawn=spawn,
                                   init_state=RigidObjectCfg.InitialStateCfg(pos=(tx, ty, E.TABLE_HEIGHT + box.rest_root_z + 0.001), rot=box.rest_rot))
    rng = np.random.default_rng(args.seed)
    clutter_items = CL.choose_items(rng, args.clutter, exclude_names=(args.container,)) if args.clutter else []
    CL.add_clutter_to_cfg(cfg, clutter_items)
    exclusions = CL.default_exclusions()[:1] + [CL.Zone(tx, ty, 0.5 * math.hypot(*box.size_m[:2]) + 0.05)]
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
    dt = sim.get_physics_dt()
    base = np.array(E.ROBOT_POS)
    table_z = E.TABLE_HEIGHT - base[2]
    rim_z = table_z + box.size_m[2]
    # Release as deep in the container as the arm can reach: releasing 3 cm below the rim of the 14.6 cm
    # KLT bin dropped the can ~11 cm and it tumbled, but floor level is out of reach at this spot. So try
    # depths from the floor (estimated 2 cm above the base) upward and use the deepest reachable one.
    h = box.size_m[2]
    depths = [d for d in (h - 0.035, 0.09, 0.07, 0.05, 0.03) if 0.0 < d <= h - 0.035] or [0.5 * h]
    run = args.run_name or f"place_{args.object}_in_{args.container}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = E.ARTIFACTS / "episodes" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for ep in range(args.episodes):
        info = T.reset_episode(sim, scene, arms, obj, rng)
        n_clutter = CL.place_clutter(scene, clutter_items, rng, exclusions) if clutter_items else 0
        for _ in range(20):  # let clutter settle
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
        for release_depth in depths:
            segs, why = plan_pick_place(kin, st[SIDE]["joint_pos"], st[SIDE]["gripper"], obj_p, grasp_direction(obj_q, obj.grasp_axis_local),
                                        table_z, obj.grasp, (tx - base[0], ty - base[1]), rim_z, release_depth, dt=dt,
                                        q_home=robot.data.default_joint_pos[0, arms.arm_ids[SIDE]].cpu().numpy())
            if segs is not None:
                break
        rec = {"episode": ep, **info, "clutter_placed": n_clutter, "release_depth": release_depth}
        if segs is None:
            rec.update(success=False, failure=why)
            print(f"EPISODE {ep}: NO PLAN ({why})", flush=True)
            summary.append(rec)
            continue
        target = robot.data.default_joint_pos.clone()
        left_hold = target[0, arms.arm_ids["left"]].cpu().numpy()
        left_grip = float(target[0, arms.finger_ids["left"][0]].item())
        frames = {c: [] for c in T.CAMERAS} if not args.no_video else None
        data = {f"{k}_{s}_{f}": [] for k in ("obs", "act") for s in ("right", "left") for f in KEYS}
        data["phase"] = []
        step = 0
        total_steps = sum(len(sg.arm_q) for sg in segs)
        pos_1s_before_end = None
        for seg_i, seg in enumerate(segs):
            for arm_q, f_q in zip(seg.arm_q, seg.finger_q):
                if step % T.RECORD_EVERY == 0:
                    st = T.measured_state(robot, arms)
                    for s in ("right", "left"):
                        for f in KEYS:
                            data[f"obs_{s}_{f}"].append(st[s][f])
                    for s, q, f in (("right", arm_q, f_q), ("left", left_hold, left_grip)):
                        data[f"act_{s}_joint_pos"].append(q)
                        data[f"act_{s}_gripper"].append(f)
                        data[f"act_{s}_ee_pose"].append(T.fk_pose(arms.kin[s], q))
                    data["phase"].append(seg_i)
                    if frames is not None:
                        for c in T.CAMERAS:
                            frames[c].append(scene[c].data.output["rgb"][0, ..., :3].cpu().numpy())
                target[0, arms.arm_ids[SIDE]] = torch.tensor(arm_q, dtype=target.dtype, device=target.device)
                target[0, arms.finger_ids[SIDE][0]] = float(f_q)
                robot.set_joint_position_target(target)
                scene.write_data_to_sim()
                sim.step(render=frames is not None and (step + 1) % T.RECORD_EVERY == 0)
                scene.update(dt)
                step += 1
                if step == total_steps - int(round(1.0 / dt)):
                    pos_1s_before_end = scene["object"].data.root_pos_w[0].cpu().numpy().copy()
                if not torch.isfinite(robot.data.joint_pos).all():
                    raise FloatingPointError(f"non-finite joint state, episode {ep} step {step}")
        # success: inside the container footprint, below its rim, settled, released
        o = scene["object"].data
        op = o.root_pos_w[0].cpu().numpy()
        speed = float(o.root_lin_vel_w[0].norm())
        moved_last_s = float(np.linalg.norm(op - pos_1s_before_end)) if pos_1s_before_end is not None else float("nan")
        inside = abs(op[0] - tx) < box.size_m[0] / 2 and abs(op[1] - ty) < box.size_m[1] / 2
        below_rim = op[2] < E.TABLE_HEIGHT + box.size_m[2] + 0.02
        e = robot.data.body_pos_w[0, arms.ee_idx[SIDE]].cpu().numpy()
        w, x, y, z = robot.data.body_quat_w[0, arms.ee_idx[SIDE]].tolist()
        from scipy.spatial.transform import Rotation

        tips = e - Rotation.from_quat([x, y, z, w]).as_matrix()[:, 2] * tip_depth(float(robot.data.joint_pos[0, arms.finger_ids[SIDE][0]]))
        released = float(np.linalg.norm(op - tips)) >= 0.06
        ok = bool(inside and below_rim and moved_last_s < 0.01 and released)
        rec.update(success=ok, inside=bool(inside), below_rim=bool(below_rim), speed=speed, moved_last_s=moved_last_s, released=released,
                   object_offset_from_container=[float(op[0] - tx), float(op[1] - ty)], phases=[s.name for s in segs])
        print(f"EPISODE {ep}: {'SUCCESS' if ok else 'FAIL'} inside={inside} below_rim={below_rim} moved_last_1s={moved_last_s * 100:.1f}cm released={released} "
              f"offset=({op[0] - tx:+.3f},{op[1] - ty:+.3f}) depth={release_depth:.3f} clutter={n_clutter}", flush=True)
        if ok or args.keep_failed:
            np.savez_compressed(out_dir / f"episode_{ep:04d}.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in data.items()},
                                success=ok, prompt=f"put the {obj.prompt.replace('pick up the ', '')} in the {args.container.replace('_', ' ')}",
                                phase_names=np.array([s.name for s in segs]))
            if frames is not None:
                for c in T.CAMERAS:
                    imageio.mimwrite(out_dir / f"episode_{ep:04d}_{c}.mp4", frames[c], fps=T.FPS_SIM // T.RECORD_EVERY, quality=8, macro_block_size=1)
            rec["saved"] = True
        summary.append(rec)
    n_ok = sum(r.get("success", False) for r in summary)
    (out_dir / "summary.json").write_text(json.dumps({"task": "pick_place", "object": args.object, "container": args.container,
                                                      "target": [tx, ty], "clutter": [it.name for it in clutter_items],
                                                      "episodes": summary, "success_rate": n_ok / max(1, len(summary))}, indent=2))
    print(f"PICK-PLACE RUN DONE: {n_ok}/{len(summary)} success -> {out_dir}", flush=True)


if __name__ == "__main__":
    E.run_main(main, simulation_app)
