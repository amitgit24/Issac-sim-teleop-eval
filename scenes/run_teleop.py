#!/usr/bin/env python3
"""Keyboard / gamepad teleoperation of the bimanual OpenArm, with episode recording.

Opens the scene in the Isaac Sim GUI. Drive the active arm's gripper in the world frame (+x across
the table away from the robot, +y to the robot's left, +z up); key/button map in
scenes/teleop_input.py and docs/TELEOP.md. Recorded episodes use the same format as the scripted
demos (docs/DECISIONS.md D15), so tools/export_lerobot.py exports them unchanged.

Usage:
  /data/isaac/isaacsim/bin/python scenes/run_teleop.py                    # GUI, keyboard + gamepad
  /data/isaac/isaacsim/bin/python scenes/run_teleop.py --object mug --device gamepad
  /data/isaac/isaacsim/bin/python scenes/run_teleop.py --headless --script pick_cube   # automated check
"""

import argparse

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--object", default="cube", help="cube | soup_can | mustard | mug")
parser.add_argument("--device", default="both", choices=("keyboard", "gamepad", "both"))
parser.add_argument("--headless", action="store_true")
parser.add_argument("--script", default="", help="replay a built-in operator script instead of live input (pick_cube)")
parser.add_argument("--no-video", action="store_true", help="record state only (no cameras)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no-jitter", action="store_true", help="object exactly at the pick-zone center, yaw 0")
parser.add_argument("--run-name", default="")
parser.add_argument("--max-seconds", type=float, default=0.0, help="stop after this much sim time (0 = until closed)")
args = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

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
from kinematics import FINGER_Q_MAX  # noqa: E402
from teleop_controller import ArmState, TeleopController, Workspace  # noqa: E402
from teleop_input import GamepadSource, KeyboardSource, MultiSource, ScriptedSource, TeleopCommand  # noqa: E402

KEYS = ("joint_pos", "gripper", "ee_pose")
STATUS_PERIOD = 0.5  # s between terminal status lines

# Built-in operator scripts: (seconds, command). Used for automated checks and demos; they go
# through exactly the same controller / IK / recording path as live input.
SCRIPTS = {
    # Right arm picks the cube at the pick-zone center (use with --no-jitter): the ready pose is
    # 5 cm to the robot's left of the cube and 0.40 m above the table, gripper yaw 155 deg.
    "pick_cube": [
        (0.5, {"events": ["record"]}),
        (1.0 / 3.0, {"linear": [0, -1, 0]}),  # 5 cm right (-y)
        (1.418, {"angular": [0, 0, -1]}),  # yaw -65 deg to 90 (square to the cube; 0/180/270 are unreachable here)
        (0.3, {}),
        (1.43, {"linear": [0, 0, -1]}),  # down 21.4 cm: fingertips 1.5 cm above the table
        (0.4, {}),
        (0.05, {"events": ["gripper"]}),
        (1.2, {}),  # fingers close (force-limited)
        (0.8, {"linear": [0, 0, 1]}),  # lift 12 cm
        (1.0, {}),
        (0.05, {"events": ["save"]}),
        (0.3, {}),
    ],
}


def make_recorder():
    data = {f"{k}_{s}_{f}": [] for k in ("obs", "act") for s in ("right", "left") for f in KEYS}
    data["phase"] = []
    return data


def main():
    obj = T.OBJECTS[args.object]
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
    arms_idx = T.Arms(robot)
    dt = sim.get_physics_dt()
    base = np.array(E.ROBOT_POS)
    rng = np.random.default_rng(args.seed)

    # workspace: over the table, gripper frame between 5 and 60 cm above the tabletop (root frame)
    table_z = E.TABLE_HEIGHT - base[2]
    ws = Workspace(
        lo=np.array([E.TABLE_NEAR_X - 0.05, -0.70, table_z + 0.05]) - np.array([base[0], base[1], 0.0]),
        hi=np.array([E.TABLE_NEAR_X + E.TABLE_TOP_SIZE[0] - 0.05, 0.70, table_z + 0.60]) - np.array([base[0], base[1], 0.0]),
        table_z=table_z,
    )

    def reset(new_object_pose=True):
        if new_object_pose:
            T.reset_episode(sim, scene, arms_idx, obj, _ZeroYawRng() if args.no_jitter else rng)
        home = {s: np.array(E.READY_RIGHT_ARM if s == "right" else E.READY_LEFT_ARM, float) for s in ("right", "left")}
        arms = {s: ArmState(kin=arms_idx.kin[s], q=home[s].copy()) for s in home}
        for s, a in arms.items():
            a.finger = a.finger_goal = (1.0 if s == "left" else -1.0) * FINGER_Q_MAX
        return TeleopController(arms=arms, workspace=ws, active="right", home_q=home)

    ctrl = reset()
    if args.script:
        source = ScriptedSource(SCRIPTS[args.script], dt)
    else:
        srcs = []
        if args.device in ("keyboard", "both"):
            srcs.append(KeyboardSource())
        if args.device in ("gamepad", "both"):
            srcs.append(GamepadSource())
        source = MultiSource(srcs)
    gamepads = [s for s in getattr(source, "sources", []) if isinstance(s, GamepadSource)]

    run = args.run_name or f"teleop_{args.object}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = E.ARTIFACTS / "episodes" / run
    summary = {"task": "teleop", "object": args.object, "episodes": []}
    recording, data, frames, ep = False, None, None, 0
    step, last_status = 0, -1.0
    print(__doc__.split("Usage:")[0].strip(), flush=True)
    print("TELEOP READY: active arm = right. Hold keys to move; SPACE gripper, TAB switch arm, R/F/X record.", flush=True)

    target = robot.data.default_joint_pos.clone()
    while simulation_app.is_running():
        # a scripted operator waits during a reconfiguration, like a person would
        cmd = TeleopCommand() if (args.script and ctrl.reconfiguring) else source.poll()
        ev = cmd.events
        if "quit" in ev:
            break
        if "switch_arm" in ev:
            ctrl.switch_arm()
            print(f"ACTIVE ARM: {ctrl.active}", flush=True)
        if "precision" in ev:
            ctrl.precision = not ctrl.precision
            print(f"PRECISION MODE: {'on (30 %)' if ctrl.precision else 'off'}", flush=True)
        if "gripper" in ev:
            ctrl.toggle_gripper()
        if "home" in ev:
            ctrl.go_home()
        if "new_episode" in ev:
            if recording:
                print("NEW EPISODE: discarding the unsaved recording", flush=True)
                recording = False
            ctrl = reset()
            target = robot.data.default_joint_pos.clone()
            print("NEW EPISODE: object re-placed, arms at ready", flush=True)
        if "record" in ev and not recording:
            recording, data, frames = True, make_recorder(), ({c: [] for c in T.CAMERAS} if not args.no_video else None)
            print("RECORDING started", flush=True)
        if "discard" in ev and recording:
            recording = False
            print("RECORDING discarded", flush=True)
        for g in gamepads:
            g.recording = recording

        targets = ctrl.step(cmd.linear, cmd.angular, dt)
        if recording and step % T.RECORD_EVERY == 0:
            st = T.measured_state(robot, arms_idx)
            for s in ("right", "left"):
                for f in KEYS:
                    data[f"obs_{s}_{f}"].append(st[s][f])
                q, fq = targets[s]
                data[f"act_{s}_joint_pos"].append(q)
                data[f"act_{s}_gripper"].append(fq)
                data[f"act_{s}_ee_pose"].append(T.fk_pose(arms_idx.kin[s], q))
            data["phase"].append(0 if ctrl.active == "right" else 1)  # teleop: phase = active arm
            if frames is not None:
                for c in T.CAMERAS:
                    frames[c].append(scene[c].data.output["rgb"][0, ..., :3].cpu().numpy())
        for s, (q, fq) in targets.items():
            target[0, arms_idx.arm_ids[s]] = torch.tensor(q, dtype=target.dtype, device=target.device)
            target[0, arms_idx.finger_ids[s][0]] = fq
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        render = not args.headless or (frames is not None and recording and (step + 1) % T.RECORD_EVERY == 0)
        sim.step(render=render)
        scene.update(dt)
        step += 1

        if "save" in ev and recording:
            recording = False
            n = len(data["phase"])
            op = scene["object"].data.root_pos_w[0].cpu().numpy()
            lift = float(op[2] - (E.TABLE_HEIGHT + obj.rest_z))
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_dir / f"episode_{ep:04d}.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in data.items()},
                                success=lift >= 0.05, prompt=obj.prompt, phase_names=np.array(["right_arm_active", "left_arm_active"]))
            if frames is not None:
                for c in T.CAMERAS:
                    imageio.mimwrite(out_dir / f"episode_{ep:04d}_{c}.mp4", frames[c], fps=T.FPS_SIM // T.RECORD_EVERY,
                                     quality=8, macro_block_size=1)
            summary["episodes"].append({"episode": ep, "frames": n, "object_lift": lift, "success": lift >= 0.05})
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            print(f"RECORDING saved: episode {ep}, {n} frames, object lift {lift:.3f} m -> {out_dir}", flush=True)
            ep += 1

        t = step * dt
        if t - last_status >= STATUS_PERIOD:
            last_status = t
            a = ctrl.arms[ctrl.active]
            p = a.pos + base
            print(f"[{t:6.1f}s] arm={ctrl.active:5s} ee=({p[0]:+.3f},{p[1]:+.3f},{p[2] - E.TABLE_HEIGHT:.3f} above table) "
                  f"gripper={'closed' if abs(a.finger_goal) < 0.1 else 'open'} precision={'on' if ctrl.precision else 'off'} "
                  f"{'REC ' + str(len(data['phase'])) if recording else ''}{' LIMIT' if a.blocked else ''}"
                  f"{' RECONFIGURING' if a.reconfig else ''}", flush=True)
        if args.max_seconds and t >= args.max_seconds:
            break
        if not torch.isfinite(robot.data.joint_pos).all():
            raise FloatingPointError(f"non-finite joint state at step {step}")
    source.close()
    print(f"TELEOP DONE: {ep} episode(s) saved" + (f" -> {out_dir}" if ep else ""), flush=True)


class _ZeroYawRng:
    """rng stand-in for --no-jitter: zero xy offset and zero yaw."""

    def uniform(self, lo, hi, size=None):
        return np.zeros(size) if size is not None else 0.0


if __name__ == "__main__":
    E.run_main(main, simulation_app)
