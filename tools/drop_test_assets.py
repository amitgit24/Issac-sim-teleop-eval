#!/usr/bin/env python3
"""Drop test for every inventory item: spawn in its resting orientation 3 cm above the floor, simulate
3 s, then check it (1) kept its size after sim.reset() (unit-correction bug M8), (2) settled
(speed < 1 cm/s), (3) stayed in its resting orientation (tilt < 10 deg), (4) rests on the floor.
Writes assets/inventory_verified.json (loaded by scenes/inventory.py) and an overview image.
CPU physics with USD updates on, so bounds are read after settling (M11).
Usage: $PY tools/drop_test_assets.py [--only name1,name2]
"""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
args = ap.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes"))
import env_common as E  # noqa: E402

E.enable_headless_cameras()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import inventory as INV  # noqa: E402

SPACING, DROP = 0.45, 0.03


@configclass
class DropSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500.0))


def main():
    items = [it for it in INV.ITEMS if not args.only or it.name in args.only.split(",")]
    cols = int(np.ceil(np.sqrt(len(items))))
    cfg = DropSceneCfg(num_envs=1, env_spacing=10.0)
    slots = {}
    for i, it in enumerate(items):
        x, y = (i % cols) * SPACING, (i // cols) * SPACING
        slots[it.name] = (x, y)
        setattr(cfg, f"obj_{it.name}", RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/obj_{it.name}", spawn=it.spawn_cfg(),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, 0.5), rot=it.rest_rot)))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device="cpu", use_fabric=False))
    scene = InteractiveScene(cfg)
    spawn_size = {it.name: E.print_world_bounds(f"/World/envs/env_0/obj_{it.name}") for it in items}
    cam = Camera(CameraCfg(prim_path="/World/OverviewCam", width=1280, height=960, data_types=["rgb"],
                           spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, clipping_range=(0.05, 50.0))))
    sim.reset()
    scene.reset()
    # after reset: size must be unchanged (M8), then place each item 3 cm above the floor
    results = {}
    for it in items:
        lo0, hi0 = spawn_size[it.name]
        lo, hi = E.print_world_bounds(f"/World/envs/env_0/obj_{it.name}")
        size0, size = np.array(hi0) - np.array(lo0), np.array(hi) - np.array(lo)
        obj = scene[f"obj_{it.name}"]
        st = obj.data.default_root_state.clone()
        x, y = slots[it.name]
        st[0, 0], st[0, 1] = x, y
        st[0, 2] = 0.5 - lo[2] + DROP  # lowest point DROP above the floor
        obj.write_root_state_to_sim(st)
        results[it.name] = {"size_spawn": size0.round(4).tolist(), "size_after_reset": size.round(4).tolist()}
    extent = (len(items) // cols) * SPACING
    cx, cy = (cols - 1) * SPACING / 2, extent / 2
    cam.set_world_poses_from_view(eyes=torch.tensor([[cx, cy - 1.6, 2.2]]), targets=torch.tensor([[cx, cy, 0.0]]))
    for _ in range(180):
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(1 / 60)
    for it in items:
        obj = scene[f"obj_{it.name}"].data
        lo, hi = E.print_world_bounds(f"/World/envs/env_0/obj_{it.name}")
        speed = float(obj.root_lin_vel_w[0].norm())
        x, y = slots[it.name]
        drift = float(np.hypot(obj.root_pos_w[0, 0] - x, obj.root_pos_w[0, 1] - y))
        r = results[it.name]
        size_ok = np.allclose(r["size_spawn"], r["size_after_reset"], rtol=0.05, atol=0.002)
        # tilt: only yaw may change; compare the up-axis instead of the full rotation
        # the object's own axis that points up in its resting orientation, then where it points now
        local_up = Rotation.from_quat([it.rest_rot[1], it.rest_rot[2], it.rest_rot[3], it.rest_rot[0]]).inv().apply([0, 0, 1])
        w, qx, qy, qz = obj.root_quat_w[0].tolist()
        up = Rotation.from_quat([qx, qy, qz, w]).apply(local_up)
        tilt = float(np.degrees(np.arccos(np.clip(np.dot(up, [0, 0, 1]), -1, 1))))
        on_floor = abs(lo[2]) < 0.01
        ok = bool(size_ok and speed < 0.01 and tilt < 10.0 and on_floor and drift < 0.10)
        r.update(size_resting=(np.array(hi) - np.array(lo)).round(4).tolist(), speed=round(speed, 4), tilt_deg=round(tilt, 2),
                 rest_root_z=round(float(obj.root_pos_w[0, 2]) - float(lo[2]), 4),  # root height above the surface
                 rest_center_offset=[round(float(obj.root_pos_w[0, 0]) - (lo[0] + hi[0]) / 2, 4),
                                     round(float(obj.root_pos_w[0, 1]) - (lo[1] + hi[1]) / 2, 4)],
                 bottom_z=round(float(lo[2]), 4), drift=round(drift, 4), verified=ok)
        print(f"DROP {it.name:13s} {'PASS' if ok else 'FAIL'} size_ok={size_ok} resting size={r['size_resting']} "
              f"speed={speed:.4f} tilt={tilt:.1f}deg bottom={lo[2]:+.4f} drift={drift:.3f}", flush=True)
    cam.update(1 / 60)
    img = Image.fromarray(cam.data.output["rgb"][0, ..., :3].cpu().numpy())
    d = ImageDraw.Draw(img)
    d.text((10, 10), "inventory drop test: " + ", ".join(f"{k}={'ok' if v['verified'] else 'FAIL'}" for k, v in results.items())[:180], fill="white")
    out_img = E.ARTIFACTS / "inventory_drop_test.png"
    img.save(out_img)
    dst = E.PROJECT_ROOT / "assets" / "inventory_verified.json"
    old = json.loads(dst.read_text()) if dst.exists() else {}
    old.update(results)
    dst.write_text(json.dumps(old, indent=2))
    n = sum(r["verified"] for r in results.values())
    print(f"DROP TEST DONE: {n}/{len(results)} verified -> {dst.name}, image {out_img}", flush=True)


if __name__ == "__main__":
    E.run_main(main, simulation_app)
