#!/usr/bin/env python3
"""Survey candidate library assets before they enter the inventory (units, up axis, physics, size).

For each candidate: the source layer's metersPerUnit / upAxis, whether it already has a rigid body,
collision and mass, whether its meshes are instanced (then collision can't be authored on them),
and its size in meters once spawned in our meter stage (Isaac Lab's spawner applies the unit
correction). Writes assets/inventory_survey.json. Usage: $PY tools/survey_assets.py
"""
import json
import os
import sys
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes"))
import env_common as E  # noqa: E402

import omni.usd  # noqa: E402
from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

ROOT = E.ASSET_ROOT.rsplit("/Isaac/", 1)[0]  # .../Assets
YCB = f"{E.ASSET_ROOT}/Isaac/Props/YCB/Axis_Aligned"
AV = f"{ROOT}/ArchVis/Residential"
CANDIDATES = {
    # kitchenware
    "ycb_bowl": (f"{YCB}/024_bowl.usd", "kitchenware"),
    "ycb_pitcher": (f"{YCB}/019_pitcher_base.usd", "kitchenware"),
    "mug_a2": (f"{E.ASSET_ROOT}/Isaac/Props/Mugs/SM_Mug_A2.usd", "kitchenware"),
    "mug_b1": (f"{E.ASSET_ROOT}/Isaac/Props/Mugs/SM_Mug_B1.usd", "kitchenware"),
    "mug_c1": (f"{E.ASSET_ROOT}/Isaac/Props/Mugs/SM_Mug_C1.usd", "kitchenware"),
    "mug_d1": (f"{E.ASSET_ROOT}/Isaac/Props/Mugs/SM_Mug_D1.usd", "kitchenware"),
    "glass_short": (f"{AV}/Kitchen/Kitchenware/Dinnerware/P_Glassware_Short.usd", "kitchenware"),
    "glass_tall": (f"{AV}/Kitchen/Kitchenware/Dinnerware/P_Glassware_Tall.usd", "kitchenware"),
    "grey_bowl": (f"{AV}/Kitchen/Kitchenware/Serving/grey_bowl.usd", "kitchenware"),
    "salt_bowl": (f"{AV}/Kitchen/Kitchenware/Serving/salt_bowl.usd", "kitchenware"),
    # tools
    "power_drill": (f"{YCB}/035_power_drill.usd", "tools"),
    "scissors": (f"{YCB}/037_scissors.usd", "tools"),
    "large_marker": (f"{YCB}/040_large_marker.usd", "tools"),
    "large_clamp": (f"{YCB}/051_large_clamp.usd", "tools"),
    "xl_clamp": (f"{YCB}/052_extra_large_clamp.usd", "tools"),
    "wood_block": (f"{YCB}/036_wood_block.usd", "tools"),
    "foam_brick": (f"{YCB}/061_foam_brick.usd", "tools"),
    # containers / place targets
    "klt_bin": (f"{E.ASSET_ROOT}/Isaac/Props/KLT_Bin/small_KLT.usd", "containers"),
    "white_tray": (f"{AV}/Kitchen/Kitchenware/Serving/white_tray.usd", "containers"),
    "fruit_bowl": (f"{AV}/Kitchen/Kitchenware/StorageAndOrganization/fruit_bowl.usd", "containers"),
    "serving_bowl": (f"{AV}/Kitchen/Kitchenware/Serving/serving_bowl.usd", "containers"),
}


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device="cpu", use_fabric=False))
    stage = omni.usd.get_context().get_stage()
    out = {}
    for i, (name, (url, cat)) in enumerate(CANDIDATES.items()):
        rec = {"url": url, "category": cat}
        layer = Sdf.Layer.FindOrOpen(url)
        if layer is None:
            rec["error"] = "cannot open"
            out[name] = rec
            print(f"SURVEY {name}: CANNOT OPEN", flush=True)
            continue
        pr = layer.pseudoRoot
        rec["meters_per_unit"] = pr.GetInfo("metersPerUnit") if pr.HasInfo("metersPerUnit") else None
        rec["up_axis"] = pr.GetInfo("upAxis") if pr.HasInfo("upAxis") else None
        path = f"/World/Survey_{i}"
        cfg = sim_utils.UsdFileCfg(usd_path=url)
        cfg.func(path, cfg, translation=(i * 1.0, 5.0, 1.0))
        root = stage.GetPrimAtPath(path)
        prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
        rec["rigid_body"] = any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims)
        rec["collision_prims"] = sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in prims)
        rec["mass_api"] = any(p.HasAPI(UsdPhysics.MassAPI) for p in prims)
        meshes = [p for p in prims if p.IsA(UsdGeom.Mesh)]
        rec["meshes"] = len(meshes)
        rec["instanced_meshes"] = sum(p.IsInstanceProxy() for p in meshes)
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(root).ComputeAlignedRange()
        lo, hi = box.GetMin(), box.GetMax()
        rec["size_m"] = [round(hi[k] - lo[k], 4) for k in range(3)]
        rec["origin_offset_m"] = [round((lo[0] + hi[0]) / 2 - i * 1.0, 4), round((lo[1] + hi[1]) / 2 - 5.0, 4), round(lo[2] - 1.0, 4)]
        out[name] = rec
        print(f"SURVEY {name:12s} mpu={rec['meters_per_unit']} up={rec['up_axis']} rigid={rec['rigid_body']} "
              f"coll={rec['collision_prims']} mass={rec['mass_api']} meshes={rec['meshes']} inst={rec['instanced_meshes']} "
              f"size={rec['size_m']} origin_off={rec['origin_offset_m']}", flush=True)
    dst = E.PROJECT_ROOT / "assets" / "inventory_survey.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"SURVEY WRITTEN: {dst}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        os._exit(0)
