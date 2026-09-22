#!/usr/bin/env python3
"""Wrap a visual-only prop USD into a physics-ready asset (rigid body + mass + collision).

Isaac Lab's UsdFileCfg only MODIFIES physics APIs that already exist, so visual-only props (the YCB
mug, Props/Mugs) need them authored first. The wrapper references the original (not copied) and
adds: RigidBodyAPI + MassAPI on the root, CollisionAPI + convexDecomposition on every mesh.
Only for assets authored in meters (metersPerUnit 1): editing inside a unit-corrected cm asset
triggers the 100x shrink bug (docs/MISTAKES.md M8).

Usage: /data/isaac/isaacsim/bin/python tools/make_physics_prop.py <source_usd_url> <out.usd> <mass_kg>
"""
import os
import sys

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


def main():
    src, out, mass = sys.argv[1], sys.argv[2], float(sys.argv[3])
    src_layer = Sdf.Layer.FindOrOpen(src)
    mpu = src_layer.pseudoRoot.GetInfo("metersPerUnit") if src_layer.pseudoRoot.HasInfo("metersPerUnit") else 1.0
    if abs(mpu - 1.0) > 1e-6:
        raise SystemExit(f"{src} has metersPerUnit {mpu}; only meter assets are supported (M8)")
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Prop").GetPrim()
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(src)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
    n = 0
    for p in Usd.PrimRange(root):
        if p.IsA(UsdGeom.Mesh):
            if p.IsInstanceProxy():
                raise SystemExit(f"{p.GetPath()} is instanced; cannot author collision on it")
            UsdPhysics.CollisionAPI.Apply(p)
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexDecomposition")
            n += 1
    if n == 0:
        raise SystemExit("no meshes found under the referenced prop")
    stage.GetRootLayer().Save()
    print(f"PHYSICS PROP WRITTEN: {out} ({n} collision meshes, mass {mass} kg)", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        os._exit(0)
