#!/usr/bin/env python3
"""Wrap a visual-only prop USD into a physics-ready asset (rigid body + mass + collision).

Isaac Lab's UsdFileCfg only MODIFIES physics APIs that already exist, so visual-only props (the YCB
mug, Props/Mugs) need them authored first. The wrapper references the original (not copied) and
adds: RigidBodyAPI + MassAPI on the root, CollisionAPI + convexDecomposition on every mesh.
Unit handling:
- meter assets: RigidBodyAPI + MassAPI on the root, CollisionAPI + convexDecomposition on each mesh.
- centimeter assets (e.g. ArchVis): nothing inside the referenced asset may be edited: any physics API
  there re-triggers the unit correction on sim.reset() and shrinks it 100x (docs/MISTAKES.md M8, M39;
  also when the wrapper itself is authored in cm). So the wrapper is in meters: /Prop (rigid body,
  mass) > /Prop/Visual (untouched reference, scaled by metersPerUnit on our own prim) + /Prop/Collision
  (invisible mesh built from the source's points converted to meters, convexDecomposition).

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
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Prop").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
    n = 0
    if abs(mpu - 1.0) < 1e-6:
        root.GetReferences().AddReference(src)
        for p in Usd.PrimRange(root):
            if p.IsA(UsdGeom.Mesh):
                if p.IsInstanceProxy():
                    raise SystemExit(f"{p.GetPath()} is instanced; cannot author collision on it")
                UsdPhysics.CollisionAPI.Apply(p)
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexDecomposition")
                n += 1
    else:
        vis_x = UsdGeom.Xform.Define(stage, "/Prop/Visual")
        vis = vis_x.GetPrim()
        vis.GetReferences().AddReference(src)  # untouched inside
        # A nested reference is NOT unit-corrected by the spawner (only a directly spawned file is), so
        # scale our own Visual prim by metersPerUnit (M39).
        vis_x.AddScaleOp().Set((mpu, mpu, mpu))
        src_stage = Usd.Stage.Open(src)
        src_root = src_stage.GetDefaultPrim()
        cache = UsdGeom.XformCache()
        root_inv = cache.GetLocalToWorldTransform(src_root).GetInverse()
        for p in Usd.PrimRange(src_root):
            if not p.IsA(UsdGeom.Mesh):
                continue
            m = UsdGeom.Mesh(p)
            xf = cache.GetLocalToWorldTransform(p) * root_inv
            pts = [xf.Transform(v) * mpu for v in m.GetPointsAttr().Get()]
            col = UsdGeom.Mesh.Define(stage, f"/Prop/Collision/mesh_{n}")
            col.CreatePointsAttr(pts)
            col.CreateFaceVertexCountsAttr(m.GetFaceVertexCountsAttr().Get())
            col.CreateFaceVertexIndicesAttr(m.GetFaceVertexIndicesAttr().Get())
            col.CreatePurposeAttr(UsdGeom.Tokens.guide)  # invisible in renders
            UsdPhysics.CollisionAPI.Apply(col.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr("convexDecomposition")
            n += 1
    if n == 0:
        raise SystemExit("no meshes found in the source prop")
    stage.GetRootLayer().Save()
    print(f"PHYSICS PROP WRITTEN: {out} ({n} collision meshes, mass {mass} kg)", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        os._exit(0)
