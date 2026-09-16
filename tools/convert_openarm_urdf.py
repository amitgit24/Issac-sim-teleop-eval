#!/usr/bin/env python3
"""Convert the OpenArm URDF to a Robot_env USD with convex-DECOMPOSITION colliders and finger mimic.

The importer's defaults (convex-hull colliders) fill in the claw fingers' pocket:
objects get pushed over by the hull and "grasps" happen through overlapping hulls instead of
fingertip contact (docs/MISTAKES.md M24). Same URDF and settings otherwise.

Usage: /data/isaac/isaacsim/bin/python tools/convert_openarm_urdf.py
(package://openarm_description resolves to the vendored third_party/openarm_description)
"""
import os
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets" / "openarm" / "urdf" / "openarm_bimanual.urdf"
OUT = ROOT / "assets" / "openarm" / "openarm_bimanual.usd"


def main():
    # resolve package://openarm_description to the vendored copy
    os.environ["ROS_PACKAGE_PATH"] = str(ROOT / "third_party") + os.pathsep + os.environ.get("ROS_PACKAGE_PATH", "")
    cfg = UrdfConverterCfg(
        asset_path=str(URDF),
        usd_dir=str(OUT.parent),
        usd_file_name=OUT.name,
        force_usd_conversion=True,
        fix_base=True,
        root_link_name="openarm_body_link0",
        merge_fixed_joints=True,
        self_collision=False,
        collider_type="convex_decomposition",
        # Keep the URDF's finger mimic (finger_joint2 follows finger_joint1, like the real gripper's
        # linkage). Isaac Lab passes this flag straight to the importer's set_parse_mimic(), so True
        # PARSES the mimic despite the option's name; the default False made the fingers independent
        # and one claw shoved objects against the other (docs/MISTAKES.md M26).
        convert_mimic_joints_to_normal_joints=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=2.0),
        ),
    )
    path = Path(UrdfConverter(cfg).usd_path)
    if not path.is_file():
        raise SystemExit(f"converter did not write {path}")
    print(f"OPENARM USD CONVERTED: {path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        os._exit(0)
