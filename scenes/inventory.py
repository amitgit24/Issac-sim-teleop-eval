"""Object inventory: every asset that can be placed in the scene, with verified physics and roles.

No Isaac imports at module level (sizes/roles are usable offline); `spawn_cfg()` builds the Isaac Lab
spawner lazily. An item is only "in inventory" once tools/drop_test_assets.py has passed for it
(`verified` below records the result: it settles upright/stable on a surface and keeps its measured
size after sim.reset(), guarding the unit-correction bug M8).

Roles: "clutter" (distractor on the table), "pick" (graspable by the claw gripper: pinch <= 13.9 cm
wide at the top 3.5 cm, see kinematics.FINGER_POCKET_DEPTH), "place_target" (container/tray to put
things in), "handover" (can be passed right -> left with the upright side-wrap, D19).
Sources: NVIDIA Isaac Sim asset server (YCB models, Isaac Props, ArchVis). Physics wrappers are built
by tools/make_physics_prop.py into assets/objects/.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBJ = ROOT / "assets" / "objects"
# YCB "Axis_Aligned" models are Y-up; this stands them upright (local +y -> world +z).
YCB_UPRIGHT = (0.70710678, 0.70710678, 0.0, 0.0)
IDENTITY = (1.0, 0.0, 0.0, 0.0)


@dataclass
class InventoryItem:
    name: str
    category: str  # kitchenware | tools | containers | groceries | primitives
    usd: str  # path (relative to repo root) or URL of the physics-ready asset
    mass: float  # kg (real-world where published, e.g. YCB)
    rest_rot: tuple = IDENTITY  # orientation (wxyz) it rests in on the table
    roles: tuple = ("clutter",)
    source: str = ""
    size_m: tuple = ()  # measured x, y, z extent in its resting orientation (drop test)
    rest_root_z: float = 0.0  # root height above the surface when resting (drop test)
    verified: bool = False  # tools/drop_test_assets.py passed (loaded from assets/inventory_verified.json)
    notes: str = ""
    # Grasp for the "pick"/"handover" roles (kinematics: claws pinch the top 3.5 cm). width = size across
    # the fingers near the top; grasp_axis_local = object-local axis the fingers close along;
    # yaw_symmetry = grasp repeats every this many radians. Pick roles count as verified only after
    # scripted picks succeed (pick_verified, filled from run_pick results).
    grasp_width: float = 0.0
    grasp_axis_local: tuple = (1.0, 0.0, 0.0)
    yaw_symmetry: float = 3.141592653589793
    yaw_tolerance: float = 0.0  # rad off the exact grasp axis still acceptable (mugs: body grasp away from the handle)
    pick_verified: str = ""

    def spawn_cfg(self):
        import isaaclab.sim as sim_utils

        path = self.usd if "://" in self.usd else str(ROOT / self.usd)
        return sim_utils.UsdFileCfg(
            usd_path=path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=4),
            mass_props=sim_utils.MassPropertiesCfg(mass=self.mass),
        )


def _obj(name):
    return str((OBJ / f"{name}_physics.usd").relative_to(ROOT))


_YCB = "YCB via NVIDIA Isaac Props/YCB/Axis_Aligned"
_MUGS = "NVIDIA Isaac Props/Mugs"
_AV = "NVIDIA ArchVis Residential/Kitchen/Kitchenware"
_ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac"

ITEMS = [
    # --- kitchenware
    InventoryItem("ycb_bowl", "kitchenware", _obj("ycb_bowl"), 0.147, YCB_UPRIGHT, ("clutter", "place_target"), _YCB + "/024_bowl"),
    InventoryItem("ycb_pitcher", "kitchenware", _obj("ycb_pitcher"), 0.178, YCB_UPRIGHT, ("clutter",), _YCB + "/019_pitcher_base"),
    InventoryItem("ycb_mug", "kitchenware", "assets/objects/ycb_mug_physics.usd", 0.25, YCB_UPRIGHT, ("clutter", "pick"), _YCB + "/025_mug",
                  notes="pick verified 8/10 in pick_task (2 orientations have no reachable grasp)", pick_verified="pick 8/10 (2 no-plan) as pick_task 'mug'"),
    InventoryItem("mug_a2", "kitchenware", _obj("mug_a2"), 0.30, IDENTITY, ("clutter", "pick", "handover"), _MUGS + "/SM_Mug_A2", grasp_width=0.092, yaw_tolerance=0.61, pick_verified="pick 9/10 (1 no-plan, 0 grasp failures); handover 4/5 (mug_c1 tested)"),
    InventoryItem("mug_b1", "kitchenware", _obj("mug_b1"), 0.30, IDENTITY, ("clutter", "pick", "handover"), _MUGS + "/SM_Mug_B1", grasp_width=0.092, yaw_tolerance=0.61, pick_verified="pick 9/10 (1 no-plan, 0 grasp failures)"),
    InventoryItem("mug_c1", "kitchenware", _obj("mug_c1"), 0.32, IDENTITY, ("clutter", "pick", "handover"), _MUGS + "/SM_Mug_C1", grasp_width=0.089, yaw_tolerance=0.61, pick_verified="pick 9/10 (1 no-plan, 0 grasp failures); handover 4/5"),
    InventoryItem("mug_d1", "kitchenware", _obj("mug_d1"), 0.32, IDENTITY, ("clutter", "pick", "handover"), _MUGS + "/SM_Mug_D1", grasp_width=0.089, yaw_tolerance=0.61, pick_verified="pick 9/10 (1 no-plan, 0 grasp failures)"),
    InventoryItem("glass_short", "kitchenware", _obj("glass_short"), 0.20, IDENTITY, ("clutter", "pick", "handover"), _AV + "/Dinnerware/P_Glassware_Short", grasp_width=0.081, yaw_symmetry=0.5236, pick_verified="pick 10/10; handover 5/5"),
    InventoryItem("glass_tall", "kitchenware", _obj("glass_tall"), 0.25, IDENTITY, ("clutter", "pick", "handover"), _AV + "/Dinnerware/P_Glassware_Tall", grasp_width=0.083, yaw_symmetry=0.5236, pick_verified="pick 10/10; handover 5/5"),
    InventoryItem("grey_bowl", "kitchenware", _obj("grey_bowl"), 0.15, IDENTITY, ("clutter", "pick"), _AV + "/Serving/grey_bowl", grasp_width=0.076, grasp_axis_local=(0.0, 1.0, 0.0), yaw_tolerance=0.61, pick_verified="pick 10/10"),
    InventoryItem("salt_bowl", "kitchenware", _obj("salt_bowl"), 0.30, IDENTITY, ("clutter", "place_target"), _AV + "/Serving/salt_bowl"),
    InventoryItem("serving_bowl", "kitchenware", _obj("serving_bowl"), 0.35, IDENTITY, ("clutter", "place_target"), _AV + "/Serving/serving_bowl"),
    # --- tools (YCB, resting flat as stored)
    InventoryItem("power_drill", "tools", _obj("power_drill"), 0.895, IDENTITY, ("clutter",), _YCB + "/035_power_drill",
                  notes="0.9 kg: too heavy for the 3 N*m pinch; clutter only"),
    InventoryItem("scissors", "tools", _obj("scissors"), 0.082, IDENTITY, ("clutter",), _YCB + "/037_scissors"),
    InventoryItem("large_marker", "tools", _obj("large_marker"), 0.016, IDENTITY, ("clutter", "pick"), _YCB + "/040_large_marker", grasp_width=0.019, yaw_tolerance=0.26, pick_verified="pick 7/10 (3 no-plan, 0 grasp failures)"),
    InventoryItem("large_clamp", "tools", _obj("large_clamp"), 0.125, IDENTITY, ("clutter",), _YCB + "/051_large_clamp"),
    InventoryItem("xl_clamp", "tools", _obj("xl_clamp"), 0.202, IDENTITY, ("clutter",), _YCB + "/052_extra_large_clamp"),
    InventoryItem("wood_block", "tools", _obj("wood_block"), 0.729, IDENTITY, ("clutter",), _YCB + "/036_wood_block"),
    InventoryItem("foam_brick", "tools", _obj("foam_brick"), 0.028, IDENTITY, ("clutter", "pick"), _YCB + "/061_foam_brick", grasp_width=0.051, grasp_axis_local=(0.0, 1.0, 0.0), yaw_tolerance=0.26, pick_verified="pick 8/10 (2 no-plan, 0 grasp failures); too short for the handover side-wrap"),
    # --- containers / place targets
    InventoryItem("klt_bin", "containers", f"{_ASSET_ROOT}/Props/KLT_Bin/small_KLT.usd", 0.5, IDENTITY, ("clutter", "place_target"),
                  "NVIDIA Isaac Props/KLT_Bin (physics included)"),
    InventoryItem("white_tray", "containers", _obj("white_tray"), 0.30, IDENTITY, ("clutter", "place_target"), _AV + "/Serving/white_tray"),
]
BY_NAME = {it.name: it for it in ITEMS}

# Drop-test results (tools/drop_test_assets.py): verified flag + measured resting size.
_VERIFIED = ROOT / "assets" / "inventory_verified.json"
if _VERIFIED.exists():
    for _name, _res in json.loads(_VERIFIED.read_text()).items():
        if _name in BY_NAME:
            BY_NAME[_name].verified = bool(_res.get("verified"))
            BY_NAME[_name].size_m = tuple(_res.get("size_resting", ()))
            BY_NAME[_name].rest_root_z = float(_res.get("rest_root_z", 0.0))


def with_role(role: str, verified_only: bool = True):
    return [it for it in ITEMS if role in it.roles and (it.verified or not verified_only)]
