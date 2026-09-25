"""Table clutter: random, non-overlapping placement of verified inventory items as distractors.

Items are chosen once per run (they are spawned with the scene) and re-placed every episode with
`place_clutter`. Placement uses conservative circular footprints (half the item's resting diagonal
+ margin) on the tabletop, and keeps out of exclusion zones (the pick zone, the handover point, the
strip in front of the robot the arms sweep through).
"""

import math
from dataclasses import dataclass

import numpy as np

import inventory as INV
import layout as L

TABLE_MARGIN = 0.04  # keep footprints this far inside the table edge
FOOTPRINT_MARGIN = 0.03  # includes up to ~2 cm root-vs-center offset (e.g. mug handles)
ROBOT_STRIP = 0.12  # nothing within this distance of the robot-side (near) table edge


@dataclass
class Zone:
    x: float
    y: float
    r: float


def default_exclusions() -> list:
    """Pick zone (object jitter + claw opening), handover point, and a spare place spot."""
    return [
        Zone(L.PICK_CENTER[0], L.PICK_CENTER[1], 0.16),
        Zone(L.TABLE_NEAR_X + 0.12, -0.04, 0.16),  # handover point (run_handover.py defaults)
    ]


def footprint(item) -> float:
    sx, sy = item.size_m[0], item.size_m[1]
    return 0.5 * math.hypot(sx, sy) + FOOTPRINT_MARGIN


def choose_items(rng, n: int, exclude_names=()) -> list:
    pool = [it for it in INV.with_role("clutter") if it.name not in exclude_names and it.size_m]
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in idx]


def sample_layout(rng, items, exclusions=None, tries: int = 400):
    """(x, y, yaw) per item on the tabletop (world frame), non-overlapping; None where no spot was found."""
    exclusions = default_exclusions() if exclusions is None else exclusions
    x_lo = L.TABLE_NEAR_X + ROBOT_STRIP
    x_hi = L.TABLE_POS[0] + L.TABLE_TOP_SIZE[0] / 2.0 - TABLE_MARGIN
    y_lo = L.TABLE_POS[1] - L.TABLE_TOP_SIZE[1] / 2.0 + TABLE_MARGIN
    y_hi = L.TABLE_POS[1] + L.TABLE_TOP_SIZE[1] / 2.0 - TABLE_MARGIN
    placed, out = [], []
    for it in sorted(items, key=footprint, reverse=True):  # big items first
        r = footprint(it)
        spot = None
        for _ in range(tries):
            x, y = rng.uniform(x_lo + r, x_hi - r), rng.uniform(y_lo + r, y_hi - r)
            if any(math.hypot(x - z.x, y - z.y) < r + z.r for z in exclusions):
                continue
            if any(math.hypot(x - px, y - py) < r + pr for px, py, pr in placed):
                continue
            spot = (x, y, rng.uniform(-math.pi, math.pi))
            placed.append((x, y, r))
            break
        out.append((it, spot))
    return out


def add_clutter_to_cfg(cfg, items):
    """Add one RigidObjectCfg per item to a scene config instance (attribute names clutter_<name>)."""
    from isaaclab.assets import RigidObjectCfg

    for i, it in enumerate(items):
        # park them off the table at spawn; place_clutter() puts them on the table every episode
        setattr(cfg, f"clutter_{it.name}", RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/clutter_{it.name}", spawn=it.spawn_cfg(),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0 + 0.4 * i, 3.0, 0.3), rot=it.rest_rot)))
    return [f"clutter_{it.name}" for it in items]


def place_clutter(scene, items, rng, exclusions=None):
    """Re-place the clutter items for a new episode. Items without a free spot are parked off the table."""
    import torch
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

    layout = sample_layout(rng, items, exclusions)
    placed = 0
    for i, (it, spot) in enumerate(layout):
        obj = scene[f"clutter_{it.name}"]
        st = obj.data.default_root_state.clone()
        if spot is None:
            st[0, :3] = torch.tensor([3.0 + 0.4 * i, 3.0, 0.3 + it.rest_root_z])
        else:
            x, y, yaw = spot
            yaw_t = torch.tensor([yaw], device=st.device)
            yq = quat_from_euler_xyz(torch.zeros_like(yaw_t), torch.zeros_like(yaw_t), yaw_t)
            st[0, 3:7] = quat_mul(yq, torch.tensor([it.rest_rot], device=st.device, dtype=st.dtype))[0]
            st[0, :3] = torch.tensor([x, y, L.TABLE_HEIGHT + it.rest_root_z + 0.002], device=st.device)
            placed += 1
        st[0, 7:] = 0.0
        obj.write_root_state_to_sim(st)
    return placed
