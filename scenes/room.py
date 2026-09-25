"""Configurable room shell and visual ArchVis furnishings for any Robot_env scene.

Import this module only after :mod:`env_common`; that module must establish Isaac's cloud asset
root before any Isaac Lab import (docs/MISTAKES.md M1/M19).
"""

from dataclasses import dataclass, replace

import env_common as E
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg


ROOM_SIZE = (5.0, 4.56, 2.7)  # clear interior x, y, z (m)
WALL_THICKNESS = 0.12
FLOOR_THICKNESS = 0.04

# Wide diagnostic view from inside the south-east corner. Keeping it inside the walls matters:
# an outside camera would only see an opaque wall.
ROOM_VIEW_EYE = (2.15, -1.95, 2.35)
ROOM_VIEW_TARGET = (-0.15, 0.05, 0.85)

ARCHVIS = f"{E.NVIDIA_ASSETS_DIR}/ArchVis/Residential"
ELLENDALE_USD = f"{ARCHVIS}/Furniture/EndTables/Ellendale.usd"
DOGTOWN_USD = f"{ARCHVIS}/Furniture/Bookshelves/Dogtown.usd"
PAINTING_USD = f"{ARCHVIS}/Decor/Pictures/Painting_01.usd"
FRAMED_WIDE_USD = f"{ARCHVIS}/Decor/Pictures/FramedPicture_Wide01.usd"
BOOK_STACK_USD = f"{ARCHVIS}/Decor/Books/BookStack_01.usd"
VASE_USD = f"{ARCHVIS}/Decor/Vases/CellVase01.usd"
ALARM_CLOCK_USD = f"{ARCHVIS}/Decor/Clocks/AlarmClock_Retro.usd"

# The older flat ArchVis files above carry centimeter stage metadata around meter-like authored
# coordinates. Direct UsdFileCfg references therefore measure 0.01x after reset. Explicit top-level
# instance scales near 1.0 replace the authored 0.01 scale and restore the measured dimensions without touching any referenced prim (M8).
SIDE_TABLE_SCALE = (1.0, 0.6, 1.087)  # 0.530 x 0.291 x 0.750 m
BOOKCASE_SCALE = (1.0, 1.0, 1.281)  # 0.952 x 0.305 x 1.800 m
ARCHVIS_CM_FIX = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class RoomCfg:
    """Independent room feature switches/counts.

    Counts select the first N predefined, measured placements. The stock layout provides two side
    tables, two bookcases, four paintings, and three small decor objects.
    """

    walls: bool = False
    floor: bool = False
    side_tables: int = 0
    shelves: int = 0
    paintings: int = 0
    decor: int = 0
    ceiling: bool = False


ROOM_PRESETS = {
    "none": RoomCfg(),
    "walls": RoomCfg(walls=True, floor=True),
    "full": RoomCfg(walls=True, floor=True, side_tables=2, shelves=2, paintings=4, decor=3),
}


def room_cfg(preset: str, **overrides) -> RoomCfg:
    """Return a named preset, optionally with dataclass-field overrides."""
    try:
        base = ROOM_PRESETS[preset]
    except KeyError as exc:
        raise ValueError(f"unknown room preset {preset!r}; choose from {tuple(ROOM_PRESETS)}") from exc
    return replace(base, **overrides)


def _cuboid(name, size, pos, color, *, collision=False):
    return name, AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Room{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=(
                sim_utils.CollisionPropertiesCfg(collision_enabled=True) if collision else None
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.72),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _archvis(name, usd_path, pos, rot=(1.0, 0.0, 0.0, 0.0), scale=ARCHVIS_CM_FIX):
    # Visual-only by design. Never add rigid/collision APIs inside these unit-corrected references.
    return name, AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Room{name}",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path, scale=scale),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def add_room(scene_cfg, cfg: RoomCfg | str):
    """Attach the requested room entities to an existing InteractiveSceneCfg instance.

    The function mutates and returns ``scene_cfg`` so it works with RobotEnvSceneCfg as well as
    pick, handover, and teleoperation subclasses. With ``"none"`` it adds nothing.
    """
    if isinstance(cfg, str):
        cfg = room_cfg(cfg)
    if not isinstance(cfg, RoomCfg):
        raise TypeError(f"cfg must be RoomCfg or preset name, got {type(cfg).__name__}")

    maxima = {"side_tables": 2, "shelves": 2, "paintings": 4, "decor": 3}
    for field, limit in maxima.items():
        value = getattr(cfg, field)
        if not isinstance(value, int) or not 0 <= value <= limit:
            raise ValueError(f"{field} must be an integer from 0 to {limit}, got {value!r}")

    xh, yh, height = ROOM_SIZE[0] / 2.0, ROOM_SIZE[1] / 2.0, ROOM_SIZE[2]
    entities = []
    if cfg.floor:
        # Top is 1 mm above the grid to avoid z-fighting; physics remains on the existing ground.
        entities.append(
            _cuboid(
                "Floor",
                (ROOM_SIZE[0], ROOM_SIZE[1], FLOOR_THICKNESS),
                (0.0, 0.0, -FLOOR_THICKNESS / 2.0 + 0.001),
                (0.30, 0.19, 0.105),
            )
        )
    if cfg.walls:
        wall_color = (0.82, 0.79, 0.71)
        entities.extend(
            [
                _cuboid("WallWest", (WALL_THICKNESS, ROOM_SIZE[1] + 2 * WALL_THICKNESS, height),
                         (-xh - WALL_THICKNESS / 2.0, 0.0, height / 2.0), wall_color, collision=True),
                _cuboid("WallEast", (WALL_THICKNESS, ROOM_SIZE[1] + 2 * WALL_THICKNESS, height),
                         (xh + WALL_THICKNESS / 2.0, 0.0, height / 2.0), wall_color, collision=True),
                _cuboid("WallSouth", (ROOM_SIZE[0], WALL_THICKNESS, height),
                         (0.0, -yh - WALL_THICKNESS / 2.0, height / 2.0), wall_color, collision=True),
                _cuboid("WallNorth", (ROOM_SIZE[0], WALL_THICKNESS, height),
                         (0.0, yh + WALL_THICKNESS / 2.0, height / 2.0), wall_color, collision=True),
            ]
        )
    if cfg.ceiling:
        entities.append(
            _cuboid(
                "Ceiling",
                (ROOM_SIZE[0] + 2 * WALL_THICKNESS, ROOM_SIZE[1] + 2 * WALL_THICKNESS, WALL_THICKNESS),
                (0.0, 0.0, height + WALL_THICKNESS / 2.0),
                (0.91, 0.90, 0.86),
                collision=True,
            )
        )

    side_tables = [
        _archvis("SideTableNorth", ELLENDALE_USD, (-1.45, yh - 0.145, 0.0), scale=SIDE_TABLE_SCALE),
        _archvis("SideTableSouth", ELLENDALE_USD, (1.45, -yh + 0.145, 0.0), scale=SIDE_TABLE_SCALE),
    ]
    # Dogtown's native back is local y=0. A +/-90-degree yaw puts that back on the wall and faces
    # the shelf openings into the room. Its measured z range is 0.00..1.80 after placement.
    s45 = 0.70710678
    shelves = [
        _archvis("BookcaseWest", DOGTOWN_USD, (-xh, 1.45, 0.0645),
                 rot=(s45, 0.0, 0.0, s45), scale=BOOKCASE_SCALE),
        _archvis("BookcaseEast", DOGTOWN_USD, (xh, 1.45, 0.0645),
                 rot=(s45, 0.0, 0.0, -s45), scale=BOOKCASE_SCALE),
    ]
    paintings = [
        _archvis("PaintingEast", PAINTING_USD, (xh - 0.015, -0.55, 1.50), rot=(0.0, 0.0, 0.0, 1.0)),
        _archvis("PaintingWest", PAINTING_USD, (-xh + 0.015, -0.55, 1.50)),
        _archvis("PaintingNorth", FRAMED_WIDE_USD, (0.25, yh - 0.015, 1.50)),
        _archvis("PaintingSouth", FRAMED_WIDE_USD, (-0.25, -yh + 0.015, 1.50),
                 rot=(0.0, 0.0, 0.0, 1.0)),
    ]
    # Side-table tops are at z=0.750 m. All decor assets have their measured lower bound at z=0.
    decor = [
        _archvis("Vase", VASE_USD, (-1.45, yh - 0.145, 0.750)),
        _archvis("BookStack", BOOK_STACK_USD, (1.45, -yh + 0.145, 0.750)),
        _archvis("AlarmClock", ALARM_CLOCK_USD, (-1.28, yh - 0.145, 0.750)),
    ]
    entities.extend(side_tables[: cfg.side_tables])
    entities.extend(shelves[: cfg.shelves])
    entities.extend(paintings[: cfg.paintings])
    entities.extend(decor[: cfg.decor])

    for name, entity_cfg in entities:
        setattr(scene_cfg, f"room_{name.lower()}", entity_cfg)
    return scene_cfg
