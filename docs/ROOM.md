# Configurable room

`scenes/room.py` adds an optional 5.00 × 4.56 × 2.70 m furnished room around any existing
Robot_env scene. The central table, robot, planners, and camera poses are unchanged. All presets
leave the ceiling off so the dome light remains effective.

## Presets

| Preset | Contents |
|---|---|
| `none` | Adds nothing. This is the default and preserves the original grid/table/robot scene. |
| `walls` | Warm neutral collidable walls plus a wood-colored visual floor over the existing physics ground. |
| `full` | `walls`, two side tables, two bookcases, four framed pictures, and three decor objects. |

All four scene entry points accept the same option:

```bash
/data/isaac/isaacsim/bin/python scenes/robot_env_scene.py --room full
/data/isaac/isaacsim/bin/python scenes/run_pick.py --object cube --room full
/data/isaac/isaacsim/bin/python scenes/run_handover.py --room full
/data/isaac/isaacsim/bin/python scenes/run_teleop.py --room full
```

For a wide diagnostic image from inside the room, use
`robot_env_scene.py --snapshot-view room --snapshot <path>`.

## Layout and clearance

- Clear interior: 5.00 m x, 4.56 m y, and 2.70 m high; walls are 0.12 m thick.
- The side-table fronts are at y = ±1.989 m, leaving 1.225 m from the central table's y edges.
- The west bookcase front is x = -2.194 m, 1.537 m behind the nearest pedestal edge. The east
  bookcase front is x = 2.194 m, 1.737 m from the central table edge.
- Paintings are centered at z = 1.50 m and embedded by about 1–2 cm into the wall surface so they
  read as wall-mounted rather than floating.
- The optional ceiling is controlled by `RoomCfg.ceiling` and is off in every named preset.

## ArchVis assets and measured sizes

The table remains the existing `EastRural_Table`. Furnishings are visual-only `AssetBaseCfg`
instances with `UsdFileCfg`; no collision or rigid-body API is applied inside a referenced asset.
The following bounds were measured in the complete scene after `sim.reset()` using
`env_common.print_world_bounds`.

| Asset server path | Instances | Measured size (m, local axes) |
|---|---:|---:|
| `<Assets>/ArchVis/Residential/Furniture/EndTables/Ellendale.usd` | 2 | 0.530 × 0.291 × 0.750 |
| `<Assets>/ArchVis/Residential/Furniture/Bookshelves/Dogtown.usd` | 2 | 0.952 × 0.305 × 1.800 |
| `<Assets>/ArchVis/Residential/Decor/Pictures/Painting_01.usd` | 2 | 0.050 × 1.410 × 1.054 |
| `<Assets>/ArchVis/Residential/Decor/Pictures/FramedPicture_Wide01.usd` | 2 | 0.989 × 0.032 × 0.450 |
| `<Assets>/ArchVis/Residential/Decor/Vases/CellVase01.usd` | 1 | 0.150 × 0.150 × 0.405 |
| `<Assets>/ArchVis/Residential/Decor/Books/BookStack_01.usd` | 1 | 0.429 × 0.229 × 0.285 |
| `<Assets>/ArchVis/Residential/Decor/Clocks/AlarmClock_Retro.usd` | 1 | 0.132 × 0.067 × 0.174 |

These older flat ArchVis files contain an authored top-level 0.01 scale. An explicit scale near
1.0 in `UsdFileCfg` replaces that authored scale and restores real-world dimensions. The side table
uses `(1.0, 0.6, 1.087)` and Dogtown uses `(1.0, 1.0, 1.281)` to obtain the dimensions above.
Keep scaling at the reference root; never edit child prims in these centimeter-authored assets.

## Customizing or extending

Use a named preset directly, or derive a custom immutable config:

```python
from dataclasses import replace

import env_common as E  # must precede room / Isaac Lab imports
from room import ROOM_PRESETS, add_room

cfg = MySceneCfg(num_envs=1, env_spacing=4.0)
custom_room = replace(ROOM_PRESETS["full"], decor=0, paintings=2, ceiling=True)
add_room(cfg, custom_room)
```

`RoomCfg` exposes `walls`, `floor`, `side_tables`, `shelves`, `paintings`, `decor`, and `ceiling`.
Counts select the first measured placements and accept 0–2 side tables, 0–2 shelves, 0–4
paintings, and 0–3 decor objects.

When adding another ArchVis asset:

1. List the server category and choose the USD path.
2. Spawn it as a visual-only `AssetBaseCfg`/`UsdFileCfg` in the real scene pipeline.
3. Measure its post-reset world bounds, including any top-level scale and rotation.
4. Place its lower bound on the floor or supporting surface and its back bound against the wall.
5. Render both the operational camera and the room-wide view before keeping it.

Import `env_common` before `room` or any other Isaac Lab module so the asset root is configured
before Isaac Lab reads it.
