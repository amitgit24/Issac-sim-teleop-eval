# Object inventory, clutter and pick-and-place

`scenes/inventory.py` is the single registry of objects that can go into a scene. An item counts as
**in inventory** only after `tools/drop_test_assets.py` passes for it: dropped 3 cm onto a surface, it
settles (speed < 1 cm/s), keeps its resting orientation (tilt < 10°), rests on the surface, and keeps
its measured size after `sim.reset()` (guards the unit-correction shrink bug, M8/M39). Results live in
`assets/inventory_verified.json` and are loaded automatically; nothing is hand-marked verified.

Sources: NVIDIA's Isaac Sim asset server — YCB models (`Props/YCB/Axis_Aligned`), Isaac Props (mugs,
KLT bin) and ArchVis kitchenware. Visual-only assets are wrapped with physics by
`tools/make_physics_prop.py` into `assets/objects/<name>_physics.usd` (rigid body + real-world mass
+ convex-decomposition collision). Centimetre ArchVis assets get a separate invisible collision mesh
built from their points in metres, so nothing inside the referenced asset is edited (M39).

## Items (21, all drop-test verified)

| Category | Items | Roles |
|---|---|---|
| Kitchenware | YCB bowl, YCB pitcher, YCB mug, mugs A2/B1/C1/D1, short glass, tall glass, grey bowl, salt bowl, serving bowl | clutter; pick: mugs, glasses, grey bowl; place target: bowls |
| Tools | power drill (0.9 kg), scissors, large marker, large clamp, XL clamp, wood block (0.73 kg), foam brick | clutter; pick: marker, foam brick |
| Containers | KLT bin, white tray | clutter; place target |

Roles: `clutter` (distractor), `pick` (graspable by the claw gripper's top pinch), `place_target`,
`handover` (right → left upright side-wrap). Pick and handover roles carry their verified results in
`pick_verified`.

## Verified pick / handover results (scripted, 10 picks / 5 handovers, seed-fixed)

| Object | Pick | Handover |
|---|---|---|
| Short glass / tall glass | 10/10 · 10/10 | 5/5 · 5/5 |
| Grey bowl | 10/10 | – |
| Mugs A2 / B1 / C1 / D1 | 9/10 each | mug C1: 4/5 |
| Foam brick | 8/10 | too short for the side-wrap (planner reports "no room") |
| Large marker | 7/10 | – |
| (earlier) cube, soup can, mustard, YCB mug | 10/10, 10/10, 10/10, 8/10 | soup can 10/10, mustard 5/5 |

Every miss is a `no plan` (no reachable grasp for that orientation, never attempted); there were
**zero executed grasp failures**. Objects with 2-fold grasp symmetry (mugs, bowl, marker, brick) use a
grasp-yaw tolerance (±35° / ±15°): exact-axis grasps first, off-axis only when needed — this raised
the mugs from 5/10 to 9/10.

## Clutter

`scenes/clutter.py`: N verified items chosen per run, re-placed every episode at random,
non-overlapping spots (circular footprints with a 3 cm margin), keeping out of the pick zone, the
handover point / place target, and the strip in front of the robot. `--clutter N` in
`scenes/run_pick.py` and `scenes/run_pick_place.py`.

## Pick-and-place

`scenes/place_planner.py` + `scenes/run_pick_place.py`: the proven top pick, a joint-space transfer
keeping the hanging object's bottom 4 cm above the rim, a straight descent into the container
(deepest reachable release depth), release, a 4 cm retreat and a joint move back to ready. Container
spots from `tools/search_place_spot.py`: the KLT bin works at inset 0.22–0.30 m, y −0.04 (used: 0.26);
salt bowl / YCB bowl have a few spots; serving bowl and the long tray have none with this robot
placement. Success = object inside the container footprint, below the rim, moved < 1 cm in the final
second, released. Soup can → KLT bin: **10/10**; with 6 clutter items: **3/3**.

```bash
PY=/path/to/isaacsim/bin/python
$PY tools/drop_test_assets.py                     # (re)verify the inventory
$PY scenes/run_pick.py --object mug_c1 --episodes 10 --clutter 6 --room full
$PY scenes/run_pick_place.py --object soup_can --container klt_bin --episodes 5 --clutter 6
$PY scenes/run_handover.py --object glass_tall --episodes 3
```

To add an object: add the source to `tools/survey_assets.py`, wrap it with `tools/make_physics_prop.py`,
add an `InventoryItem`, run the drop test, then (for pick) set `grasp_width` / `grasp_axis_local` /
`yaw_symmetry` and verify with `run_pick.py`.
