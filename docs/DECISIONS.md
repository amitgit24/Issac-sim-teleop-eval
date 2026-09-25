# Decisions — Robot_env

> Note: `openarm_manipulation` in these notes is the author's earlier OpenArm prototype (not part of this repository). This project started from it, then replaced every part that failed verification; the OpenArm files it needs are vendored in `third_party/` and `assets/openarm/`.

Each entry: decision, alternatives, reason, evidence. Referenced as `D#` from [PROCESS_LOG.md](PROCESS_LOG.md).

## D1 — Same stack and style as openarm_manipulation (2026-09-23)
**Decision:** Isaac Sim 5.1.0 python (`/data/isaac/isaacsim/bin/python`), Isaac Lab `InteractiveSceneCfg`, and a plain `SimulationApp` created before any `isaaclab` import, as in `openarm_scene.py`.
**Reason:** keeps the openarm data-collection/eval patterns directly reusable.
**Consequence:** the asset root must be set by hand (see M1). Isaac Lab's `AppLauncher` would do this automatically.

## D2 — Ground and lighting (2026-09-23)
**Decision:** Isaac grid `GroundPlaneCfg` at z=0, a `SphereLight` at (0, 0, 2.5) with intensity 150000 as the "center light", and a `DomeLight` fill at 800.
**Alternatives:** a cuboid ground (the openarm approach) or only the center light.
**Reason:** the grid gives visual scale in renders. With only the center light, surfaces facing away from it render nearly black (M4).

## D3 — Table: ArchVis EastRural_Table (2026-09-23)
**Decision:** `Assets/ArchVis/Residential/Furniture/DiningSets/EastRural/EastRural_Table.usd`, centered at the origin: 0.914 × 1.528 m, top at z=0.762.
**Alternatives rendered side by side:** SeattleLabTable (black cabinet with camera arm), PackingTable (steel packing station), Simple_Room `table_low` (white sculpted), Office SM_TableA/D (office desks), Jennings/DesPeres/Whittershins (loaded but rendered invisible).
**Reason:** the closest match to `wooden_table.jpg` (wooden top, 4 legs, dining height). The legs are white metal rather than oak; an all-wood table would need a custom or imported asset.
**Evidence:** contact-sheet render during step 1.

## D4 — Table collision via an invisible proxy box (2026-09-23)
**Decision:** don't modify the table asset. Collision comes from `TableTopCollider`, an invisible static `CuboidCfg` of 0.914 × 1.528 × 0.04 whose top is flush at z=0.762.
**Alternatives:** convex-hull colliders on the asset's meshes. That shrinks the table 100× (M8).
**Reason:** avoids the unit-rescale bug, and a flat box gives clean contacts for manipulation.
**Limitation:** the legs have no collision yet. Add leg boxes if the arms or objects can reach them.
**Evidence:** a 5 cm cube dropped onto the table settled at z=0.787 (0.762 + 0.025).

## D5 — Reuse the OpenArm USD by reference (2026-09-24)
**Decision:** `OPENARM_USD = ../openarm_manipulation/scenes/generated/openarm_bimanual.usd`, no copy. Articulation and actuator settings are copied from `openarm_task_common.py`.
**Reason:** one source of truth, regenerated in that project by `convert_openarm_urdf.py`. The script fails fast if the file is missing.

## D6 — Robot on a pedestal behind the near long edge (2026-09-24)
**Decision:** robot base at (-0.577, 0, 0.414) on a 0.30 × 0.40 × 0.414 m pedestal with collision, facing +x across the 0.914 m table depth, 0.12 m gap to the edge.
**Reason:** the tabletop (0.762) is above the robot's shoulder height (0.698) when it stands on the floor. The openarm project found shoulders 0.2 m above the table too cramped (joint4 saturated in IK) and settled at about 0.35 m. The 0.12 m gap lets the hanging arms clear the edge.
**Rule:** keep `ROBOT_POS` derived from the table constants. If the table changes, the ready pose must be re-solved (D7).

## D7 — Start pose: grippers down, 0.20 m above the table, 0.10 m inset (2026-09-24)
**Decision:** `READY_JOINT_POS`. Left arm `-0.5354, -0.9970, 1.0048, 1.8043, 0.5688, -0.0346, -1.3498`, right arm its mirror, fingers 0 (open). Grippers at (-0.357, ±0.200, 0.962).
**Alternatives:** q=0 (arms hanging below table height, useless for manipulation). Higher hover heights: 0.25 m cuts the minimum joint-limit margin from 7% to 5%.
**Reason:** grippers point straight down (the natural approach direction for top grasps), fingertips are at least 0.10 m clear, and all joints are at least 7% from their limits. The openarm project's worst IK failures came from joints sitting at their limits.
**Method:** `tools/solve_ready_pose.py` (URDF forward kinematics + scipy bounded least-squares, only the pointing axis constrained). Re-run it if the robot or table moves.

## D8 — Robot gravity disabled (2026-09-24)
**Decision:** `disable_gravity=True` on the robot's rigid bodies.
**Alternatives:** keep gravity and raise stiffness, which changes the dynamics compared to openarm, or add gravity-compensation feed-forward.
**Reason:** with gravity on, stiffness 100 lets the bent ready pose sag about 0.05 rad (joints 1–4). Disabling it models an ideally gravity-compensated arm, like real arm controllers, and matches Isaac Lab's Franka config.
**Consequence:** differs from openarm (gravity on). Revisit if policies trained here must transfer to that sim.

## D9 — Pick demonstrator: offline joint-space planning (2026-09-24)
**Decision:** `scenes/pick_planner.py` plans the whole pick before moving: free-space min-jerk joint move to pre-grasp, then a dense straight vertical descent (IK every 1 cm, chained, no joint jump > 0.3 rad), close, lift, hold. The arm tracks joint targets.
**Alternatives:** openarm's online `DifferentialIKController` (drifted about 2 cm / 20°, pinned joints).
**Reason:** `scenes/kinematics.py` (URDF FK + bounded IK) matches the sim exactly (0.000 mm over 40 random configs, `tests/check_kinematics.py`), and the gravity-free arm tracks targets to ~1e-4 rad, so the plan is exact.

## D10 — Robot placement from reachability (supersedes D6) (2026-09-24)
**Decision:** shoulders 0.40 m above the tabletop, base 0.05 m behind the near edge (`scenes/layout.py`). The body column is ±3 cm deep at tabletop height, so it clears.
**Evidence:** `tools/map_pick_workspace.py` pickable cells: 8 at the old 0.35/0.12, 17 at 0.45/0.0, 21 at 0.40/0.05, 0 at ≥ 0.50. Pick zone: 0.10–0.20 m in from the edge, y −0.25 to −0.35 (right arm).

## D11 — Ready pose re-solved (supersedes D7 values) (2026-09-24)
Grippers down and fully open, 0.30 m above the table at (inset 0.10, ±0.25). Wrist yaw chosen for the widest joint margin (11.4 %) by `tools/solve_ready_pose.py`. The left arm is solved independently and is an exact mirror.

## D12 — Robot_env's own robot USD (supersedes D5) (2026-09-24)
**Decision:** `tools/convert_openarm_urdf.py` converts the same URDF to `assets/openarm/` with convex-decomposition colliders and the finger mimic kept; `env_common.harden_finger_mimics()` makes the mimic rigid.
**Reason:** M24, M26, M27. openarm_manipulation's files are left untouched.

## D13 — Force-limited gripper (2026-09-24)
Only `finger_joint1` is driven (joint2 is its mimic), and closing commands q = 0. The finger actuator caps effort at 3 N·m (~30 N), so the squeeze doesn't depend on where the claws touch. Robot solver iterations are 32/4 and depenetration velocity 5 m/s (M25).

## D14 — Grasp near the object's top (2026-09-24)
Fingertip height = object height − 3.5 cm (min 1 cm), because the claw pocket is only ~4 cm deep (M23). The pre-grasp keeps the fingertips 3 cm above the object's top. The fingers open to object width + 4 cm.

## D15 — Recording format (2026-09-24)
30 Hz (60 Hz physics, record every 2nd step). The observation is measured *before* the step's action is applied. Both arms, each with 7 joint positions, the gripper (finger_joint1), and the EE pose (xyz + quat wxyz in the robot root frame). The action has the same fields, commanded; its EE pose is the exact FK of the commanded joints. `episode_XXXX.npz` plus an MP4 per camera (cam_high, cam_right_wrist, cam_left_wrist; 640×480, 30 fps), with `summary.json`.

## D16 — Cameras (2026-09-24)
`cam_high` is fixed, looking over the pick area. Wrist cameras are fixed to each `ee_base_link` at (−0.10, 0, −0.08), below the housing, aimed at the fingertip point (0, 0, −0.16), focal length 10. Chosen from rendered candidates (M28). Three renders after each reset, because RTX output lags the physics state.

## D17 — Objects (2026-09-24)
The cube (primitive, friction 1.2/1.0, "max" combine) comes first, then YCB `Axis_Aligned_Physics` soup can and mustard bottle, stood upright with +90° about x (the assets are Y-up). Each object defines a grasp axis in its own frame that is rotated into the world at plan time. The mug is pending: the YCB and Props mugs are visual-only, so physics has to be added.

## D18 — Multi-seed planning with an approach safety check (2026-09-24)
The pre-grasp IK is seeded from the current pose, a known low-reach configuration, and 4 random seeds; the feasible plan with the least joint travel wins. The joint-space approach is sampled (25 points) and rejected if the fingertips pass below the object's top minus 1 cm. A "no plan" result is reported, never executed (mug: 2/10 orientations have no reachable grasp yaw perpendicular to the handle).

## D19 — Handover design: upright can, left side-wrap (2026-09-24)
**Decision:** `scenes/handover_planner.py` + `scenes/run_handover.py` (soup can):
1. Right does the proven top pick and carries the can upright to the handover point.
2. Left comes in horizontally from its own side (fingers pointing −y, claws rolled 90° so they close along x), fingertips 1 cm past the can's axis, pads below right's claws.
3. Left closes (force-limited), right opens and lifts straight up 10 cm.
4. Left backs off +y and up while holding the can.

Handover point from `tools/search_handover_point.py` (10 feasible points): can at inset 0.12, y −0.04, right gripper 0.36 m above the table. That point is feasible at 0.30, 0.36 and 0.42 m.
**Rejected:** reorienting the can horizontally and having the left grip from above (M31–M34).
**Why it works:** the left claws wrap the can's lower body with its center of mass inside the grip; the two hands occupy disjoint heights (right: top 3.5 cm; left: pads ±2.9 cm below that).
**Success check:** can ≥ 5 cm above resting height, ≤ 6 cm from the LEFT fingertip point, ≥ 8 cm from the right fingertips.

## D20 — LeRobot v2.1 dataset schema (2026-09-25)
`tools/export_lerobot.py` (written by Codex from a spec, verified independently): `observation.state` [16] = right joints 1–7, right gripper, left joints 1–7, left gripper (measured); `observation.ee_pose` [14]; `action` [16] (commanded joint + gripper targets); `action.ee_pose` [14] (exact FK of the commanded joints); `observation.phase`; three video streams; task = the episode prompt; 30 fps. Only successful episodes by default. `tools/validate_lerobot.py` checks every number exactly and images within a lossy-encoding tolerance. Runs in a LeRobot environment (`lerobot==0.1.0`), not Isaac Sim's Python. First dataset: `robot_env_demo_v0`, 13 episodes / 3,135 frames.
**pi0.5 integration is paused** by the owner's choice; the joint-space `action` makes it a direct fit later.

## D21 — Teleop design (2026-09-25)
World-frame end-effector velocity control with online IK from the previous joints, a safe workspace clamp (fingertips never below the table), hold-on-limit, and automatic reconfiguration (M35). Keyboard + Xbox-layout gamepad through `carb.input` (our own mapping, not Isaac Lab's Se3 devices, so arm switching and recording controls fit). Scripted source for tests. Recording is identical to the scripted demos (D15); `phase` = active arm (0 right, 1 left).

## D22 — Object inventory with a drop-test gate (2026-09-25)
The owner chose kitchenware, tools and containers from the NVIDIA/YCB library, for clutter, pick-and-place and handover variety. `scenes/inventory.py` is the registry; `tools/survey_assets.py` measures candidates (units, up axis, physics, size); `tools/make_physics_prop.py` wraps visual-only ones (M39); `tools/drop_test_assets.py` gates entry (21/21 pass). Fruit bowl dropped (44 × 44 × 67 cm, includes a stand). Real masses (YCB published values where available).

## D23 — Clutter (2026-09-25)
Random non-overlapping placement per episode with conservative circular footprints and exclusion zones (pick zone, handover / place target, robot-side strip). Items are chosen per run and re-placed per episode (spawning per episode would be slow).

## D24 — Pick-and-place into a fixed container (2026-09-25)
KLT bin as a kinematic place target at the feasible spot found offline (inset 0.26 m, y −0.04). Deepest reachable release, strict success check (inside, below rim, at rest, released). Serving bowl and white tray stay clutter-only: no reachable spot with this robot placement.

## D25 — Room built by Codex (2026-09-25)
`scenes/room.py` + `docs/ROOM.md`: presets `none` (default, unchanged results), `walls`, `full` (5.0 × 4.56 × 2.7 m, 2 side tables, 2 bookcases, 4 pictures, decor). Visual-only ArchVis furniture spawned directly; collidable walls; ceiling off. Verified by Codex and re-checked visually: picks 3/3 with and without the room. Note: two furniture pieces are scaled non-uniformly to realistic sizes.

## D26 — Grasp-yaw tolerance (2026-09-25)
`GraspSpec.yaw_tolerance`: exact-axis grasps are tried first, then ±tol/2 and ±tol, with a travel penalty for off-axis. Mugs / grey bowl ±35° (body grasp away from the handle), marker / foam brick ±15°.

## D27 — Pick → handover → place task (2026-09-25)
`scenes/transfer_planner.py` (mine) + `scenes/run_handover_place.py` (written by Codex from my spec, verified by me). The proven handover (D19), then the left carries the object over a fixed KLT bin and **releases it just above the rim**. The left claws hold the object's lower body from the side, so lowering it into the bin would drive the claws into the wall. Then back out along the gripper axis and return home. The bin spot (inset 0.40, y +0.24) is the center of a 20-spot block found feasible for all 4 test objects by `tools/search_transfer_spot.py`, clear of the pick zone and the left hand's approach corridor. The wrist may turn while carrying (M45); the carry takes 4 s (M46); handover yaws keep a mug's handle away from the left hand (M47).
