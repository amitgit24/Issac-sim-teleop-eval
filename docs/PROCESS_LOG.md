# Process log — Robot_env (chronological, end to end)

> Note: `openarm_manipulation` in these notes is the author's earlier OpenArm prototype (not part of this repository). This project started from it, then replaced every part that failed verification; the OpenArm files it needs are vendored in `third_party/` and `assets/openarm/`.

What was done, in order. Decisions: [DECISIONS.md](DECISIONS.md) (`D#`). Mistakes: [MISTAKES.md](MISTAKES.md) (`M#`).
Scene file: `scenes/robot_env_scene.py`. Verify: `/data/isaac/isaacsim/bin/python scenes/robot_env_scene.py --headless --steps 120 --snapshot artifacts/scene_snapshot.png`.

---

## Step 0 — Survey (2026-09-23)

1. Looked at `project/`, which has `isaac_sim_scene/` (Agibot reference), `openarm_manipulation/` (full OpenArm pipeline) and `Robot_env/` (new, containing only `wooden_table.jpg`, a plain oak 4-leg table used as the reference).
2. Read the openarm docs (`README.md`, `docs/agent_environment.md`) and `scenes/openarm_scene.py` for the stack and style: Isaac Sim 5.1.0, Isaac Lab 0.47.2, `InteractiveSceneCfg`, `SimulationApp` created before any `isaaclab` import.
3. Both earlier projects build their tables from primitive cuboids. No table USDs exist on local disk.
4. Checked the NVIDIA cloud asset server: SeattleLabTable, PackingTable, ThorlabsTable and Simple_Room `table_low` all returned HTTP 200.

## Step 1 — Ground + center light + table (2026-09-23)

1. First version: `GroundPlaneCfg` grid, a `SphereLight` above the center, a weak `DomeLight` fill, and SeattleLabTable (D1, D2).
2. Run 1 crashed in `spawn_ground_plane`: the asset root was unset, giving `None/Isaac` (M1). The crash still exited 0 (M2). Fixed by setting `/persistent/isaac/asset_root/cloud` before importing `isaaclab`, then exiting with `os._exit(code)`.
3. Run 2 worked, but the render showed SeattleLabTable is a black cabinet with a camera arm (M3). The scene was also too dark (M4).
4. Listed the S3 bucket and found the Office props and the NVIDIA ArchVis furniture tables.
5. Rendered a contact sheet of 8 candidate tables. The first attempt timed out (M5); the cleanup then killed its own shell (M6). The second attempt succeeded.
6. Picked `ArchVis/.../DiningSets/EastRural/EastRural_Table.usd` (D3).
7. Raised the lights to sphere 150000 at z=2.5 plus dome 800.
8. The table came out 1 cm tall. An explicit `scale=0.01` changed nothing (M7). Bisecting showed that applying `CollisionAPI` to the asset's meshes causes a second cm→m shrink on `sim.reset()` (M8, M9).
9. Switched to an invisible collision box flush with the tabletop (D4) and added a startup check that fails if the table isn't 0..0.762 m tall.
10. Verified: table bounds (-0.457..0.457, -0.764..0.764, 0..0.762). A 5 cm cube dropped onto the table settled at z=0.787, as expected.
11. Wrote `README.md`.

## Step 2 — Add bimanual OpenArm (2026-09-24)

1. Reused `openarm_manipulation/scenes/generated/openarm_bimanual.usd` by reference (D5).
2. Read the URDF: fixed base, root `openarm_body_link0`, shoulder mounts at z=0.698 and y=±0.031 (arms at ±0.155), 7 revolute joints + 2 fingers per arm (18 total), ~0.5 m reach, facing +x, arms hanging down at q=0.
3. Tabletop (0.762) is above shoulder height (0.698), so the robot needs a pedestal (D6). Base at (-0.577, 0, 0.414): 0.12 m behind the -x long edge, shoulders 0.35 m above the tabletop.
4. Copied the actuator settings from openarm (implicit PD, stiffness 100, damping 2).
5. Added checks: joint states stay finite, gripper positions printed. Moved the camera to a front-right 3/4 view.
6. Verified: 18 joints, max joint error 0.0007 rad, hanging grippers at z=0.676 and 0.12 m behind the table edge (clear).

## Step 3 — Start ("ready") pose (2026-09-24)

1. The openarm project has no joint-space ready pose; its expert uses IK to targets.
2. Wrote an offline FK + bounded IK solver, now `tools/solve_ready_pose.py` (URDF chain, scipy `least_squares`, joints kept inside limits).
3. The first target (0.25 m above the table, 0.18 m inset, full orientation = identity) was infeasible: 31 mm / 48° error (M10).
4. Constrained only the "fingers point down" axis, then swept hover height (0.10–0.25) × inset (0.05–0.20). Every combination was feasible; joint-limit margin shrinks as height increases.
5. Chose 0.20 m above the table and 0.10 m inset, y=±0.20 (D7). Solution: error 0.02 mm / 0.01°, min margin 7% (joint7). The right arm is the exact mirror.
6. Put it in `READY_JOINT_POS` as the articulation's init/default state.
7. First run: 0.05 rad sag at joints 1–4 under gravity. The check caught it and exited 1, which also confirmed the M2 fix. Disabled gravity on the robot links (D8), bringing the error to 0.0001 rad.
8. The fingertip check read stale USD bounds (M11). Switched to physics body pose minus a conservative 0.10 m fingertip drop (M12).
9. Verified: both grippers at (-0.357, ±0.200, 0.962), 0.1 mm error, 0.0° tilt, fingertips at least 0.10 m above the table, exit 0.

## Step 4 — Rethink the pick approach, measure the gripper, choose placement (2026-09-24)

Goal: position-based pick with the right arm, simple object first, then household objects, recording joints and EE pose (obs and action) plus videos. Instruction: don't trust openarm's scripts; its demonstrator reached 12.5 %.

1. Checked the preconditions: gravity-off is used by Isaac Lab (`FRANKA_PANDA_HIGH_PD_CFG`, Galbot, Fourier, Kuka-Allegro), so D8 stands. YCB physics assets (soup can, mustard, boxes) and mugs are available; imageio, imageio_ffmpeg and ffmpeg are installed.
2. Read openarm's failure analysis. Its "stochastic" 2 cm / 20° residual matches the gravity sag measured in step 3 (0.05 rad at 0.5 m reach).
3. Refactored: `scenes/env_common.py` (library) plus thin entry scripts; `scenes/layout.py` (placement constants, no Isaac imports). Hit M19 (import order).
4. Measured the gripper in sim (CPU physics, `use_fabric=False`, mesh points in the gripper frame) and rendered close-ups: q = 0 is CLOSED (M15); fingertips 0.152–0.171 m below the gripper frame (M16); opening 18.5 cm/rad, max 13.9 cm, along the gripper y-axis; closing center at the gripper origin. Found that `scene.reset()` does not apply the default pose (M17).
5. Wrote `scenes/kinematics.py` (URDF FK + bounded IK) and `tests/check_kinematics.py`: FK matches the sim exactly (0.000 mm on 40 random configs) and IK round trips land exactly (M20 on the test design).
6. Mapped the pickable workspace (`tools/map_pick_workspace.py`). The step-2 placement gave 8 cells. The best was shoulders 0.40 m above the table with a 0.05 m gap: 21 cells (D10, M21). Re-solved the ready pose (D11).

## Step 5 — Pick task: cube, then household objects (2026-09-24)

1. `scenes/pick_planner.py` (D9), `scenes/pick_task.py` (objects, cameras, reset, success, observations) and `scenes/run_pick.py` (episodes, npz + MP4 per camera, `--trace`, `--no-video`).
2. Cube: 5/5 on the first run. Checked the video: black first frame and mis-aimed wrist cameras. Fixed with 3 renders after reset and a wrist camera below the housing chosen from rendered candidates (D16, M28).
3. Soup can 1/5. Diagnosis chain, each step verified before the next:
   - weak squeeze → force-limited closing (M22, D13);
   - claw pocket only ~4 cm deep → grasp near the top (M23, D14);
   - openarm USD had convex-hull colliders → own USD with convex decomposition (M24, D12);
   - a finger sank through the can → 3 N·m cap and 32/4 solver iterations (M25);
   - the finger mimic was dropped → convert with the mimic kept (M26);
   - the mimic was soft → rigid (M27).
   New permanent test: `tests/check_gripper_contact.py` (both fingers must stop on a fixed box).
4. Mustard 2/10: the ready pose was too low (M29). Raised it to 0.40 m (15.2 % joint margin); then cube and can failed to plan (M30), fixed with multi-seed planning and an approach check (D18).
5. Mug: wrapped the visual-only YCB mug with physics (`tools/make_physics_prop.py`, `assets/objects/ycb_mug_physics.usd`). Grasp across the body, perpendicular to the handle.
6. Verified (10 episodes each, seed 10, final code): cube 10/10, soup can 10/10, mustard 10/10, mug 8/10 (2 orientations have no reachable grasp yaw, reported as "no plan"; 8/8 executed grasps succeeded). Side-view renders confirm real, symmetric two-finger grasps on the can and the mug.

## Step 6 — Handover, right to left (soup can) (2026-09-24)

1. First design: right reorients the can horizontally, left grips its free end from above. Searched handover points offline with the real planner (`tools/search_handover_point.py`) and hit three blockers in turn: the pick yaw was over-constrained (M31); sideways pointing is only reachable with the claws rolled 60–90° (M32, `tools/map_right_pointing.py`); then in sim the can slipped out, first wedged by an equator pinch (M33), then gripped too far from its center of mass (M34).
2. Redesign (D19): the can stays upright. Right carries it by its top; left wraps its lower body horizontally from its own side. The offline search found 10 feasible points; chose inset 0.12, y −0.04, right gripper 0.36 m above the table.
3. Traced run: symmetric left contact (0.32/0.36). The can drops only 9 mm as the right releases, and the left carries it away.
4. Verified: 10/10 (seed 20, no video) + 3/3 with video (`artifacts/episodes/video_handover`). Final regression: scene check, `tests/check_kinematics.py` and `tests/check_gripper_contact.py` all pass.

## Step 7 — LeRobot export and keyboard/gamepad teleop (2026-09-25)

1. pi0.5 integration paused (owner's call). LeRobot export delegated to Codex (high reasoning) with a written spec: `tools/export_lerobot.py`, `tools/validate_lerobot.py`, `docs/LEROBOT_EXPORT.md`. Verified independently: 13 episodes / 3,135 frames validate, and frame alignment holds on high-motion frames (20/20 on both cameras). Default decoder issue noted (M38).
2. Teleop built: `scenes/teleop_input.py` (keyboard, gamepad, scripted), `scenes/teleop_controller.py` (EE velocity control + IK + clamp + hold), `scenes/run_teleop.py` (GUI app, status line, R/F/X recording). Controls in `docs/TELEOP.md` (D21).
3. First scripted test stopped at a branch boundary (M35). Tried a lower teleop home (16/36) and IK joint-centering (worse), then added automatic reconfiguration. Offline reach test: 18 of the 25 physically reachable tasks (M36).
4. Fixed the scripted operator's yaw (M37). In sim: turn −65°, descend with one automatic reconfiguration, grasp, lift the cube 0.119 m; the episode exports and validates in LeRobot format.
5. Input test with injected carb events: 28/28 (keyboard + virtual gamepad). GUI launch on a display with live input: clean. No human operator session recorded yet.

## Step 8 — Asset inventory, room, clutter, pick-and-place, handover variety (2026-09-25)

1. Asked the owner about assets: kitchenware + tools + containers, from the NVIDIA/YCB library, for clutter, pick-and-place and handover variety.
2. Surveyed 21 candidates (`tools/survey_assets.py`): YCB items are Y-up and visual-only; Props mugs are meters and visual-only; ArchVis items are centimetres; the KLT bin has physics.
3. Physics wrappers for all visual-only items; the cm case needed a split visual/collision design (M39). Drop test 21/21 after fixing a tilt-check bug (M40). Registry `scenes/inventory.py`, `docs/INVENTORY.md`.
4. Room (owner request mid-step) delegated to Codex with a spec: `scenes/room.py`, presets none / walls / full, `--room` in all runners (D25). Verified.
5. Clutter (`scenes/clutter.py`, `--clutter N`) and pick-and-place (`scenes/place_planner.py`, `scenes/run_pick_place.py`, container spot search). Fixed retreat/depth (M41) and the settled check (M42): soup can → KLT bin 10/10, with 6 clutter items 3/3.
6. Inventory pick verification: 0 grasp failures; misses were "no plan" for 2-fold-symmetric items. Yaw tolerance (D26, M44): mugs 9/10, grey bowl 10/10, glasses 10/10, foam brick 8/10, marker 7/10.
7. Handover variety: `--object` plus fixes (M43): soup can 5/5, glass_short 5/5, glass_tall 5/5, mustard 5/5, mug_c1 4/5; foam brick too short for the side-wrap.

## Step 9 — Showcase videos (2026-09-25)

1. `--showcase` option on run_pick / run_pick_place / run_handover: a 1280×720 camera orbiting ±38° on the room side, recorded per episode (not part of the dataset). `tools/record_room_tour.py`: 360° orbit of the furnished room (radius kept inside the furniture).
2. Recorded in the full room with 6 clutter items: mug_c1 pick 2/2, soup can → bin 2/2. The tall-glass handover succeeded 2/2 but the glass is transparent and nearly invisible on video, so the showcase uses the mustard bottle (2/2) with a closer orbit centered on the handover point.
3. `tools/make_showcase_video.py`: title cards, room tour, task clips with a wrist-camera inset and captions, fades → `media/showcase.mp4` (60 s, 8.2 MB) and `media/showcase.gif` (README hero). Text via PIL (this ffmpeg has no drawtext); long titles auto-shrink (the end-card URL overflowed at first).

## Step 10 — Pick → handover → place into a bin (2026-09-25)

1. New task requested: the right hand picks, hands the object to the left, and the left places it in a bin; test with 3–4 objects. Wrote the planner myself; delegated the runner to Codex with a spec (it wrote and syntax-checked it; I ran and verified it).
2. Offline bin-spot search: 0/36 at first (M45). Allowing a wrist turn while carrying gave 20 spots for all 4 objects; chose inset 0.40, y +0.24.
3. Sim: soup can, mustard, short glass 5/5 each; mug 2/5. Traces showed two separate causes: the mug slipping in a fast carry (M46, carry slowed to 4 s) and the handle hitting the left hand at the handover (M47, handle-aware handover yaw). Mug: 10/10.
4. Final: soup can 5/5, mustard 5/5, short glass 5/5, mug 10/10; plain handover regression soup can 5/5, mug 5/5 (was 4/5). Unit tests 20/20 (new: transfer plan, handle direction). Showcase clip added to `media/showcase.mp4` (fixed a dtype crash on the way, M48).

## Next (not started)

- Record human teleop demos; larger scripted data collection; export with `tools/export_lerobot.py`.
- pi0.5 integration (paused) and the evaluation harness.
- Mug: 2/10 orientations have no reachable grasp yaw; a tilted grasp or a second pick zone could cover them.
- Table legs still have no collision (D4).
