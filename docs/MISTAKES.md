# Mistakes and lessons — Robot_env

> Note: `openarm_manipulation` in these notes is the author's earlier OpenArm prototype (not part of this repository). This project started from it, then replaced every part that failed verification; the OpenArm files it needs are vendored in `third_party/` and `assets/openarm/`.

Each entry: symptom → root cause → fix → lesson. Referenced as `M#` from [PROCESS_LOG.md](PROCESS_LOG.md).
Check this list before repeating similar work.

## Isaac Sim / Isaac Lab

**M1 — Library assets resolved to `None/Isaac`.**
Symptom: `spawn_ground_plane` crashed with `Stage.GetPrimAtPath(Stage, NoneType)`, because no `Plane` child prim was found.
Cause: a plain `SimulationApp` doesn't set `/persistent/isaac/asset_root/cloud`. Isaac Lab's `AppLauncher` kit files do. The openarm project never noticed because it only used primitives.
Fix: set it to `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1` before importing `isaaclab`.
Lesson: when `ISAAC_NUCLEUS_DIR` is used with a plain `SimulationApp`, set the asset root first.

**M2 — Failures exited with code 0.**
Symptom: a traceback was printed, but the exit code was 0. The openarm scripts share this pattern (`raise SystemExit(code)` after `simulation_app.close()`).
Cause: `SimulationApp.close()` terminates the process with status 0.
Fix: on failure, call `os._exit(exit_code)` before `close()`. Confirmed: the step 3 sag check exited 1.
Lesson: never trust an Isaac script's exit code without testing a failing path.

**M3 — Picked a table by its name.**
Symptom: "SeattleLabTable" (Isaac Lab's standard manipulation table) is a black cabinet on casters with a camera-mount arm.
Fix: rendered all candidates side by side before choosing.
Lesson: render assets before committing to one; names and "commonly used" don't tell you what they look like.

**M4 — Scene too dark.**
Symptom: first snapshot mean brightness was 29.8/255.
Fix: sphere light 60000 → 150000 and lowered to z=2.5; dome fill 300 → 800 (mean brightness ~70).
Lesson: check a rendered image, not only whether the script runs.

**M7 — Explicit `scale=0.01` had no effect.**
Symptom: bounds were unchanged after adding it.
Cause: Isaac Lab's USD spawner already sets `xformOp:scale=0.01` for cm-authored assets (asset metersPerUnit 0.01 vs stage 1.0).
Fix: removed it.
Lesson: inspect the spawned prim's xformOps before "fixing" units.

**M8 — Table shrank 100× (1 cm tall).**
Symptom: the table was 0.762 m right after spawn but 0.008 m after `sim.reset()`.
Cause: applying `UsdPhysics.CollisionAPI`/`MeshCollisionAPI` to meshes inside the referenced cm asset re-triggered the cm→m correction, a second 0.01×. Bisected: correct after the scene was created and after the camera was spawned, wrong after colliders plus reset.
Fix: don't edit prims inside the asset; use a proxy collision box (D4).
Lesson: treat unit-corrected referenced assets as read-only. Keep the startup placement check that exits 1 when bounds are wrong.

**M9 — Standalone probes disagreed with the real scene.**
Symptom: the candidate probe showed the correct size, while the scene showed 1 cm.
Cause: the probe never applied physics APIs (see M8).
Lesson: probes must reproduce the real pipeline's steps, not only the spawn.

**M11 — USD bounds are stale when GPU physics is on.**
Symptom: the fingertip check reported -0.256 m (below the table) while the gripper was 0.20 m above it.
Cause: with GPU/Fabric physics, simulated poses are not written back to USD transforms, so `BBoxCache` reads the old pose.
Fix: use `robot.data.body_pos_w` / `body_quat_w`.
Lesson: for moving articulations, read poses from Isaac Lab's physics data, never from USD.

## Robotics / IK

**M10 — IK over-constrained.**
Symptom: the first ready-pose solve had 31 mm / 48° error.
Cause: required the full gripper orientation = identity (yaw included) at a target only 0.10 m below shoulder height. With joint6 limited to ±45°, that is infeasible.
Fix: constrain only the pointing axis (gripper z-axis = world up) and sweep height × inset.
Lesson: constrain only what the task needs. Sweep the workspace before choosing a target.

**M12 — Overstated fingertip length.**
Symptom: assumed the fingertips are ~0.10 m below `ee_base_link`.
Evidence: openarm tuned this from renders at ~0.05–0.07 m (`GRASP_Z_OFFSET` notes in `openarm_expert_common.py`).
Fix: kept 0.10 m, but labeled it as a conservative upper bound (`FINGERTIP_DROP`).
Lesson: check numbers against the sibling project's measured values before stating them.

**M13 — Gravity sag broke the pose check.**
Symptom: 0.05 rad error at joints 1–4.
Cause: stiffness 100 was tuned for openarm's hanging pose, where gravity torque is about zero.
Fix: D8 (robot gravity off).
Lesson: actuator gains copied from another project only hold for the poses that project used.

## Tooling / process

**M5 — Long probe hit its timeout with no output.**
Cause: the ArchVis asset download was slow. The grep pipe without `--line-buffered` hid progress, and `SimulationApp.close()` hung after the work was done.
Fix: log to a file, use `grep --line-buffered` or a monitor, and call `os._exit(0)` in probes instead of `close()`.

**M6 — `pkill -f <script>` killed its own shell.**
Cause: the pattern also matched the bash command line running `pkill`.
Fix: check with `ps aux | grep "[c]andidates.py"` and kill by PID.

**M14 — Scratch script named `bisect.py`.**
Symptom: "Unable to bootstrap inner kit kernel: cannot import name 'SimulationApp'...".
Cause: the file shadowed Python's stdlib `bisect` module.
Fix: renamed it to `scale_trace.py`.
Lesson: never name scripts after stdlib modules.

## Gripper, kinematics and pick task (step 4–5, 2026-09-24)

**M15 — openarm's gripper convention is inverted.**
Symptom: renders at finger q = 0 / −0.4 / −0.785 show q = 0 is CLOSED (fingertips touching) and |q| = 0.785 is fully open (13.9 cm). openarm's `GRIPPER_OPEN = 0.0`, `GRIPPER_CLOSED = −0.45` approached closed and "closed" to half-open.
Lesson: render a joint's range before trusting a named constant. This likely explains much of openarm's 12.5 % pick rate.

**M16 — fingertip depth was 2.5× off (supersedes M12).**
Measured: fingertips sit 0.152 m (open) to 0.171 m (closed) below `ee_base_link`, not the ~0.05–0.07 m openarm tuned, nor my 0.10 m "conservative bound". So step 3's "fingertips ≥ 0.10 m above the table" was really about 0.03 m.
Lesson: measure geometry in sim (CPU physics with `use_fabric=False` so USD transforms update; mesh points → gripper frame).

**M17 — `scene.reset()` does not move the robot to its default pose.**
Symptom: after reset the arm started at q = 0 and swung to the ready pose over the first steps.
Fix: `robot.write_joint_state_to_sim(default)` plus `set_joint_position_target(default)` on every reset.

**M18 — body poses read right after `sim.reset()` are not valid yet.**
Symptom: a camera aimed at the gripper position looked at the robot base.
Fix: step physics once before reading `body_pos_w`.

**M19 — the refactor broke M1 again.**
Symptom: `None/Isaac` after splitting the scene into `env_common.py`, because the entry script imported `isaaclab.sim` first.
Fix: `env_common` is always the first isaaclab-related import (documented in its docstring).

**M20 — my IK test checked the wrong thing.**
Targets put the fingers into the table, or out of reach; then "solver error vs target" failed on unreachable poses.
Fix: the test checks sim-vs-solver agreement and reports reachability separately.

**M21 — robot placement by heuristic left almost nothing pickable.**
The step-2 placement (shoulders 0.35 m above the table, 0.12 m gap) gave 8 pickable cells. My first sweep also required ONE yaw for grasp, pre-grasp and lift (over-constrained), and one monitor tailed the wrong output file (results duplicated).
Fix: correct criterion (grasp yaw fixed by the object, continuous vertical descent/lift, no IK branch jumps); placement re-chosen from the map (D10).

**M22 — "close to 2 cm narrower than the object" barely squeezed, and I misread the result.**
The finger stopping near −0.287 was just the position target being reached, not contact.
Fix: command fully closed with a force-limited finger actuator (D13).

**M23 — the claws only form a ~4 cm deep pocket.**
Measured inner gap (fully open) by height above the tips: 14 cm up to 2 cm, 11.9 at 3 cm, 8.7 at 4 cm, then < 2.5 cm (knuckles; housing 4.1 cm above tips). A can grasped low had its top hit the knuckles.
Fix: fingertips go 3.5 cm below the object's top (`FINGER_POCKET_DEPTH`).

**M24 — openarm's robot USD uses convex-HULL colliders.**
The hulls fill in the claw pocket: grasps "worked" through overlapping hulls rather than fingertip contact, and taller objects got pushed.
Fix: Robot_env converts its own USD with `convex_decomposition` (D12). The fixed-box test then stops the fingers at |q| ≈ 0.27–0.41 for 6 cm (hulls: 0.55).

**M25 — too much grip force and too few solver iterations: a finger sank through the object.**
At 10 N·m (~100 N) with openarm's 8/2 solver iterations and depenetration velocity 1.0, one finger pushed through a 0.35 kg can.
Fix: 3 N·m cap, 32/4 iterations, depenetration 5 m/s.

**M26 — the URDF's finger mimic was dropped.**
`finger_joint2` should mimic `finger_joint1` (as the real linkage does). Isaac Lab passes `convert_mimic_joints_to_normal_joints` straight to the importer's `set_parse_mimic()`, so the default `False` drops the mimic, despite the option's name. The independent claws then shoved objects against each other.
Fix: convert with `True`, and drive only `finger_joint1`.

**M27 — the imported mimic is a soft spring that yields under contact.**
naturalFrequency 25, dampingRatio 0.005: it tracks exactly in free motion, but the object pushed one claw open.
Fix: `harden_finger_mimics()` sets naturalFrequency 0, making it a rigid constraint. Result: both fingers stop symmetrically on the can (−0.355 / −0.371).

**M28 — wrist camera placement, and a misread image.**
My first tilt sign aimed away from the fingers; mounts above the housing only see the housing. I also read the off-axis wrist view as "the can is tipping". The per-step trace showed it never tilted.
Fix: camera below the housing on the −x side, aimed at the fingertips, chosen by rendering candidates. Lesson: confirm image impressions with a trace (object tilt/offset per step) before acting on them.

**M29 — the ready pose was too low for tall objects.**
Symptom: mustard bottle 2/10. The ready fingertips (0.129 m above the table) sat inside the 0.191 m bottle's space at reset, so it was knocked over before any grasp.
Fix: ready 0.40 m above the table (fingertips 0.229 m), inset/y re-chosen for joint margin (15.2 %). Result: mustard 10/10.

**M30 — seeding pre-grasp IK only from the current pose.**
Symptom: after raising the ready pose, cube and can were "no plan" (0/10). The new ready configuration sits on an IK branch that cannot descend continuously to low grasps.
Fix: the planner tries several seeds (current pose, a known low-reach configuration, random restarts), keeps the least-travel feasible plan, and rejects any joint-space approach whose fingertips dip below the object's top. Result: 10/10 again.
Also: a scratch render script without `harden_finger_mimics()` showed asymmetric fingers again (M9 repeated). Probes must use the real setup functions.

## Handover (step 6, 2026-09-24)

**M31 — over-constraining the handover pick.**
Required the right pick to close along world y so the later reorientation would put its claws on top and bottom. That yaw wasn't reachable in the pick zone (48/48 "right pick not reachable"). For a cylinder gripped across its axis the pick yaw doesn't matter.

**M32 — assumed the right gripper could point sideways with its claws vertical.**
Fingers pointing +y are reachable only with the claws rolled 60–90° (`tools/map_right_pointing.py`); roll 0 hits the wrist limits.

**M33 — pinching a horizontal cylinder at its equator.**
The left claws squeezed the can downward like a wedge. Moving the fingertips below the axis made the grip hold, but the can then pivoted out (next).

**M34 — gripping far from the center of mass.**
With the right hand on one end, the left could only grip 4.8 cm from the can's center. When the right released, the can slid out along its axis. The whole "horizontal can" design (right reorients, left grips from above) was dropped for the upright design (D19); the old planner is kept only as a reference outside the repo.

## LeRobot export and teleop (step 7, 2026-09-25)

**M35 — step-by-step IK gets stuck at IK branch boundaries.**
The first teleop test stopped at 0.235 m (LIMIT) descending onto the cube: the ready pose's IK branch
cannot reach the table continuously (the same structure as M30, but teleop can't jump branches
mid-motion). A lower "teleop home" reached only 16/36 tasks; a joint-centering term in the IK made it
worse (3–4/36). Fix: automatic reconfiguration with a fingertip-clearance check along the transition.

**M36 — the first teleop reach score was misleading.**
18/36 looked like a 50 % failure rate, but it required a specific yaw everywhere; only 25/36 tasks are
physically reachable at all, so the controller completes 18/25. Score against feasibility.

**M37 — scripted operator chose an unreachable yaw.**
The pick_cube script turned +25° (to yaw 180); at the pick center only yaw 90 is reachable for the
cube (checked offline with the real controller). Fixed to −65°. A scripted operator must also wait
during reconfiguration, like a person would.

**M38 — LeRobot's default video decoder (torchcodec) does not load here.**
Use `video_backend="pyav"`. Also: an image tolerance picked just above the worst observed diff (5.0
vs 4.95) is not evidence by itself; confirmed alignment separately on high-motion frames (20/20).

## Inventory, room, pick-and-place (step 8, 2026-09-25)

**M39 — physics wrappers for centimetre assets.**
Authoring the wrapper in the source's units still re-triggered the shrink (glass 0.0953 m → 0.001 m after `sim.reset()`). A meter wrapper that references the cm asset untouched was not unit-corrected at all (glass 9.5 m tall): only a directly spawned file is corrected, not a nested reference. Fix: meter wrapper; `/Prop/Visual` = untouched reference scaled by metersPerUnit on our own prim; `/Prop/Collision` = invisible mesh from the source points in metres. Verified: size unchanged after reset, rests on the table.

**M40 — drop-test tilt check applied the rest rotation twice.**
It reported exactly 90° for the three upright-rotated YCB items (bowl, pitcher, mug), while their resting sizes showed them upright. Fixed: tilt = angle between world up and R_now · (R_rest⁻¹ · z).

**M41 — pick-and-place was over-constrained at first.**
0 feasible container spots: the 10 cm straight retreat after release crossed an IK branch boundary. After release the gripper is free, so the fix is a 4 cm straight lift, then a checked joint-space move home. Releasing at floor level in the deep bin was also unreachable, so the runner uses the deepest reachable release depth.

**M42 — "settled" measured as instantaneous speed was noisy.**
Cans lying in the ribbed bin read 2–4 cm/s (flagged as failures) while moving ≤ 0.1 cm over the last second. Settled is now displacement over the last second < 1 cm.

**M43 — handover assumptions from the soup can leaked to other objects.**
The planner always used the can's grasp direction (mugs: "right pick not reachable"), and the success check measured from the object's center with a fixed 6 cm (a 14 cm glass held low "failed" while clearly held). Fixed: use each object's grasp axis; threshold = 3 cm + half the object's height.

**M44 — exact grasp axis halved coverage for 2-fold-symmetric objects.**
Mugs, bowl, marker and brick were pickable in only 5/10 orientations. A per-object yaw tolerance (exact axis first) raised mugs to 9/10 with zero grasp failures.
