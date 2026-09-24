# Robot_env — developer notes (steps 1–6)

Project docs (read the one you need):
- [PROCESS_LOG.md](PROCESS_LOG.md): every step, in order
- [DECISIONS.md](DECISIONS.md): decisions with alternatives and reasons (`D#`)
- [MISTAKES.md](MISTAKES.md): mistakes, root causes, fixes, lessons (`M#`); check before similar work

Sister project to `../openarm_manipulation`. Same stack: Isaac Sim 5.1.0
(`/data/isaac/isaacsim/bin/python`) + Isaac Lab 0.47.2, `InteractiveSceneCfg` style.

## Step 1 (done): ground + center light + library table

`scenes/robot_env_scene.py`

| Element | Asset | Notes |
|---|---|---|
| Ground | Isaac grid ground plane (`GroundPlaneCfg`) | z = 0 |
| Center light | `SphereLight` at (0, 0, 2.5) | plus a weak dome fill light |
| Table | NVIDIA ArchVis `EastRural_Table.usd` | wooden top, 0.914 x 1.528 m, top surface z = 0.762, centered at origin |
| Table collision | invisible static box, flush with the tabletop | the ArchVis asset is visual-only |

```bash
cd Issac-sim-teleop-eval   # repo root
/data/isaac/isaacsim/bin/python scenes/robot_env_scene.py                  # GUI
/data/isaac/isaacsim/bin/python scenes/robot_env_scene.py --headless --steps 120 \
    --snapshot artifacts/scene_snapshot.png                                 # headless check + image
```

The headless run exits non-zero if the table is not at the expected placement.

## Step 2 (done): bimanual OpenArm robot

> **Superseded in step 4:** robot USD is now `assets/openarm/` (own conversion, D12), placement is shoulders 0.40 m above the table with a 0.05 m gap (D10), finger actuators are force-limited (D13). Current values live in `scenes/layout.py` / `scenes/env_common.py`.

| Element | Value |
|---|---|
| Robot USD | `../openarm_manipulation/scenes/generated/openarm_bimanual.usd` (referenced, not copied; 18 joints, fixed base) |
| Placement | base at (-0.577, 0, 0.414), behind the table's -x long edge, facing +x across the 0.914 m depth |
| Pedestal | 0.30 x 0.40 x 0.414 m box with collision, so the shoulders sit 0.35 m above the tabletop (the openarm project found 0.2 m too cramped) |
| Actuators | same implicit PD as openarm (stiffness 100, damping 2); robot gravity disabled (see step 3) |

## Step 3 (done): start ("ready") pose

> **Superseded in steps 4–5:** the ready pose is now 0.40 m above the table at inset 0.15, y ±0.25, fingers fully open (D11, M29), and the "fingertips ≥ 0.10 m" figure below was wrong (fingertips are 0.17 m below the gripper frame, M16). Current values: `scenes/layout.py`.

`READY_JOINT_POS` in `scenes/robot_env_scene.py` is the robot's initial/default joint state:

| | Value |
|---|---|
| Grippers | pointing straight down, open (finger joints 0) |
| Gripper frame (`ee_base_link`) | (-0.357, +-0.200, 0.962): 0.10 m in from the table's near edge, 0.20 m above the tabletop |
| Fingertips | >= 0.10 m above the tabletop |
| Left arm joints 1-7 | -0.5354, -0.9970, 1.0048, 1.8043, 0.5688, -0.0346, -1.3498 |
| Right arm joints 1-7 | mirror: 0.5354, 0.9970, -1.0048, 1.8043, -0.5688, 0.0346, 1.3498 |

Solved offline with URDF forward kinematics + bounded IK: every joint stays >= 7% of its range
from a limit (tightest: joint7). Higher hover poses leave less margin, e.g. 0.25 m above the table
drops it to 5%. Re-solve if the robot or table placement changes.

Robot gravity is disabled (an ideally gravity-compensated arm, as in Isaac Lab's Franka config). With
gravity on, stiffness 100 lets this bent pose sag ~0.05 rad at joints 1-4. The openarm project kept
gravity on because its home pose hangs straight down.

Every headless run checks the settled pose and exits 1 if it drifts: joint error <= 0.02 rad,
gripper position within 1 cm, tilt <= 3 deg, fingertips clear of the table. Last run: 0.0001 rad,
0.1 mm, 0.0 deg.

## Steps 4–6 (done): pick task, household objects, handover

Full story in [PROCESS_LOG.md](PROCESS_LOG.md). In short: the gripper and its kinematics were measured rather than assumed, which found real problems in the openarm setup (inverted open/close, fingertips 0.17 m deep, convex-hull colliders, a dropped finger mimic). The robot placement and ready pose were re-chosen by reachability. Picks are planned exactly offline and executed in joint space.

| Task | Result (final code) |
|---|---|
| Pick cube (5 cm) | 10/10 |
| Pick soup can | 10/10 |
| Pick mustard bottle | 10/10 |
| Pick mug | 8/10 (2 orientations have no reachable grasp, reported as "no plan"; 8/8 executed) |
| Handover right → left (soup can) | 10/10 + 3/3 on video |

```bash
cd Issac-sim-teleop-eval   # repo root
PY=/data/isaac/isaacsim/bin/python
$PY scenes/run_pick.py --object cube --episodes 5          # objects: cube, soup_can, mustard, mug
$PY scenes/run_pick.py --object mug --episodes 10 --no-video --trace   # fast check with a per-step trace
$PY scenes/run_handover.py --episodes 3
$PY tools/episode_contact_sheet.py artifacts/episodes/<run> 0          # one-image summary of an episode
# tests
$PY scenes/robot_env_scene.py --headless --steps 120 && $PY tests/check_kinematics.py && $PY tests/check_gripper_contact.py
```

Each episode in `artifacts/episodes/<run>/`: `episode_XXXX.npz` (obs = measured, act = commanded, both arms: 7 joints, gripper, EE pose xyz + quat wxyz in the robot root frame, 30 Hz), one MP4 per camera (`cam_high`, `cam_right_wrist`, `cam_left_wrist`), and `summary.json`.

Code map: `scenes/layout.py` (placement constants) · `scenes/env_common.py` (scene, robot, checks) · `scenes/kinematics.py` (FK/IK, gripper model) · `scenes/pick_planner.py`, `scenes/handover_planner.py` (offline planners) · `scenes/pick_task.py` (objects, cameras, reset, observations) · `scenes/run_pick.py`, `scenes/run_handover.py` (episodes + recording) · `tools/` (asset conversion, workspace maps, pose solvers) · `tests/`.

## Next steps

See the end of [PROCESS_LOG.md](PROCESS_LOG.md).
