# Engineering notes: how I built this

This page is the short version of how the project was built and why it works. The full, unedited
trail is in [PROCESS_LOG.md](PROCESS_LOG.md) (every step in order), [DECISIONS.md](DECISIONS.md)
(D1–D26, each with alternatives and evidence) and [MISTAKES.md](MISTAKES.md) (M1–M44: symptom → root
cause → fix → lesson).

## How I work

1. **Measure before trusting.** Constants inherited from an earlier OpenArm prototype were treated as
   hypotheses. The gripper, the collision shapes, the reachable workspace and every asset's units
   were measured in simulation before anything was built on them.
2. **Evidence over impressions.** Every claim is backed by a number, a render or a per-step trace.
   When an image suggested one thing (a can "tipping") and the trace showed another (no tilt at
   all), the trace won.
3. **Plan exactly, then execute.** Motions are planned offline with URDF kinematics that match the
   simulator to 0.000 mm, then tracked in joint space. No online IK drift.
4. **Gates, not hopes.** Nothing enters use without a check: assets pass a drop test, poses pass
   reachability searches, planners report `no plan` instead of attempting a bad grasp, and every
   headless run exits non-zero on failure.
5. **Write it down.** Each step logs what was tried, what failed and why the final choice was made,
   so the reasoning can be reviewed, not just the result.
6. **Honest status.** Results report exactly what was verified; unfinished work is marked as such.

## Case studies

### 1. The gripper was the opposite of what the code assumed
- **Symptom:** the earlier prototype's scripted pick succeeded 12.5 % of the time, "stochastically".
- **Investigation:** rendered the fingers at q = 0 / −0.4 / −0.785 and measured the finger meshes in
  the gripper frame (CPU physics with USD updates, since GPU physics doesn't write back poses).
- **Root cause:** q = 0 is **closed**, not open, so the old code approached with a closed gripper.
  The fingertips are 0.15–0.17 m below the gripper frame (assumed 0.05–0.07 m), and the claw pocket
  is only ~4 cm deep.
- **Result:** with the measured model, picks went to **10/10** (cube, soup can, mustard).
  → M15, M16, M23.

### 2. Grasps that "worked" for the wrong reason
- **Symptom:** the fingers closed to ~0 rad while "holding" a 5 cm cube, which is physically impossible.
- **Root cause (three layers):** convex-hull finger colliders filled in the claw (grasps came from
  overlapping hulls); the URDF's finger mimic joint was dropped by the importer (an option whose name
  means the opposite of what it does); once restored, the mimic was a soft spring that yielded under
  contact.
- **Fix:** own URDF → USD conversion with convex decomposition, mimic kept and made rigid,
  force-limited closing (3 N·m), plus a permanent contact test (both fingers must stop on a fixed box).
- **Result:** symmetric finger contact on every object (e.g. −0.355 / −0.371 rad on the 6.8 cm can).
  → M22, M24–M27, D12, D13.

### 3. Robot placement chosen by reachability, not by eye
- **Symptom:** the first placement left almost no pickable table area.
- **Approach:** an offline workspace map (grasp, straight descent, lift, IK continuity, joint margins)
  over mounting heights and edge gaps.
- **Result:** pickable cells went from **8 → 21**; the ready pose's joint margin from 5.9 % → 15.2 %.
  → M21, M29, D10, D11.

### 4. Assets that shrink 100× after a physics reset
- **Symptom:** a table (and later a glass) was the right size after spawning and 1 cm tall after
  `sim.reset()`.
- **Root cause:** centimetre-authored assets get a unit correction; editing any prim inside them
  re-triggers it.
- **Fix:** never edit inside such assets: invisible proxy colliders for the table, and a
  visual/collision split wrapper for props. Every inventory item is now gated by a drop test that
  checks size after reset (**21/21 pass**). → M8, M39, D4, D22.

### 5. A handover that needed a different idea, not more tuning
- **First design:** turn the can horizontal, then grip it from above with the other hand. It failed
  three different ways (unreachable wrist orientations, a wedge-shaped pinch, gripping too far from
  the center of mass).
- **Redesign:** keep the can upright; the right hand holds the top, the left wraps the lower body
  from the side, so the two hands use disjoint heights.
- **Result:** **10/10**, then generalized to other objects (glasses 5/5, mustard 5/5, mug 4/5).
  → M31–M34, M43, D19.

### 6. Teleop that doesn't get stuck
- **Symptom:** live step-by-step IK stopped short of the table (an IK branch boundary; joint 6 is
  limited to ±45°).
- **Tried:** a lower home pose (16/36 reach tasks) and IK joint-centering (worse: 3/36).
- **Fix:** automatic reconfiguration: find another IK solution and move there only if the transition
  keeps the fingertips ≥ 3 cm above the table.
- **Result:** 18 of the 25 physically reachable teleop tasks, with limits clearly reported to the
  operator. → M35, M36, D21.

## Verification at a glance

| Check | Where | Result |
|---|---|---|
| FK vs simulator, 40 random configurations | `tests/check_kinematics.py` | 0.000 mm |
| Both fingers stop on a fixed box | `tests/check_gripper_contact.py` | pass |
| Teleop key / button mapping (injected events) | `tests/check_teleop_inputs.py` | 28/28 |
| Unit tests (planners, kinematics, teleop safety, clutter, inventory) | `tests/unit`, CI | 18/18 |
| Picks / pick-and-place / handover | `scenes/run_*.py` | see the README results table |
| Dataset export (exact values + frame alignment) | `tools/validate_lerobot.py` | pass |
