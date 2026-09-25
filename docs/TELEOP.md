# Teleoperation (keyboard / gamepad)

`scenes/run_teleop.py` opens the scene in the Isaac Sim GUI and lets you drive either arm's gripper
in the world frame, seen from the robot: **+x forward** across the table, **+y to the robot's
left**, **+z up**. Episodes are recorded in the same format as the scripted demos, so
`tools/export_lerobot.py` exports them unchanged.

```bash
PY=/path/to/isaacsim/bin/python
$PY scenes/run_teleop.py                              # GUI, keyboard + gamepad, cube
$PY scenes/run_teleop.py --object mug --device gamepad
$PY scenes/run_teleop.py --headless --script pick_cube --no-jitter   # automated end-to-end check
```

## Controls

| Action | Keyboard | Gamepad (Xbox layout) |
|---|---|---|
| Forward / back (x) | W / S (hold) | left stick up / down |
| Left / right (y) | A / D | left stick left / right |
| Up / down (z) | Q / E | right stick up / down |
| Yaw (turn gripper) | J / L | right stick left / right |
| Pitch | I / K | D-pad up / down |
| Roll | U / O | D-pad left / right |
| Gripper open / close | SPACE | A |
| Switch active arm | TAB | B |
| Precision mode (30 % speed) | P | X |
| Active arm to ready pose | H | Y |
| Start recording | R | Start |
| Finish and save | F | Start (again) |
| Discard recording | X | Back |
| New episode (re-place object, arms to ready) | N | RB |
| Quit | ESC | — |

Speeds at full deflection: 0.15 m/s and 0.8 rad/s. Sticks have a 0.12 dead zone. Closing the gripper
commands it fully closed; the 3 N·m finger effort cap sets the squeeze (D13).

## Safety and behaviour

- **Workspace clamp:** the target stays over the table, with the gripper frame 5–60 cm above the
  tabletop, and the **fingertips can never be commanded below the tabletop**.
- **Unreachable poses:** the arm holds its last reachable pose and the status line shows `LIMIT`.
  Pushing into a limit never makes the arm jump.
- **Automatic reconfiguration:** the arm's reachable space is split across IK branches (joint6 is
  limited to ±45°). When a move hits a branch boundary, the controller finds another joint
  configuration for the requested pose and, if the joint-space transition keeps the fingertips ≥ 3 cm
  above the table, moves there smoothly (~1 s, input paused, status `RECONFIGURING`).
- A status line prints every 0.5 s: active arm, gripper position (relative to the table), gripper
  state, precision, recording frame count, LIMIT / RECONFIGURING.

## Verified

- `tests/check_teleop_inputs.py`: real carb keyboard events and a virtual (connected) gamepad are
  injected headless; all 28 mappings pass (held keys, one-shot taps, dead zone, Start = record/save).
- `tests/check_teleop_reach.py` (offline, controller only): from the ready pose, turn the gripper
  (0/±45/90°), move over each pick-zone cell, descend to grasp height and lift. 25 of the 36 tasks
  are physically reachable at all; the controller completes **18/25** of those. The misses need a
  reconfiguration whose transition would dip the fingertips too low, so it is refused: approach from
  a different yaw, turn after descending, or press H and try again.
- `scenes/run_teleop.py --headless --script pick_cube --no-jitter`: the scripted operator (same
  controller/IK/recording path) turns −65°, descends (one automatic reconfiguration), grasps and
  lifts the cube 0.119 m; the episode saves with 3 camera videos and passes the LeRobot validator.
- GUI launch on a display with live keyboard + gamepad subscriptions: runs and exits cleanly.
  A human operator session has not been recorded yet.
