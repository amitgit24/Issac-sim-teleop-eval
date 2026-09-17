"""Scene layout constants shared by the sim scripts and the offline tools (no Isaac imports).

Every placement number lives here so the scene, the kinematics tools and the demonstrator can
never disagree. After changing the table or robot placement, re-run
tools/map_pick_workspace.py and tools/solve_ready_pose.py (docs/DECISIONS.md D6, D7).
"""

# Table (ArchVis EastRural_Table, see env_common.py), centered at the world origin.
TABLE_POS = (0.0, 0.0, 0.0)
TABLE_TOP_SIZE = (0.914, 1.528, 0.04)  # x, y, thickness of the collision slab (m)
TABLE_HEIGHT = 0.762  # top surface height (m)
TABLE_NEAR_X = TABLE_POS[0] - TABLE_TOP_SIZE[0] / 2.0  # the long edge the robot stands at

# Robot mounting. OpenArm's shoulders are 0.698 m above its base. The placement was chosen by
# reachability, not by eye (tools/map_pick_workspace.py): with the fingertips 0.15-0.17 m below the
# gripper frame and joint6 limited to +-45 deg, a fingers-down grasp needs the forearm pointing
# steeply down. Shoulders 0.40 m above the table with the base 0.05 m behind the edge gave the most
# pickable table cells (21, vs 8 for the earlier 0.35 m / 0.12 m guess, 0 for >= 0.50 m).
# The body column is +-0.03 m deep at tabletop height, so the 0.05 m gap clears it.
OPENARM_SHOULDER_Z = 0.698
SHOULDER_ABOVE_TABLE = 0.40
EDGE_GAP = 0.05
ROBOT_POS = (TABLE_NEAR_X - EDGE_GAP, TABLE_POS[1], TABLE_HEIGHT + SHOULDER_ABOVE_TABLE - OPENARM_SHOULDER_Z)
PEDESTAL_SIZE = (0.30, 0.40, ROBOT_POS[2])

# Right-arm pick zone: where every sampled cube yaw is pickable in the workspace map.
# Objects spawn at PICK_CENTER + uniform jitter of +-PICK_JITTER in x and y, random yaw.
PICK_CENTER = (TABLE_NEAR_X + 0.15, -0.30)
PICK_JITTER = 0.03

# Ready pose: grippers pointing down, open, above each arm's side of the pick area, high enough
# that the fingertips (<= 0.171 m below the gripper frame) clear the tallest object (mustard bottle,
# 0.191 m) by 4 cm: at 0.30 m the claws hit the bottle at reset (docs/MISTAKES.md M29).
# Inset/y chosen by the solver for joint margin (0.10/0.25: 5.9 %, 0.15/0.25: 15.2 %).
# Joint values from tools/solve_ready_pose.py (the wrist yaw is chosen for the widest
# joint-limit margin). Left arm is the mirror image of the right.
READY_INSET = 0.15
READY_Y = 0.25
READY_ABOVE_TABLE = 0.40
READY_RIGHT_ARM = (-0.3352, 2.2832, -0.0703, 2.0730, -0.0112, -0.5278, 0.7554)  # yaw +155, margin 15.2%
READY_LEFT_ARM = (0.3352, -2.2832, 0.0703, 2.0730, 0.0112, 0.5278, -0.7555)
