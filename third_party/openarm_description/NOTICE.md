# Vendored: openarm_description (subset)

Source: https://github.com/enactic/openarm_description, commit `14ff67b638ff1c738a1b9a6be8aaa5ce5ed2c831`.
License: Apache License 2.0 (`LICENSE.txt`, unchanged).

Only the 22 mesh files referenced by `assets/openarm/urdf/openarm_bimanual.urdf` (bimanual OpenArm v2.0
with the pinch gripper) are included, unmodified, plus `package.xml` so `package://openarm_description`
resolves. The URDF was generated from this package's xacro (`openarm_v20.urdf.xacro`,
preset `default_bimanual`).
