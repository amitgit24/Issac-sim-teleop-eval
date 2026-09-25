# LeRobot v2.1 export

`tools/export_lerobot.py` converts successful recorded NPZ episodes and their three camera MP4s into a local LeRobot v2.1 video dataset at 30 Hz. It writes `export_manifest.json` in the dataset root, mapping each source run/episode to its dataset episode index and recording episode/frame/skip counts. Failed episodes are excluded unless `--include-failed` is used. An episode missing any camera is skipped with a warning; an existing video whose decoded frame count differs from the NPZ length is an error.

| Feature | dtype | shape | dimension names |
| --- | --- | --- | --- |
| `observation.state` | float32 | `[16]` | right joints 1–7, right gripper, left joints 1–7, left gripper |
| `observation.ee_pose` | float32 | `[14]` | right then left `x,y,z,qw,qx,qy,qz` |
| `action` | float32 | `[16]` | right joints 1–7, right gripper, left joints 1–7, left gripper |
| `action.ee_pose` | float32 | `[14]` | right then left `x,y,z,qw,qx,qy,qz` |
| `observation.phase` | int64 | `[1]` | planner phase index |
| `observation.images.cam_high` | video | `[480,640,3]` | height, width, channels |
| `observation.images.cam_right_wrist` | video | `[480,640,3]` | height, width, channels |
| `observation.images.cam_left_wrist` | video | `[480,640,3]` | height, width, channels |

The state fields are measured before the action; action EE poses are the recorded exact FK of the commanded joints. Each frame also carries the NPZ `prompt` as its LeRobot task string.

From the repository root, export and validate with:

```bash
LD_LIBRARY_PATH=/home/user/ffmpeg-7.1/lib:${LD_LIBRARY_PATH:-} \
  /home/user/RoboTwin/policy/pi05/.venv/bin/python tools/export_lerobot.py \
  --runs artifacts/episodes/video_cube artifacts/episodes/video_soup_can \
         artifacts/episodes/video_mustard artifacts/episodes/video_mug \
         artifacts/episodes/video_handover \
  --out artifacts/datasets/robot_env_demo_v0 --repo-id local/robot_env_demo_v0 --overwrite

LD_LIBRARY_PATH=/home/user/ffmpeg-7.1/lib:${LD_LIBRARY_PATH:-} \
  /home/user/RoboTwin/policy/pi05/.venv/bin/python tools/validate_lerobot.py \
  --dataset artifacts/datasets/robot_env_demo_v0 \
  --runs artifacts/episodes/video_cube artifacts/episodes/video_soup_can \
         artifacts/episodes/video_mustard artifacts/episodes/video_mug \
         artifacts/episodes/video_handover
```

The tools require a Python environment with LeRobot v2.1 (`lerobot==0.1.0` in the verified environment), NumPy, and PyAV; they do not import Isaac Sim. The FFmpeg shared-library path is needed for PyAV, and `/home/user/ffmpeg-7.1/bin/ffmpeg` is the matching command-line binary for inspection. LeRobot's default SVT-AV1 encoder is used when available, with H.264 as a fallback.

The validator reloads the dataset with `LeRobotDataset`, requires exact float32 numeric values, phase indices, episode/frame counts, and task strings, and decodes both source and exported video. By default it compares every frame with a per-frame RGB mean-absolute-difference tolerance of 5.0 intensity levels, allowing normal lossy re-encoding. Use `--image-stride N` for a sampled image check or `--image-mad-tolerance X` to override the threshold.

**Decoder backend:** both tools load videos with `video_backend="pyav"`. LeRobot's default
`torchcodec` backend fails to load its native library in the verified environment
(`Could not load this library: .../libtorchcodec*.so`); pass `video_backend="pyav"` when loading these
datasets elsewhere too.

**Frame alignment (independent check, 2026-09-25):** on the 20 highest-motion frames of a handover
episode, every decoded frame was closer to its own source frame than to either neighbour (cam_high
20/20: 3.58 vs 4.67 mean abs diff; right wrist 20/20: 3.68 vs 9.97). The 5.0 image tolerance is lossy
re-encoding, not an off-by-one. Teleop episodes (`scenes/run_teleop.py`) export unchanged.
