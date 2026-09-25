#!/usr/bin/env python3
"""Export recorded Robot_env episodes as a local LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


FPS = 30
CAMERAS = ("cam_high", "cam_right_wrist", "cam_left_wrist")
IMAGE_PREFIX = "observation.images."
STATE_NAMES = [
    *(f"right_joint{i}" for i in range(1, 8)),
    "right_gripper",
    *(f"left_joint{i}" for i in range(1, 8)),
    "left_gripper",
]
EE_NAMES = [
    *(f"right_{name}" for name in ("x", "y", "z", "qw", "qx", "qy", "qz")),
    *(f"left_{name}" for name in ("x", "y", "z", "qw", "qx", "qy", "qz")),
]
ARRAY_SHAPES = {
    "obs_right_joint_pos": (7,),
    "obs_right_gripper": (),
    "obs_right_ee_pose": (7,),
    "obs_left_joint_pos": (7,),
    "obs_left_gripper": (),
    "obs_left_ee_pose": (7,),
    "act_right_joint_pos": (7,),
    "act_right_gripper": (),
    "act_right_ee_pose": (7,),
    "act_left_joint_pos": (7,),
    "act_left_gripper": (),
    "act_left_ee_pose": (7,),
    "phase": (),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="source run directories")
    parser.add_argument("--out", type=Path, required=True, help="local LeRobot dataset directory")
    parser.add_argument("--repo-id", required=True, help="LeRobot repository id, for example local/demo")
    parser.add_argument("--include-failed", action="store_true", help="also export episodes marked failed")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output directory")
    return parser.parse_args()


def import_dependencies():
    try:
        import av
        import numpy as np
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module
        from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
        from lerobot.common.datasets.video_utils import encode_video_frames
    except ImportError as exc:
        if "libavformat" in str(exc) or "libavcodec" in str(exc) or "libavutil" in str(exc):
            raise RuntimeError(
                "FFmpeg shared libraries could not be loaded. Set LD_LIBRARY_PATH to the FFmpeg "
                "lib directory (for example <ffmpeg-prefix>/lib:$LD_LIBRARY_PATH) and retry."
            ) from exc
        raise RuntimeError(
            "LeRobot export dependencies are unavailable. Run this tool with a Python environment "
            "that provides lerobot v2.1, NumPy, and PyAV."
        ) from exc
    return av, np, lerobot_dataset_module, LeRobotDataset, CODEBASE_VERSION, encode_video_frames


def features() -> dict[str, dict]:
    result = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
        "observation.ee_pose": {"dtype": "float32", "shape": (14,), "names": EE_NAMES},
        "action": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
        "action.ee_pose": {"dtype": "float32", "shape": (14,), "names": EE_NAMES},
        "observation.phase": {"dtype": "int64", "shape": (1,), "names": ["phase"]},
    }
    for camera in CAMERAS:
        result[f"{IMAGE_PREFIX}{camera}"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }
    return result


def episode_number(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("episode_"))
    except ValueError as exc:
        raise ValueError(f"invalid episode filename: {path}") from exc


def validate_npz(path: Path, np):
    data = np.load(path, allow_pickle=False)
    missing = (set(ARRAY_SHAPES) | {"success", "prompt"}) - set(data.files)
    if missing:
        data.close()
        raise ValueError(f"{path}: missing arrays {sorted(missing)}")

    phase = data["phase"]
    if phase.ndim != 1 or len(phase) == 0:
        data.close()
        raise ValueError(f"{path}: phase must have non-empty shape (T,), got {phase.shape}")
    frame_count = len(phase)
    for key, trailing_shape in ARRAY_SHAPES.items():
        value = data[key]
        expected = (frame_count, *trailing_shape)
        if value.shape != expected:
            data.close()
            raise ValueError(f"{path}: {key} has shape {value.shape}, expected {expected}")
        if value.dtype != np.float32:
            data.close()
            raise ValueError(f"{path}: {key} has dtype {value.dtype}, expected float32")
    if not np.isfinite(data["phase"]).all() or not np.equal(data["phase"], np.floor(data["phase"])).all():
        data.close()
        raise ValueError(f"{path}: phase values must be finite integer indices")
    if data["success"].shape != () or data["prompt"].shape != ():
        data.close()
        raise ValueError(f"{path}: success and prompt must be scalars")
    return data, frame_count, bool(data["success"].item()), str(data["prompt"].item())


def inspect_video(path: Path, expected_frames: int, av) -> None:
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if (stream.width, stream.height) != (640, 480):
                raise ValueError(
                    f"{path}: video is {stream.width}x{stream.height}, expected 640x480"
                )
            count = sum(1 for _ in container.decode(stream))
    except (OSError, IndexError) as exc:
        raise ValueError(f"{path}: cannot decode video: {exc}") from exc
    if count != expected_frames:
        raise ValueError(f"{path}: video has {count} frames, source npz has {expected_frames}")


def select_episodes(runs: list[Path], include_failed: bool, av, np):
    selected = []
    skipped_failed = 0
    skipped_no_video = 0
    for run_index, run in enumerate(runs):
        if not run.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {run}")
        if not (run / "summary.json").is_file():
            raise FileNotFoundError(f"run has no summary.json: {run}")
        npz_paths = sorted(run.glob("episode_[0-9][0-9][0-9][0-9].npz"), key=episode_number)
        if not npz_paths:
            raise ValueError(f"run has no episode npz files: {run}")
        for npz_path in npz_paths:
            data, frame_count, success, prompt = validate_npz(npz_path, np)
            data.close()
            if not success and not include_failed:
                skipped_failed += 1
                continue
            video_paths = {camera: run / f"{npz_path.stem}_{camera}.mp4" for camera in CAMERAS}
            missing = [path for path in video_paths.values() if not path.is_file()]
            if missing:
                skipped_no_video += 1
                print(
                    f"WARNING: skipping {npz_path}: missing video(s): "
                    + ", ".join(path.name for path in missing),
                    file=sys.stderr,
                )
                continue
            for video_path in video_paths.values():
                inspect_video(video_path, frame_count, av)
            selected.append(
                {
                    "run_index": run_index,
                    "run": run,
                    "npz": npz_path,
                    "videos": video_paths,
                    "frames": frame_count,
                    "success": success,
                    "prompt": prompt,
                }
            )
    if not selected:
        raise ValueError("no episodes qualified for export")
    return selected, skipped_failed, skipped_no_video


def choose_video_codec(av) -> str:
    for codec in ("libsvtav1", "h264"):
        try:
            av.codec.Codec(codec, "w")
            return codec
        except Exception:
            pass
    raise RuntimeError("neither the libsvtav1 nor h264 video encoder is available through PyAV")


def install_codec_override(module, encoder, codec: str) -> None:
    if codec == "libsvtav1":
        return

    def encode_with_available_codec(imgs_dir, video_path, fps, **kwargs):
        return encoder(imgs_dir, video_path, fps, vcodec=codec, **kwargs)

    module.encode_video_frames = encode_with_available_codec


def vector(data, prefix: str, index: int, np):
    return np.concatenate(
        (
            data[f"{prefix}_right_joint_pos"][index],
            np.asarray([data[f"{prefix}_right_gripper"][index]], dtype=np.float32),
            data[f"{prefix}_left_joint_pos"][index],
            np.asarray([data[f"{prefix}_left_gripper"][index]], dtype=np.float32),
        )
    ).astype(np.float32, copy=False)


def ee_pose(data, prefix: str, index: int, np):
    return np.concatenate(
        (data[f"{prefix}_right_ee_pose"][index], data[f"{prefix}_left_ee_pose"][index])
    ).astype(np.float32, copy=False)


def add_episode(dataset, episode: dict, av, np) -> None:
    data = np.load(episode["npz"], allow_pickle=False)
    containers = []
    try:
        iterators = {}
        for camera, path in episode["videos"].items():
            container = av.open(str(path))
            containers.append(container)
            iterators[camera] = iter(container.decode(container.streams.video[0]))

        for index in range(episode["frames"]):
            frame = {
                "observation.state": vector(data, "obs", index, np),
                "observation.ee_pose": ee_pose(data, "obs", index, np),
                "action": vector(data, "act", index, np),
                "action.ee_pose": ee_pose(data, "act", index, np),
                "observation.phase": np.asarray([data["phase"][index]], dtype=np.int64),
                "task": episode["prompt"],
            }
            for camera, iterator in iterators.items():
                try:
                    decoded = next(iterator)
                except StopIteration as exc:
                    raise ValueError(f"{episode['videos'][camera]} ended before frame {index}") from exc
                frame[f"{IMAGE_PREFIX}{camera}"] = decoded.to_ndarray(format="rgb24")
            dataset.add_frame(frame)
        dataset.save_episode()
    finally:
        data.close()
        for container in containers:
            container.close()


def safe_prepare_output(out: Path, overwrite: bool) -> None:
    if not out.exists():
        return
    if not overwrite:
        raise FileExistsError(f"output already exists (pass --overwrite to replace it): {out}")
    resolved = out.resolve()
    if resolved in {Path.cwd().resolve(), Path(resolved.anchor)}:
        raise ValueError(f"refusing to overwrite unsafe output path: {out}")
    if out.is_dir():
        shutil.rmtree(out)
    else:
        out.unlink()


def run(args: argparse.Namespace) -> None:
    av, np, module, LeRobotDataset, codebase_version, encoder = import_dependencies()
    if codebase_version != "v2.1":
        raise RuntimeError(f"this exporter requires LeRobot v2.1, found {codebase_version}")

    runs = [path.resolve() for path in args.runs]
    if len(set(runs)) != len(runs):
        raise ValueError("--runs contains a duplicate directory")
    selected, skipped_failed, skipped_no_video = select_episodes(
        runs, args.include_failed, av, np
    )
    codec = choose_video_codec(av)
    install_codec_override(module, encoder, codec)
    safe_prepare_output(args.out, args.overwrite)

    dataset = LeRobotDataset.create(
        args.repo_id,
        fps=FPS,
        root=args.out,
        features=features(),
        use_videos=True,
        image_writer_threads=4,
        video_backend="pyav",
    )
    manifest_episodes = []
    total_frames = 0
    try:
        for dataset_index, episode in enumerate(selected):
            print(
                f"EXPORT episode={dataset_index} source={episode['npz']} frames={episode['frames']}",
                flush=True,
            )
            add_episode(dataset, episode, av, np)
            total_frames += episode["frames"]
            manifest_episodes.append(
                {
                    "source_run_index": episode["run_index"],
                    "source_run": args.runs[episode["run_index"]].as_posix(),
                    "source_episode": episode["npz"].name,
                    "dataset_episode_index": dataset_index,
                    "frames": episode["frames"],
                    "success": episode["success"],
                    "task": episode["prompt"],
                }
            )
    finally:
        dataset.stop_image_writer()

    manifest = {
        "format": "robot_env_lerobot_export_manifest",
        "format_version": 1,
        "lerobot_codebase_version": codebase_version,
        "repo_id": args.repo_id,
        "fps": FPS,
        "video_codec": codec,
        "include_failed": args.include_failed,
        "runs": [path.as_posix() for path in args.runs],
        "episodes": manifest_episodes,
        "counts": {
            "runs": len(args.runs),
            "episodes": len(manifest_episodes),
            "frames": total_frames,
            "skipped_failed": skipped_failed,
            "skipped_no_video": skipped_no_video,
        },
    }
    (args.out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"LEROBOT EXPORT COMPLETE episodes={len(manifest_episodes)} frames={total_frames} "
        f"output={args.out}"
    )


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"LEROBOT EXPORT FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
