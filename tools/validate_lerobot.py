#!/usr/bin/env python3
"""Validate a Robot_env LeRobot export against its source NPZ and MP4 files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CAMERAS = ("cam_high", "cam_right_wrist", "cam_left_wrist")
IMAGE_PREFIX = "observation.images."
DEFAULT_IMAGE_MAD = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="local LeRobot dataset directory")
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="original run directories")
    parser.add_argument(
        "--image-stride",
        type=int,
        default=1,
        help="compare every Nth video frame (default: every frame)",
    )
    parser.add_argument(
        "--image-mad-tolerance",
        type=float,
        default=DEFAULT_IMAGE_MAD,
        help="maximum per-frame mean absolute RGB difference in 0..255 units (default: 5.0)",
    )
    return parser.parse_args()


def import_dependencies():
    try:
        import av
        import numpy as np
        from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    except ImportError as exc:
        if "libavformat" in str(exc) or "libavcodec" in str(exc) or "libavutil" in str(exc):
            raise RuntimeError(
                "FFmpeg shared libraries could not be loaded. Set LD_LIBRARY_PATH to the FFmpeg "
                "lib directory (for example <ffmpeg-prefix>/lib:$LD_LIBRARY_PATH) and retry."
            ) from exc
        raise RuntimeError(
            "LeRobot validation dependencies are unavailable. Run this tool with a Python environment "
            "that provides lerobot v2.1, NumPy, and PyAV."
        ) from exc
    return av, np, LeRobotDataset, CODEBASE_VERSION


def expected_features(export_module) -> dict[str, dict]:
    return export_module.features()


def source_vector(data, prefix: str, np):
    return np.concatenate(
        (
            data[f"{prefix}_right_joint_pos"],
            data[f"{prefix}_right_gripper"][:, None],
            data[f"{prefix}_left_joint_pos"],
            data[f"{prefix}_left_gripper"][:, None],
        ),
        axis=1,
    ).astype(np.float32, copy=False)


def source_ee_pose(data, prefix: str, np):
    return np.concatenate(
        (data[f"{prefix}_right_ee_pose"], data[f"{prefix}_left_ee_pose"]), axis=1
    ).astype(np.float32, copy=False)


def as_numpy(value, np):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def first_array_mismatch(name: str, actual, expected, np) -> str | None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return f"{name}: shape {actual.shape} != {expected.shape}"
    if actual.dtype != expected.dtype:
        return f"{name}: dtype {actual.dtype} != {expected.dtype}"
    unequal = np.argwhere(actual != expected)
    if unequal.size:
        index = tuple(int(i) for i in unequal[0])
        return f"{name}{index}: dataset={actual[index]!r} source={expected[index]!r}"
    return None


def load_manifest(dataset_path: Path) -> dict:
    path = dataset_path / "export_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"dataset has no export_manifest.json: {dataset_path}")
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "robot_env_lerobot_export_manifest" or manifest.get("format_version") != 1:
        raise ValueError(f"unsupported export manifest: {path}")
    return manifest


def validate_schema(dataset, expected: dict, manifest: dict) -> None:
    if dataset.fps != 30:
        raise ValueError(f"dataset fps is {dataset.fps}, expected 30")
    for key, feature in expected.items():
        if key not in dataset.features:
            raise ValueError(f"dataset is missing feature {key}")
        actual = dataset.features[key]
        if actual["dtype"] != feature["dtype"]:
            raise ValueError(f"{key}: dtype {actual['dtype']} != {feature['dtype']}")
        if tuple(actual["shape"]) != tuple(feature["shape"]):
            raise ValueError(f"{key}: shape {actual['shape']} != {feature['shape']}")
        if actual["names"] != feature["names"]:
            raise ValueError(f"{key}: names {actual['names']} != {feature['names']}")
    if dataset.num_episodes != len(manifest["episodes"]):
        raise ValueError(
            f"episode count {dataset.num_episodes} != manifest {len(manifest['episodes'])}"
        )
    expected_frames = sum(int(episode["frames"]) for episode in manifest["episodes"])
    if dataset.num_frames != expected_frames:
        raise ValueError(f"frame count {dataset.num_frames} != manifest {expected_frames}")


def resolve_source(entry: dict, runs: list[Path]) -> Path:
    run_index = entry.get("source_run_index")
    if not isinstance(run_index, int) or not 0 <= run_index < len(runs):
        raise ValueError(f"invalid source_run_index in manifest: {run_index!r}")
    return runs[run_index] / entry["source_episode"]


def compare_numeric(dataset, manifest: dict, runs: list[Path], np) -> None:
    offset = 0
    for expected_episode_index, entry in enumerate(manifest["episodes"]):
        if entry["dataset_episode_index"] != expected_episode_index:
            raise ValueError(
                f"manifest dataset episode index {entry['dataset_episode_index']} != {expected_episode_index}"
            )
        source_path = resolve_source(entry, runs)
        if not source_path.is_file():
            raise FileNotFoundError(f"source episode does not exist: {source_path}")
        with np.load(source_path, allow_pickle=False) as source:
            frame_count = len(source["phase"])
            if frame_count != entry["frames"]:
                raise ValueError(
                    f"{source_path}: frame count {frame_count} != manifest {entry['frames']}"
                )
            prompt = str(source["prompt"].item())
            if prompt != entry["task"]:
                raise ValueError(f"{source_path}: task differs from manifest")
            expected_arrays = {
                "observation.state": source_vector(source, "obs", np),
                "observation.ee_pose": source_ee_pose(source, "obs", np),
                "action": source_vector(source, "act", np),
                "action.ee_pose": source_ee_pose(source, "act", np),
                "observation.phase": source["phase"].astype(np.int64),
            }
            actual_arrays = {}
            rows = dataset.hf_dataset.select(range(offset, offset + frame_count))
            for key, expected in expected_arrays.items():
                values = rows[key]
                if key == "observation.phase":
                    actual = np.asarray([int(as_numpy(value, np)) for value in values], dtype=np.int64)
                else:
                    actual = np.stack([as_numpy(value, np) for value in values]).astype(
                        np.float32, copy=False
                    )
                actual_arrays[key] = actual
                mismatch = first_array_mismatch(
                    f"episode {expected_episode_index} {key}", actual, expected, np
                )
                if mismatch:
                    raise ValueError(mismatch)

            episode_indices = [int(as_numpy(value, np)) for value in rows["episode_index"]]
            if episode_indices != [expected_episode_index] * frame_count:
                raise ValueError(f"episode {expected_episode_index}: episode_index column mismatch")
            task_indices = [int(as_numpy(value, np)) for value in rows["task_index"]]
            for local_index, task_index in enumerate(task_indices):
                task = dataset.meta.tasks[task_index]
                if task != prompt:
                    raise ValueError(
                        f"episode {expected_episode_index} frame {local_index}: task {task!r} != {prompt!r}"
                    )
        offset += frame_count


def decoded_frames(path: Path, av):
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                yield frame.to_ndarray(format="rgb24")
    except (OSError, IndexError) as exc:
        raise ValueError(f"cannot decode {path}: {exc}") from exc


def compare_video_pair(
    source_path: Path,
    dataset_path: Path,
    expected_frames: int,
    stride: int,
    tolerance: float,
    av,
    np,
) -> tuple[int, float]:
    source_iter = decoded_frames(source_path, av)
    dataset_iter = decoded_frames(dataset_path, av)
    compared = 0
    worst = 0.0
    for index in range(expected_frames):
        try:
            source_frame = next(source_iter)
        except StopIteration as exc:
            raise ValueError(f"{source_path}: ended before frame {index}") from exc
        try:
            dataset_frame = next(dataset_iter)
        except StopIteration as exc:
            raise ValueError(f"{dataset_path}: ended before frame {index}") from exc
        if source_frame.shape != dataset_frame.shape:
            raise ValueError(
                f"{dataset_path} frame {index}: shape {dataset_frame.shape} != {source_frame.shape}"
            )
        if index % stride == 0:
            mad = float(
                np.abs(dataset_frame.astype(np.int16) - source_frame.astype(np.int16)).mean()
            )
            compared += 1
            worst = max(worst, mad)
            if mad > tolerance:
                raise ValueError(
                    f"{dataset_path} frame {index}: image mean-abs-diff {mad:.4f} > {tolerance:.4f}"
                )
    try:
        next(source_iter)
        raise ValueError(f"{source_path}: has more than {expected_frames} frames")
    except StopIteration:
        pass
    try:
        next(dataset_iter)
        raise ValueError(f"{dataset_path}: has more than {expected_frames} frames")
    except StopIteration:
        pass
    return compared, worst


def compare_images(dataset, manifest: dict, runs: list[Path], stride: int, tolerance: float, av, np):
    compared = 0
    worst = 0.0
    for entry in manifest["episodes"]:
        source_npz = resolve_source(entry, runs)
        stem = source_npz.stem
        episode_index = entry["dataset_episode_index"]
        for camera in CAMERAS:
            source_video = source_npz.parent / f"{stem}_{camera}.mp4"
            dataset_video = dataset.root / dataset.meta.get_video_file_path(
                episode_index, f"{IMAGE_PREFIX}{camera}"
            )
            n, pair_worst = compare_video_pair(
                source_video,
                dataset_video,
                int(entry["frames"]),
                stride,
                tolerance,
                av,
                np,
            )
            compared += n
            worst = max(worst, pair_worst)
    return compared, worst


def run(args: argparse.Namespace) -> None:
    if args.image_stride < 1:
        raise ValueError("--image-stride must be at least 1")
    if args.image_mad_tolerance < 0:
        raise ValueError("--image-mad-tolerance must be non-negative")

    av, np, LeRobotDataset, codebase_version = import_dependencies()
    if codebase_version != "v2.1":
        raise RuntimeError(f"this validator requires LeRobot v2.1, found {codebase_version}")

    # Import the exporter locally so the schema has one authoritative definition.
    import export_lerobot

    manifest = load_manifest(args.dataset)
    runs = [path.resolve() for path in args.runs]
    if len(runs) != len(manifest.get("runs", [])):
        raise ValueError(f"--runs count {len(runs)} != manifest {len(manifest.get('runs', []))}")
    dataset = LeRobotDataset(
        repo_id=manifest["repo_id"], root=args.dataset, video_backend="pyav"
    )
    validate_schema(dataset, expected_features(export_lerobot), manifest)
    compare_numeric(dataset, manifest, runs, np)
    image_frames, worst_mad = compare_images(
        dataset,
        manifest,
        runs,
        args.image_stride,
        args.image_mad_tolerance,
        av,
        np,
    )
    print(
        f"IMAGE CHECK PASSED comparisons={image_frames} worst_mean_abs_diff={worst_mad:.4f} "
        f"tolerance={args.image_mad_tolerance:.4f}"
    )
    print(
        f"LEROBOT VALIDATION PASSED episodes={dataset.num_episodes} frames={dataset.num_frames}"
    )


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"LEROBOT VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
