"""Dataset validation and LLFF camera-ID mapping."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class SceneIndex:
    scene: str
    dataset_dir: Path
    camera_ids: tuple[int, ...]
    image_paths: tuple[Path, ...]
    poses: np.ndarray

    def indices_for(self, camera_ids: Iterable[int]) -> np.ndarray:
        lookup = {camera_id: index for index, camera_id in enumerate(self.camera_ids)}
        requested = tuple(int(value) for value in camera_ids)
        missing = sorted(set(requested) - set(lookup))
        if missing:
            raise ValueError(f"{self.scene}: missing camera IDs {missing}")
        return np.asarray([lookup[value] for value in requested], dtype=np.int64)

    def poses_for(self, camera_ids: Iterable[int]) -> np.ndarray:
        return self.poses[self.indices_for(camera_ids)]


def camera_id_from_name(path: Path, pattern: str) -> int:
    match = re.search(pattern, path.stem)
    if match is None:
        raise ValueError(f"cannot extract camera ID from {path.name!r} using {pattern!r}")
    value = match.groupdict().get("id") or match.group(0)
    return int(value)


def build_scene_index(
    scene: str,
    dataset_dir: str | Path,
    *,
    camera_id_pattern: str = r"(?P<id>\d+)(?!.*\d)",
) -> SceneIndex:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    image_dir = dataset_dir / "images"
    pose_path = dataset_dir / "poses_bounds.npy"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"{scene}: image directory not found: {image_dir}")
    if not pose_path.is_file():
        raise FileNotFoundError(f"{scene}: LLFF poses not found: {pose_path}")

    image_paths = tuple(
        sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    )
    camera_ids = tuple(camera_id_from_name(path, camera_id_pattern) for path in image_paths)
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError(f"{scene}: camera IDs parsed from filenames are not unique")

    poses_bounds = np.load(pose_path)
    if poses_bounds.ndim != 2 or poses_bounds.shape[1] < 17:
        raise ValueError(f"{scene}: unexpected poses_bounds.npy shape {poses_bounds.shape}")
    if len(poses_bounds) != len(image_paths):
        raise ValueError(
            f"{scene}: {len(image_paths)} images but {len(poses_bounds)} pose rows"
        )
    poses = poses_bounds[:, :-2].reshape((-1, 3, 5))[:, :, :4]
    return SceneIndex(scene, dataset_dir, camera_ids, image_paths, poses)


def validate_split(
    index: SceneIndex,
    candidate_ids: Iterable[int],
    test_ids: Iterable[int],
    *,
    expected_total: int = 32,
) -> None:
    candidates = tuple(int(value) for value in candidate_ids)
    tests = tuple(int(value) for value in test_ids)
    if len(candidates) != len(set(candidates)):
        raise ValueError("candidate camera IDs contain duplicates")
    if len(tests) != len(set(tests)):
        raise ValueError("test camera IDs contain duplicates")
    overlap = sorted(set(candidates) & set(tests))
    if overlap:
        raise ValueError(f"candidate and test splits overlap: {overlap}")
    union = set(candidates) | set(tests)
    if len(union) != expected_total:
        raise ValueError(f"split contains {len(union)} cameras; expected {expected_total}")
    if union != set(index.camera_ids):
        missing = sorted(set(index.camera_ids) - union)
        unknown = sorted(union - set(index.camera_ids))
        raise ValueError(f"split/file mismatch; missing={missing}, unknown={unknown}")

