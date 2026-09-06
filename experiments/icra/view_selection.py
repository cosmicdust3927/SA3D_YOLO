"""Deterministic camera ordering and exact-budget subsampling.

The geometry score follows the method used in the accompanying JEI manuscript:
rotation-angle distance plus ``alpha`` times translation distance, with an
exponentially decayed history term.  Camera IDs are used only to break numeric
ties, so results do not depend on NumPy's sort stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np


Method = Literal["filename", "random", "proposed"]


@dataclass(frozen=True)
class Selection:
    method: Method
    budget: int
    ordered_camera_ids: tuple[int, ...]
    selected_camera_ids: tuple[int, ...]
    selection_seed: int | None = None


def _validate_poses(poses: np.ndarray, camera_ids: np.ndarray) -> None:
    if poses.ndim != 3 or poses.shape[1] < 3 or poses.shape[2] < 4:
        raise ValueError("poses must have shape [N, 3, 4] or [N, 4, 4]")
    if len(poses) == 0 or len(poses) != len(camera_ids):
        raise ValueError("poses and camera_ids must contain the same cameras")
    if len(np.unique(camera_ids)) != len(camera_ids):
        raise ValueError("camera_ids must be unique")
    if not np.all(np.isfinite(poses)):
        raise ValueError("poses must contain only finite values")


def mean_pose(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    rotations = poses[:, :3, :3]
    mean_rotation_matrix = rotations.mean(axis=0)
    u, _, vt = np.linalg.svd(mean_rotation_matrix)
    correction = np.eye(3, dtype=np.float64)
    correction[2, 2] = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = u @ correction @ vt
    result[:3, 3] = poses[:, :3, 3].mean(axis=0)
    return result


def rotation_distance(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative_rotation = rotation_a.T @ rotation_b
    cosine = (np.trace(relative_rotation) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def pose_distance(pose_a: np.ndarray, pose_b: np.ndarray, alpha: float = 1.0) -> float:
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    rotation = rotation_distance(pose_a[:3, :3], pose_b[:3, :3])
    translation = np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3])
    return float(rotation + alpha * translation)


def _stable_extreme(
    values: np.ndarray,
    candidate_indices: np.ndarray,
    camera_ids: np.ndarray,
    mode: Literal["min", "max"],
) -> int:
    candidate_values = values[candidate_indices]
    extreme = np.min(candidate_values) if mode == "min" else np.max(candidate_values)
    tied = candidate_indices[
        np.isclose(candidate_values, extreme, rtol=1e-10, atol=1e-12)
    ]
    return int(tied[np.argmin(camera_ids[tied])])


def geometry_order(
    poses: np.ndarray,
    camera_ids: Iterable[int],
    *,
    alpha: float = 1.0,
    beta: float = 0.5,
    initial: Literal["mean", "farthest"] = "mean",
) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    camera_ids = np.asarray(tuple(camera_ids), dtype=np.int64)
    _validate_poses(poses, camera_ids)
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must satisfy 0 < beta < 1")
    if initial not in {"mean", "farthest"}:
        raise ValueError("initial must be 'mean' or 'farthest'")

    reference = mean_pose(poses)
    mean_distances = np.asarray(
        [pose_distance(pose, reference, alpha) for pose in poses]
    )
    start = _stable_extreme(
        mean_distances,
        np.arange(len(camera_ids)),
        camera_ids,
        "min" if initial == "mean" else "max",
    )

    distances = np.zeros((len(camera_ids), len(camera_ids)), dtype=np.float64)
    for first in range(len(camera_ids)):
        for second in range(first + 1, len(camera_ids)):
            value = pose_distance(poses[first], poses[second], alpha)
            distances[first, second] = value
            distances[second, first] = value

    remaining = np.ones(len(camera_ids), dtype=bool)
    remaining[start] = False
    ordered_indices = [start]
    weighted_scores = distances[:, start].copy()
    while remaining.any():
        candidates = np.flatnonzero(remaining)
        next_index = _stable_extreme(
            weighted_scores, candidates, camera_ids, "min"
        )
        ordered_indices.append(next_index)
        remaining[next_index] = False
        if remaining.any():
            # Equivalent to sum_j beta**(i-1-j) d(T_candidate, T_j).
            weighted_scores = beta * weighted_scores + distances[:, next_index]

    ordered = camera_ids[np.asarray(ordered_indices)]
    if len(np.unique(ordered)) != len(camera_ids):
        raise RuntimeError("geometry order must contain each camera exactly once")
    return ordered


def uniform_exact_budget(order: Iterable[int], budget: int) -> np.ndarray:
    """Select exactly ``budget`` nested, uniformly spaced order positions.

    The endpoint-free rule ``floor(k*N/K)`` generalizes manuscript stride
    sampling to non-divisor budgets such as 16 out of 24.  For the requested
    budgets 4, 8, and 16, every smaller selection is nested in the larger one.
    """

    order = np.asarray(tuple(order), dtype=np.int64)
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        raise TypeError("budget must be an integer")
    if not 1 <= int(budget) <= len(order):
        raise ValueError("budget must be between 1 and the number of cameras")
    positions = np.floor(np.arange(budget) * len(order) / budget).astype(np.int64)
    selected = order[positions]
    if len(np.unique(selected)) != budget:
        raise RuntimeError("exact-budget subsampling produced duplicate cameras")
    return selected


def select_views(
    method: Method,
    candidate_ids: Iterable[int],
    poses: np.ndarray,
    budget: int,
    *,
    selection_seed: int | None = None,
    alpha: float = 1.0,
    beta: float = 0.5,
    initial: Literal["mean", "farthest"] = "mean",
) -> Selection:
    candidate_ids = np.asarray(tuple(candidate_ids), dtype=np.int64)
    poses = np.asarray(poses, dtype=np.float64)
    _validate_poses(poses, candidate_ids)

    if method == "filename":
        order = np.sort(candidate_ids)
        seed = None
    elif method == "random":
        if selection_seed is None:
            raise ValueError("random selection requires selection_seed")
        order = np.random.default_rng(selection_seed).permutation(candidate_ids)
        seed = int(selection_seed)
    elif method == "proposed":
        order = geometry_order(
            poses, candidate_ids, alpha=alpha, beta=beta, initial=initial
        )
        seed = None
    else:
        raise ValueError(f"unsupported selection method: {method}")

    selected = uniform_exact_budget(order, budget)
    return Selection(
        method=method,
        budget=int(budget),
        ordered_camera_ids=tuple(int(value) for value in order),
        selected_camera_ids=tuple(int(value) for value in selected),
        selection_seed=seed,
    )

