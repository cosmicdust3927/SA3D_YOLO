"""Strict scene-level multi-instance segmentation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.optimize import linear_sum_assignment


MaskSet = Mapping[str, Mapping[int, np.ndarray]]


@dataclass(frozen=True)
class InstanceMetrics:
    scene_instance_miou: float
    foreground_iou: float
    boundary_f1: float
    failure_rate: float
    num_predictions: int
    num_ground_truth: int
    num_matched: int
    matched_ious: tuple[float, ...]
    matching: tuple[tuple[str, str, float], ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _binary(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    return array > 0


def _resize(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    binary = _binary(mask)
    if binary.shape != shape:
        from PIL import Image

        binary = np.asarray(
            Image.fromarray(binary.astype(np.uint8)).resize(
                (shape[1], shape[0]), resample=Image.Resampling.NEAREST
            )
        ).astype(bool)
    return binary


def _counts_across_views(
    prediction: Mapping[int, np.ndarray],
    ground_truth: Mapping[int, np.ndarray],
    view_ids: Sequence[int],
) -> tuple[int, int]:
    intersection = 0
    union = 0
    for view_id in view_ids:
        gt = ground_truth.get(view_id)
        pred = prediction.get(view_id)
        if gt is None and pred is None:
            continue
        reference = _binary(gt if gt is not None else pred)
        gt_binary = np.zeros_like(reference) if gt is None else _resize(gt, reference.shape)
        pred_binary = np.zeros_like(reference) if pred is None else _resize(pred, reference.shape)
        intersection += int(np.logical_and(pred_binary, gt_binary).sum())
        union += int(np.logical_or(pred_binary, gt_binary).sum())
    return intersection, union


def instance_iou(
    prediction: Mapping[int, np.ndarray],
    ground_truth: Mapping[int, np.ndarray],
    view_ids: Sequence[int],
) -> float:
    intersection, union = _counts_across_views(prediction, ground_truth, view_ids)
    return float(intersection / union) if union else 0.0


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = _binary(mask)
    eroded = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    return np.logical_and(mask, np.logical_not(eroded))


def boundary_f1_pair(
    prediction: Mapping[int, np.ndarray],
    ground_truth: Mapping[int, np.ndarray],
    view_ids: Sequence[int],
    tolerance_pixels: int = 2,
) -> float:
    if tolerance_pixels < 0:
        raise ValueError("tolerance_pixels must be non-negative")
    matched_pred = matched_gt = pred_total = gt_total = 0
    kernel = np.ones((3, 3), dtype=bool)
    for view_id in view_ids:
        gt = ground_truth.get(view_id)
        pred = prediction.get(view_id)
        if gt is None and pred is None:
            continue
        reference = _binary(gt if gt is not None else pred)
        gt_binary = np.zeros_like(reference) if gt is None else _resize(gt, reference.shape)
        pred_binary = np.zeros_like(reference) if pred is None else _resize(pred, reference.shape)
        pred_boundary = _boundary(pred_binary)
        gt_boundary = _boundary(gt_binary)
        pred_total += int(pred_boundary.sum())
        gt_total += int(gt_boundary.sum())
        if tolerance_pixels == 0:
            dilated_gt, dilated_pred = gt_boundary, pred_boundary
        else:
            dilated_gt = binary_dilation(
                gt_boundary, structure=kernel, iterations=tolerance_pixels
            )
            dilated_pred = binary_dilation(
                pred_boundary, structure=kernel, iterations=tolerance_pixels
            )
        matched_pred += int(np.logical_and(pred_boundary, dilated_gt).sum())
        matched_gt += int(np.logical_and(gt_boundary, dilated_pred).sum())
    precision = matched_pred / pred_total if pred_total else 0.0
    recall = matched_gt / gt_total if gt_total else 0.0
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def evaluate_instances(
    predictions: MaskSet,
    ground_truth: MaskSet,
    view_ids: Sequence[int],
    *,
    failure_iou_threshold: float = 0.5,
    boundary_tolerance_pixels: int = 2,
) -> InstanceMetrics:
    if not 0.0 <= failure_iou_threshold <= 1.0:
        raise ValueError("failure_iou_threshold must be in [0, 1]")
    pred_names = sorted(predictions)
    gt_names = sorted(ground_truth)
    iou_matrix = np.zeros((len(pred_names), len(gt_names)), dtype=np.float64)
    for row, pred_name in enumerate(pred_names):
        for col, gt_name in enumerate(gt_names):
            iou_matrix[row, col] = instance_iou(
                predictions[pred_name], ground_truth[gt_name], view_ids
            )

    if iou_matrix.size:
        rows, cols = linear_sum_assignment(-iou_matrix)
    else:
        rows = cols = np.asarray([], dtype=np.int64)
    denominator = max(len(pred_names), len(gt_names), 1)
    matched_ious = tuple(float(iou_matrix[row, col]) for row, col in zip(rows, cols))
    scene_miou = float(sum(matched_ious) / denominator)

    matching = tuple(
        (pred_names[row], gt_names[col], float(iou_matrix[row, col]))
        for row, col in zip(rows, cols)
    )
    boundary_scores = [
        boundary_f1_pair(
            predictions[pred_names[row]],
            ground_truth[gt_names[col]],
            view_ids,
            tolerance_pixels=boundary_tolerance_pixels,
        )
        for row, col in zip(rows, cols)
    ]
    boundary_f1 = float(sum(boundary_scores) / denominator)

    failed = denominator - len(matched_ious)
    failed += sum(value < failure_iou_threshold for value in matched_ious)
    failure_rate = float(failed / denominator)

    foreground_pred: dict[int, np.ndarray] = {}
    foreground_gt: dict[int, np.ndarray] = {}
    for view_id in view_ids:
        pred_masks = [
            _binary(instance[view_id])
            for instance in predictions.values()
            if view_id in instance
        ]
        gt_masks = [
            _binary(instance[view_id])
            for instance in ground_truth.values()
            if view_id in instance
        ]
        available = gt_masks + pred_masks
        if not available:
            continue
        shape = available[0].shape
        pred_union = np.zeros(shape, dtype=bool)
        gt_union = np.zeros(shape, dtype=bool)
        for mask in pred_masks:
            pred_union |= _resize(mask, shape)
        for mask in gt_masks:
            gt_union |= _resize(mask, shape)
        foreground_pred[view_id] = pred_union
        foreground_gt[view_id] = gt_union
    foreground_iou = instance_iou(foreground_pred, foreground_gt, view_ids)
    return InstanceMetrics(
        scene_instance_miou=scene_miou,
        foreground_iou=foreground_iou,
        boundary_f1=boundary_f1,
        failure_rate=failure_rate,
        num_predictions=len(pred_names),
        num_ground_truth=len(gt_names),
        num_matched=len(matched_ious),
        matched_ious=matched_ious,
        matching=matching,
    )


def load_instance_masks(root: str | Path, view_ids: Sequence[int]) -> dict[str, dict[int, np.ndarray]]:
    """Load ``root/<instance>/<view_id>.png`` masks; absent views remain absent."""

    import imageio.v2 as imageio

    root = Path(root)
    if not root.is_dir():
        return {}
    result: dict[str, dict[int, np.ndarray]] = {}
    for instance_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        masks: dict[int, np.ndarray] = {}
        for view_id in view_ids:
            candidates = [
                instance_dir / f"{view_id:02d}.png",
                instance_dir / f"{view_id}.png",
                instance_dir / f"{view_id:03d}.png",
            ]
            path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if path is not None:
                mask = imageio.imread(path)
                masks[int(view_id)] = mask
        result[instance_dir.name] = masks
    return result
