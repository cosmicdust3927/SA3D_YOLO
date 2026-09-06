"""Headless, failure-aware, multi-instance SA3D condition runner.

Each YOLO detection in the first selected view starts an independent binary
SA3D optimization from the same NeRF checkpoint. YOLO boxes are refined with
SAM; later-view detections are associated to the current rendered instance by
IoU/confidence score. Missing detections are recorded as skipped views rather
than replaced by a fabricated center mask.
"""

from __future__ import annotations

import argparse
import copy
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.icra.metrics import evaluate_instances, load_instance_masks
from lib import sam3d, utils
from lib.bbox_utils import compute_bbox_by_cam_frustrm
from lib.config_loader import Config
from lib.configs import config_parser


@dataclass
class Detection:
    detection_index: int
    box_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    proxy_mask: np.ndarray


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        second = cv2.resize(
            second.astype(np.uint8),
            (first.shape[1], first.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    first = first.astype(bool)
    second = second.astype(bool)
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 0.0


def _detect(model, image: np.ndarray, class_id: int, confidence: float) -> list[Detection]:
    height, width = image.shape[:2]
    results = model.predict(
        source=image,
        imgsz=(height, width),
        classes=class_id,
        conf=confidence,
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []
    boxes = results[0].boxes.xyxy.detach().cpu().numpy()
    confidences = results[0].boxes.conf.detach().cpu().numpy()
    classes = results[0].boxes.cls.detach().cpu().numpy().astype(int)
    raw_masks = None if results[0].masks is None else results[0].masks.data.detach().cpu().numpy()

    detections: list[Detection] = []
    for index, (box, score, detected_class) in enumerate(zip(boxes, confidences, classes)):
        if raw_masks is None:
            mask = np.zeros((height, width), dtype=bool)
            x1, y1, x2, y2 = np.rint(box).astype(int)
            mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = True
        else:
            mask = cv2.resize(
                raw_masks[index].astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0.5
        detections.append(
            Detection(
                detection_index=index,
                box_xyxy=tuple(float(value) for value in box),
                confidence=float(score),
                class_id=int(detected_class),
                proxy_mask=mask,
            )
        )
    # A spatial sort makes instance labels reproducible across Ultralytics versions.
    return sorted(
        detections,
        key=lambda item: (*item.box_xyxy, -item.confidence, item.detection_index),
    )


def _sam_mask(predictor, image: np.ndarray, detection: Detection) -> np.ndarray:
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.asarray(detection.box_xyxy, dtype=np.float32),
        multimask_output=True,
    )
    if masks is None or len(masks) == 0:
        raise RuntimeError("SAM returned no masks for the YOLO box prompt")
    return masks[int(np.argmax(scores))].astype(np.float32)


def _raw_image(data_dict: dict, data_index: int) -> np.ndarray:
    image = data_dict["images"][data_index].cpu().numpy()
    return utils.to8b(image)


def _detection_record(detection: Detection) -> dict:
    result = asdict(detection)
    result.pop("proxy_mask")
    return result


def _render_test_masks(segmentation, data_dict: dict, output_dir: Path, threshold: float) -> None:
    test_indices = np.asarray(data_dict["i_test"], dtype=np.int64)
    camera_ids = np.asarray(data_dict["camera_ids"], dtype=np.int64)
    camera_parameters = [
        data_dict["poses"][test_indices],
        data_dict["HW"][test_indices],
        data_dict["Ks"][test_indices],
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for local_index, data_index in enumerate(test_indices):
            _, _, _, logits, _ = segmentation.render_view(
                local_index, cam_params=camera_parameters
            )
            mask = logits[..., 0].detach().cpu().numpy() > threshold
            camera_id = int(camera_ids[data_index])
            cv2.imwrite(str(output_dir / f"{camera_id:02d}.png"), mask.astype(np.uint8) * 255)


def build_parser() -> argparse.ArgumentParser:
    parser = config_parser()
    parser.description = "Run one ICRA multi-instance SA3D condition"
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--yolo_confidence", type=float, default=0.25)
    parser.add_argument("--association_iou_weight", type=float, default=0.6)
    parser.add_argument("--mask_threshold", type=float, default=0.0)
    parser.add_argument("--failure_iou_threshold", type=float, default=0.5)
    parser.add_argument("--boundary_tolerance_pixels", type=int, default=2)
    return parser


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("SA3D condition execution requires a CUDA GPU")
    cfg = Config.fromfile(args.config)
    utils.seed_everything(args)
    data_dict = utils.load_everything(args=args, cfg=cfg)
    camera_ids = np.asarray(data_dict["camera_ids"], dtype=np.int64)
    train_indices = np.asarray(data_dict["i_train"], dtype=np.int64)
    test_indices = np.asarray(data_dict["i_test"], dtype=np.int64)
    selected_ids = [int(camera_ids[index]) for index in train_indices]
    test_ids = [int(camera_ids[index]) for index in test_indices]
    result_dir = Path(args.result_dir).resolve()
    prediction_root = result_dir / "predictions"
    result_dir.mkdir(parents=True, exist_ok=True)

    target_class_id = int(cfg.yolo.target_class_id)
    yolo = sam3d.model_yolo
    detections_by_view: dict[int, list[Detection]] = {}
    for local_index, data_index in enumerate(train_indices):
        detections_by_view[local_index] = _detect(
            yolo,
            _raw_image(data_dict, int(data_index)),
            target_class_id,
            args.yolo_confidence,
        )

    initial_detections = detections_by_view[0]
    run_log: dict = {
        "selected_camera_ids": selected_ids,
        "test_camera_ids": test_ids,
        "target_class_id": target_class_id,
        "initial_detection_count": len(initial_detections),
        "instances": [],
    }
    skipped = 0
    total_operations = max(1, len(initial_detections) * len(train_indices))
    xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)

    for instance_index, initial_detection in enumerate(initial_detections):
        instance_name = f"instance_{instance_index:03d}"
        instance_log = {
            "instance": instance_name,
            "initial_detection": _detection_record(initial_detection),
            "views": [],
            "status": "running",
        }
        run_log["instances"].append(instance_log)
        instance_args = copy.copy(args)
        instance_args.sp_name = f"_{instance_name}"
        segmentation = None
        try:
            segmentation = sam3d.Sam3D(
                instance_args,
                cfg,
                cfg_model=cfg.coarse_model_and_render,
                cfg_train=cfg.coarse_train,
                xyz_min=xyz_min.clone(),
                xyz_max=xyz_max.clone(),
                data_dict=data_dict,
                stage="coarse",
            )
            segmentation.init_model()
            initial_image = _raw_image(data_dict, int(train_indices[0]))
            initial_mask = _sam_mask(
                segmentation.predictor, initial_image, initial_detection
            )
            segmentation.train_step(0, sam_mask=initial_mask)
            instance_log["views"].append(
                {
                    "camera_id": selected_ids[0],
                    "status": "trained",
                    "association": "initial_detection",
                }
            )

            for local_index in range(1, len(train_indices)):
                detections = detections_by_view[local_index]
                if not detections:
                    skipped += 1
                    instance_log["views"].append(
                        {"camera_id": selected_ids[local_index], "status": "skipped_no_detection"}
                    )
                    continue
                with torch.no_grad():
                    _, _, _, logits, _ = segmentation.render_view(local_index)
                    rendered = logits[..., 0].detach().cpu().numpy() > args.mask_threshold
                scored = [
                    (
                        args.association_iou_weight * _binary_iou(rendered, detection.proxy_mask)
                        + (1.0 - args.association_iou_weight) * detection.confidence,
                        detection,
                    )
                    for detection in detections
                ]
                score, selected_detection = max(
                    scored,
                    key=lambda item: (item[0], item[1].confidence, -item[1].detection_index),
                )
                image = _raw_image(data_dict, int(train_indices[local_index]))
                refined_mask = _sam_mask(
                    segmentation.predictor, image, selected_detection
                )
                segmentation.train_step(local_index, sam_mask=refined_mask)
                instance_log["views"].append(
                    {
                        "camera_id": selected_ids[local_index],
                        "status": "trained",
                        "association_score": float(score),
                        "detection": _detection_record(selected_detection),
                    }
                )

            segmentation.save_ckpt()
            _render_test_masks(
                segmentation,
                data_dict,
                prediction_root / instance_name,
                args.mask_threshold,
            )
            instance_log["status"] = "completed"
        except Exception as exc:  # Preserve partial failures as experimental outcomes.
            completed = len(instance_log["views"])
            skipped += max(0, len(train_indices) - completed)
            instance_log["status"] = "failed"
            instance_log["error"] = f"{type(exc).__name__}: {exc}"
            instance_log["traceback"] = traceback.format_exc()
        finally:
            del segmentation
            torch.cuda.empty_cache()

    gt = load_instance_masks(args.gt_dir, test_ids)
    predictions = load_instance_masks(prediction_root, test_ids)
    metrics = evaluate_instances(
        predictions,
        gt,
        test_ids,
        failure_iou_threshold=args.failure_iou_threshold,
        boundary_tolerance_pixels=args.boundary_tolerance_pixels,
    ).to_dict()
    skip_rate = 1.0 if not initial_detections else float(skipped / total_operations)
    metrics.update(
        {
            "segmentation_view_skip_rate": skip_rate,
            "initial_detection_failure": int(not initial_detections),
            "condition_failure": int(
                not initial_detections
                or all(item["status"] == "failed" for item in run_log["instances"])
            ),
        }
    )
    run_log["metrics"] = metrics
    _json_dump(result_dir / "condition_result.json", run_log)
    return run_log


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
