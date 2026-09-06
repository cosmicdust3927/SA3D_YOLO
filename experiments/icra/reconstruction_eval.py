"""Evaluate dumped fixed-test NeRF RGB predictions with four metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from lib import utils
from lib.config_loader import Config
from lib.load_data import load_data


def evaluate(config_path: str, prediction_dir: str, output_path: str) -> dict:
    cfg = Config.fromfile(config_path)
    data = load_data(cfg.data)
    test_indices = np.asarray(data["i_test"], dtype=np.int64)
    prediction_paths = sorted(Path(prediction_dir).glob("*.png"))
    if len(prediction_paths) != len(test_indices):
        raise ValueError(
            f"expected {len(test_indices)} prediction images, found {len(prediction_paths)}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    camera_ids = np.asarray(data["camera_ids"], dtype=np.int64)
    for prediction_path, data_index in zip(prediction_paths, test_indices):
        prediction = imageio.imread(prediction_path)[..., :3].astype(np.float32) / 255.0
        ground_truth = np.asarray(data["images"][data_index], dtype=np.float32)[..., :3]
        if prediction.shape != ground_truth.shape:
            raise ValueError(
                f"shape mismatch for {prediction_path}: {prediction.shape} vs {ground_truth.shape}"
            )
        mse = float(np.mean(np.square(prediction - ground_truth)))
        rows.append(
            {
                "camera_id": int(camera_ids[data_index]),
                "psnr": float(-10.0 * np.log10(max(mse, np.finfo(np.float32).tiny))),
                "ssim": float(utils.rgb_ssim(prediction, ground_truth, max_val=1.0)),
                "lpips_vgg": float(utils.rgb_lpips(ground_truth, prediction, "vgg", device)),
                "lpips_alex": float(utils.rgb_lpips(ground_truth, prediction, "alex", device)),
            }
        )
    summary = {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in ("psnr", "ssim", "lpips_vgg", "lpips_alex")
    }
    payload = {"per_view": rows, "mean": summary}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args.config, args.prediction_dir, args.output)
    print(json.dumps(result["mean"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

