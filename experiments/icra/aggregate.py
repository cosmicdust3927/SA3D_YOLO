"""Aggregate raw condition JSON without mixing primary and end-to-end phases."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "scene_instance_miou",
    "foreground_iou",
    "boundary_f1",
    "failure_rate",
    "segmentation_view_skip_rate",
    "initial_detection_failure",
    "condition_failure",
)
RECONSTRUCTION_METRICS = ("psnr", "ssim", "lpips_vgg", "lpips_alex")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(root: str | Path) -> dict:
    root = Path(root)
    output: dict[str, dict] = {}
    for phase in ("main", "end_to_end"):
        raw_rows: list[dict] = []
        for result_path in sorted((root / phase).glob("**/condition_result.json")):
            manifest_path = result_path.parent / "condition_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            row = {
                "phase": phase,
                "scene": manifest["scene"],
                "budget": manifest["budget"],
                "method": manifest["method"],
                "selection_seed": manifest.get("selection_seed"),
            }
            row.update({key: result["metrics"].get(key) for key in METRICS})
            reconstruction_path = manifest.get("reconstruction_metrics")
            if reconstruction_path and Path(reconstruction_path).is_file():
                reconstruction = json.loads(
                    Path(reconstruction_path).read_text(encoding="utf-8")
                )["mean"]
                row.update(
                    {key: reconstruction.get(key) for key in RECONSTRUCTION_METRICS}
                )
            raw_rows.append(row)

        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for row in raw_rows:
            grouped[(row["scene"], row["budget"], row["method"])].append(row)
        summary_rows = []
        for (scene, budget, method), rows in sorted(grouped.items()):
            summary = {
                "phase": phase,
                "scene": scene,
                "budget": budget,
                "method": method,
                "n": len(rows),
            }
            for metric in METRICS + RECONSTRUCTION_METRICS:
                values = np.asarray(
                    [row[metric] for row in rows if row.get(metric) is not None],
                    dtype=np.float64,
                )
                summary[f"{metric}_mean"] = float(values.mean()) if len(values) else None
                summary[f"{metric}_std"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                )
            summary_rows.append(summary)
        _write_csv(root / "aggregates" / f"{phase}_raw_results.csv", raw_rows)
        _write_csv(root / "aggregates" / f"{phase}_summary.csv", summary_rows)
        comparisons = []
        summary_lookup = {
            (row["scene"], row["budget"], row["method"]): row
            for row in summary_rows
        }
        for scene, budget, method in sorted(summary_lookup):
            if method != "proposed":
                continue
            proposed = summary_lookup[(scene, budget, "proposed")]
            for baseline in ("filename", "random"):
                baseline_row = summary_lookup.get((scene, budget, baseline))
                if baseline_row is None:
                    continue
                comparison = {
                    "phase": phase,
                    "scene": scene,
                    "budget": budget,
                    "baseline": baseline,
                }
                for metric in METRICS + RECONSTRUCTION_METRICS:
                    proposed_value = proposed.get(f"{metric}_mean")
                    baseline_value = baseline_row.get(f"{metric}_mean")
                    if proposed_value is not None and baseline_value is not None:
                        comparison[f"{metric}_proposed_minus_baseline"] = (
                            proposed_value - baseline_value
                        )
                comparisons.append(comparison)
        _write_csv(root / "aggregates" / f"{phase}_comparisons.csv", comparisons)
        output[phase] = {
            "raw_conditions": len(raw_rows),
            "summary_rows": len(summary_rows),
            "comparison_rows": len(comparisons),
        }

    common_reconstruction = []
    for path in sorted((root / "main" / "common_nerf").glob("*/reconstruction_metrics.json")):
        common_reconstruction.append(
            {"scene": path.parent.name, **json.loads(path.read_text(encoding="utf-8"))["mean"]}
        )
    _write_csv(root / "aggregates" / "main_common_nerf_reconstruction.csv", common_reconstruction)
    (root / "aggregates" / "aggregation_manifest.json").write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
