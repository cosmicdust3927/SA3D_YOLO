"""Plan and execute the ICRA sparse-view experiment matrix.

The planner is intentionally strict: it refuses to create a plan until all 32
camera IDs and all eight held-out GT views are present. The executor is
single-GPU/sequential by default, writes every command and log, and resumes only
from concrete output files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

from experiments.icra import SCHEMA_VERSION
from experiments.icra.aggregate import aggregate
from experiments.icra.dataset import build_scene_index, validate_split
from experiments.icra.metrics import load_instance_masks
from experiments.icra.view_selection import select_views


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={SCHEMA_VERSION}")
    if sorted(config["budgets"]) != [4, 8, 16]:
        raise ValueError("the ICRA protocol requires budgets [4, 8, 16]")
    if len(config["random_selection_seeds"]) != 10:
        raise ValueError("the ICRA protocol requires exactly 10 random selection seeds")
    if len(set(config["random_selection_seeds"])) != 10:
        raise ValueError("random selection seeds must be unique")
    if len(config["scenes"]) != 4:
        raise ValueError("the ICRA protocol requires exactly four scenes")
    return config


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(_file_sha256(path).encode())
    return digest.hexdigest()


def _condition_name(method: str, selection_seed: int | None) -> str:
    return method if selection_seed is None else f"{method}_seed{selection_seed:02d}"


def build_plan(config_path: str | Path, output_root: str | Path) -> dict:
    config = _load_config(config_path)
    output_root = Path(output_root).expanduser().resolve()
    candidate_ids = config["candidate_camera_ids"]
    test_ids = config["test_camera_ids"]
    proposed = config["proposed"]
    conditions = []
    scenes = {}

    for scene, scene_cfg in config["scenes"].items():
        index = build_scene_index(
            scene,
            _resolve(scene_cfg["dataset_dir"]),
            camera_id_pattern=config["camera_id_pattern"],
        )
        validate_split(index, candidate_ids, test_ids)
        gt = load_instance_masks(_resolve(scene_cfg["gt_instances_dir"]), test_ids)
        if not gt:
            raise FileNotFoundError(
                f"{scene}: no GT instances found in {scene_cfg['gt_instances_dir']}; "
                "expected <gt_instances_dir>/<instance>/<camera_id>.png"
            )
        missing_gt = {
            instance: sorted(set(test_ids) - set(masks))
            for instance, masks in gt.items()
            if set(test_ids) - set(masks)
        }
        if missing_gt:
            raise ValueError(f"{scene}: incomplete fixed-test GT masks: {missing_gt}")
        candidate_poses = index.poses_for(candidate_ids)
        scene_record = {
            **scene_cfg,
            "dataset_dir": str(_resolve(scene_cfg["dataset_dir"])),
            "nerf_config": str(_resolve(scene_cfg["nerf_config"])),
            "seg_config": str(_resolve(scene_cfg["seg_config"])),
            "gt_instances_dir": str(_resolve(scene_cfg["gt_instances_dir"])),
            "camera_ids_in_llff_order": list(index.camera_ids),
            "gt_instance_names": sorted(gt),
            "poses_bounds_sha256": _file_sha256(index.dataset_dir / "poses_bounds.npy"),
            "gt_instances_sha256": _tree_sha256(_resolve(scene_cfg["gt_instances_dir"])),
        }
        scenes[scene] = scene_record

        for budget in config["budgets"]:
            selections = [
                select_views("filename", candidate_ids, candidate_poses, budget),
                select_views(
                    "proposed",
                    candidate_ids,
                    candidate_poses,
                    budget,
                    alpha=proposed["alpha"],
                    beta=proposed["beta"],
                    initial=proposed["initial"],
                ),
            ]
            selections.extend(
                select_views(
                    "random",
                    candidate_ids,
                    candidate_poses,
                    budget,
                    selection_seed=seed,
                )
                for seed in config["random_selection_seeds"]
            )
            for selection in selections:
                record = {
                    "scene": scene,
                    "budget": budget,
                    "method": selection.method,
                    "selection_seed": selection.selection_seed,
                    "ordered_camera_ids": list(selection.ordered_camera_ids),
                    "selected_camera_ids": list(selection.selected_camera_ids),
                }
                record["condition_id"] = (
                    f"{scene}/k{budget:02d}/"
                    f"{_condition_name(selection.method, selection.selection_seed)}"
                )
                conditions.append(record)

    try:
        source_git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_git_commit = None
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "repo_root": str(REPO_ROOT),
        "output_root": str(output_root),
        "source_config": str(Path(config_path).resolve()),
        "source_git_commit": source_git_commit,
        "config": config,
        "scenes": scenes,
        "conditions": conditions,
        "counts": {
            "common_nerf": len(scenes),
            "main_conditions": len(conditions),
            "end_to_end_conditions": len(conditions),
        },
    }
    protocol["plan_sha256"] = _fingerprint(
        {
            "schema_version": protocol["schema_version"],
            "source_git_commit": source_git_commit,
            "config": config,
            "scenes": scenes,
            "conditions": conditions,
        }
    )
    _dump_json(output_root / "experiment_plan.json", protocol)
    _write_selected_views_csv(output_root / "selected_views.csv", conditions)
    _write_environment(output_root / "environment.json")
    return protocol


def _write_selected_views_csv(path: Path, conditions: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene", "budget", "method", "selection_seed",
                "ordered_camera_ids", "selected_camera_ids",
            ],
        )
        writer.writeheader()
        for condition in conditions:
            writer.writerow(
                {
                    **{key: condition.get(key) for key in writer.fieldnames[:4]},
                    "ordered_camera_ids": " ".join(map(str, condition["ordered_camera_ids"])),
                    "selected_camera_ids": " ".join(map(str, condition["selected_camera_ids"])),
                }
            )


def _write_environment(path: Path) -> None:
    def command(*parts: str) -> str | None:
        try:
            return subprocess.check_output(parts, cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    assets = {}
    for asset in (
        REPO_ROOT / "yolov8x-seg.pt",
        REPO_ROOT / "dependencies" / "sam_ckpt" / "sam_vit_h_4b8939.pth",
    ):
        assets[str(asset.relative_to(REPO_ROOT))] = (
            _file_sha256(asset) if asset.is_file() else None
        )
    _dump_json(
        path,
        {
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": command("git", "rev-parse", "HEAD"),
            "git_status": command("git", "status", "--short"),
            "nvidia_smi": command("nvidia-smi"),
            "pip_freeze": command(sys.executable, "-m", "pip", "freeze"),
            "model_asset_sha256": assets,
        },
    )


def _write_runtime_config(
    path: Path,
    base_config: str,
    dataset_dir: str,
    basedir: Path,
    expname: str,
    train_ids: list[int],
    test_ids: list[int],
    camera_id_pattern: str,
    target_class_id: int,
) -> None:
    body = (
        f"_base_ = {base_config!r}\n"
        f"basedir = {str(basedir)!r}\n"
        f"expname = {expname!r}\n"
        "data = dict(\n"
        f"    datadir={dataset_dir!r},\n"
        "    llffhold=0,\n"
        f"    camera_id_pattern={camera_id_pattern!r},\n"
        f"    train_camera_ids={train_ids!r},\n"
        f"    test_camera_ids={test_ids!r},\n"
        ")\n"
        f"yolo = dict(target_class_id={target_class_id!r})\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _run_command(command: list[str], log_path: Path, cwd: Path = REPO_ROOT) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    (log_path.parent / f"{log_path.stem}.command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    _dump_json(
        log_path.parent / f"{log_path.stem}.status.json",
        {"returncode": process.returncode, "elapsed_seconds": time.time() - started},
    )
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def _nerf_job(
    plan: dict,
    scene: str,
    train_ids: list[int],
    job_dir: Path,
    stop_at: int,
) -> tuple[Path, Path]:
    config = plan["config"]
    scene_cfg = plan["scenes"][scene]
    runtime_config = job_dir / "runtime_nerf_config.py"
    _write_runtime_config(
        runtime_config,
        scene_cfg["nerf_config"],
        scene_cfg["dataset_dir"],
        job_dir,
        "model",
        train_ids,
        config["test_camera_ids"],
        config["camera_id_pattern"],
        config["segmentation"]["target_class_id"],
    )
    checkpoint = job_dir / "model" / "fine_last.tar"
    prediction_dir = job_dir / "model" / "render_test_fine_last"
    reconstruction_metrics = job_dir / "reconstruction_metrics.json"
    job_manifest = {
        "plan_sha256": plan["plan_sha256"],
        "scene": scene,
        "train_camera_ids": train_ids,
        "test_camera_ids": config["test_camera_ids"],
        "stop_at": stop_at,
        "training_seed": config["training_seed"],
        "runtime_config_sha256": _file_sha256(runtime_config),
    }
    job_manifest_path = job_dir / "nerf_manifest.json"
    if checkpoint.is_file() and job_manifest_path.is_file():
        existing = json.loads(job_manifest_path.read_text(encoding="utf-8"))
        if existing != job_manifest:
            raise RuntimeError(
                f"refusing to reuse NeRF outputs with a different manifest: {job_dir}"
            )
    elif checkpoint.is_file():
        raise RuntimeError(f"checkpoint exists without a manifest: {checkpoint}")
    _dump_json(job_manifest_path, job_manifest)
    if not checkpoint.is_file() or not prediction_dir.is_dir():
        _run_command(
            [
                sys.executable,
                "run.py",
                f"--config={runtime_config}",
                f"--seed={config['training_seed']}",
                f"--stop_at={stop_at}",
                f"--i_weights={config['nerf']['checkpoint_interval']}",
                "--render_test",
                "--dump_images",
            ],
            job_dir / "logs" / "nerf.log",
        )
    if not reconstruction_metrics.is_file():
        _run_command(
            [
                sys.executable,
                "-m",
                "experiments.icra.reconstruction_eval",
                f"--config={runtime_config}",
                f"--prediction_dir={prediction_dir}",
                f"--output={reconstruction_metrics}",
            ],
            job_dir / "logs" / "reconstruction_eval.log",
        )
    return checkpoint, reconstruction_metrics


def _segmentation_job(
    plan: dict,
    condition: dict,
    phase: str,
    checkpoint: Path,
    condition_dir: Path,
) -> None:
    config = plan["config"]
    scene_cfg = plan["scenes"][condition["scene"]]
    seg_config = condition_dir / "runtime_seg_config.py"
    _write_runtime_config(
        seg_config,
        scene_cfg["seg_config"],
        scene_cfg["dataset_dir"],
        condition_dir,
        "segmentation",
        condition["selected_camera_ids"],
        config["test_camera_ids"],
        config["camera_id_pattern"],
        config["segmentation"]["target_class_id"],
    )
    manifest = {
        **condition,
        "phase": phase,
        "plan_sha256": plan["plan_sha256"],
        "training_seed": config["training_seed"],
        "nerf_checkpoint": str(checkpoint.resolve()),
        "nerf_checkpoint_sha256": _file_sha256(checkpoint),
        "seg_config": str(seg_config.resolve()),
        "seg_config_sha256": _file_sha256(seg_config),
    }
    manifest_path = condition_dir / "condition_manifest.json"
    result = condition_dir / "condition_result.json"
    if result.is_file():
        if not manifest_path.is_file():
            raise RuntimeError(f"result exists without a manifest: {result}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable_existing = dict(existing)
        comparable_existing.pop("reconstruction_metrics", None)
        if comparable_existing != manifest:
            raise RuntimeError(
                f"refusing to reuse condition outputs with a different manifest: {condition_dir}"
            )
        return
    _dump_json(manifest_path, manifest)
    seg = config["segmentation"]
    _run_command(
        [
            sys.executable,
            "-m",
            "experiments.icra.sa3d_runner",
            f"--config={seg_config}",
            f"--ft_path={checkpoint}",
            f"--seed={config['training_seed']}",
            "--segment",
            "--save_ckpt",
            f"--result_dir={condition_dir}",
            f"--gt_dir={scene_cfg['gt_instances_dir']}",
            f"--yolo_confidence={seg['yolo_confidence']}",
            f"--association_iou_weight={seg['association_iou_weight']}",
            f"--mask_threshold={seg['mask_threshold']}",
            f"--failure_iou_threshold={seg['failure_iou_threshold']}",
            f"--boundary_tolerance_pixels={seg['boundary_tolerance_pixels']}",
        ],
        condition_dir / "logs" / "segmentation.log",
    )


def execute(plan_path: str | Path, phase: str, max_conditions: int | None = None) -> None:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    root = Path(plan["output_root"])
    phases = ["main", "end_to_end"] if phase == "all" else [phase]
    completed = 0
    common_checkpoints: dict[str, Path] = {}

    if "main" in phases:
        for scene in plan["scenes"]:
            checkpoint, _ = _nerf_job(
                plan,
                scene,
                plan["config"]["candidate_camera_ids"],
                root / "main" / "common_nerf" / scene,
                plan["config"]["nerf"]["common_stop_at"],
            )
            common_checkpoints[scene] = checkpoint
        for condition in plan["conditions"]:
            condition_dir = root / "main" / condition["condition_id"]
            _segmentation_job(
                plan,
                condition,
                "main",
                common_checkpoints[condition["scene"]],
                condition_dir,
            )
            completed += 1
            if max_conditions is not None and completed >= max_conditions:
                aggregate(root)
                return

    if "end_to_end" in phases:
        for condition in plan["conditions"]:
            condition_dir = root / "end_to_end" / condition["condition_id"]
            checkpoint, reconstruction_metrics = _nerf_job(
                plan,
                condition["scene"],
                condition["selected_camera_ids"],
                condition_dir / "nerf",
                plan["config"]["nerf"]["end_to_end_stop_at"],
            )
            _segmentation_job(
                plan, condition, "end_to_end", checkpoint, condition_dir
            )
            manifest_path = condition_dir / "condition_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["reconstruction_metrics"] = str(reconstruction_metrics.resolve())
            _dump_json(manifest_path, manifest)
            completed += 1
            if max_conditions is not None and completed >= max_conditions:
                aggregate(root)
                return
    aggregate(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--config", required=True)
    plan_parser.add_argument("--output", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--plan", required=True)
    run_parser.add_argument(
        "--phase", choices=["main", "end_to_end", "all"], default="all"
    )
    run_parser.add_argument("--max_conditions", type=int)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--root", required=True)
    args = parser.parse_args()

    if args.command == "plan":
        result = build_plan(args.config, args.output)
        print(json.dumps(result["counts"], indent=2, sort_keys=True))
    elif args.command == "run":
        execute(args.plan, args.phase, args.max_conditions)
    else:
        print(json.dumps(aggregate(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
