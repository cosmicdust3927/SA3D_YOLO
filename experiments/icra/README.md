# Reproducible ICRA experiment pipeline

## 1. Prepare data and manual instance GT

Each scene must keep the standard LLFF `images/` and `poses_bounds.npy`. Camera
IDs are parsed from image filenames. Manual test masks use this layout:

```text
data/360_v2/Set1/gt_instances/
├── person_001/
│   ├── 01.png
│   ├── 05.png
│   └── ... eight fixed test views
└── person_002/
    ├── 01.png
    └── ...
```

Instance directory names must be consistent across the eight views. Empty or
occluded instances still need an all-zero mask so that GT completeness is
auditable.

## 2. Create an immutable plan

```bash
cp experiments/icra/config.example.json experiments/icra/config.json
python -m experiments.icra.pipeline plan \
  --config experiments/icra/config.json \
  --output runs/icra_2027
```

Planning validates the 32-camera split and GT, then writes all ordered and
selected view IDs. Expected counts are 4 common NeRF jobs, 144 primary
segmentation conditions, and 144 end-to-end conditions.

## 3. Run and resume

```bash
python -m experiments.icra.pipeline run \
  --plan runs/icra_2027/experiment_plan.json \
  --phase main

python -m experiments.icra.pipeline run \
  --plan runs/icra_2027/experiment_plan.json \
  --phase end_to_end
```

The executor is sequential for one GPU. Re-running the same command resumes
from concrete checkpoints and `condition_result.json` files. Use
`--max_conditions 1` for an initial end-to-end smoke run.

## 4. Rebuild result tables

```bash
python -m experiments.icra.pipeline aggregate --root runs/icra_2027
```

Every condition contains its selected IDs, selection/training seed, generated
config and hash, NeRF checkpoint and hash, exact commands, stdout/stderr log,
status JSON, predicted masks, per-instance association/skip log, and raw
metrics. Environment metadata and model-weight hashes are stored at run root.

See [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) for metric definitions and
reporting rules.

