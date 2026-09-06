# ICRA sparse-view segmentation experiment protocol

## Claim under test

> With the same 3D radiance field and the same segmentation-view budget, the
> proposed camera-selection method achieves higher 3D instance-segmentation
> performance than filename and random selection.

The primary experiment isolates segmentation-view selection. The auxiliary
end-to-end experiment changes both NeRF and segmentation inputs and must never
be pooled with the primary result table.

## Leakage-safe split

- Fixed test IDs: `01 05 09 13 21 25 29 33`
- Candidate IDs: `02 03 04 06 07 08 10 11 12 14 15 16 22 23 24 26 27 28 30 31 32 34 35 36`
- Test RGB, masks, and poses are unavailable to NeRF fitting, SA3D fitting,
  YOLO/SAM prompt creation, and camera ordering.
- LLFF image rows are mapped to camera IDs parsed from filenames. Any missing,
  duplicated, overlapping, or unexpected ID aborts planning.

## Exact-budget selection

All methods first create a 24-view order. For budget `K`, positions
`floor(k * 24 / K), k=0,...,K-1` are selected. This is the exact-budget
generalization of stride sampling and gives nested subsets for K=4, 8, and 16.

- **Filename:** ascending camera ID order.
- **Random:** NumPy `PCG64` permutation, independently seeded with 0 through 9.
- **Proposed:** mean-pose initialization, SE(3) distance
  `d = rotation_angle + alpha * translation_distance`, and history recurrence
  `score <- beta * score + distance_to_latest`. Defaults are alpha=1 and
  beta=0.5, matching the repository implementation and manuscript setup.

Training randomness is fixed at seed 777. Random selection seeds change only
the selected camera subset, not the training seed.

## Primary experiment

For each scene, one NeRF is trained from all 24 candidate views. Every
Filename, Proposed, and Random condition uses the exact same checkpoint. Only
the K views presented to YOLO, SAM, and SA3D change.

The first selected view is processed by YOLO. Every target-class detection,
ordered spatially for stable IDs, creates an independent binary SA3D instance.
The YOLO box prompts SAM; the best SAM mask initializes the instance. Later
views associate detections to the current rendered instance using
`0.6 * mask_IoU + 0.4 * YOLO_confidence`, then use the associated box as a new
SAM prompt. A missing detection is a logged skip. A missing first-view
detection yields zero predictions and a real failed condition; no synthetic
center mask is fabricated.

## Auxiliary end-to-end experiment

Each condition trains NeRF from its selected K views, evaluates the fixed test
views, then trains SA3D using the same ordered K views. It has its own directory
and result tables.

## Metrics

Instance matching uses one Hungarian assignment per scene. Each pairwise IoU
is computed by summing intersection and union pixels over all eight test views.

- **Scene-level instance mIoU:** sum of matched IoUs divided by
  `max(number of predictions, number of GT instances)`. Unmatched entries are
  zeros.
- **Foreground IoU:** all predicted instances and all GT instances are unioned
  per view, then intersection and union are accumulated over eight views.
- **Boundary F1:** matched-instance boundary F1 with a two-pixel tolerance,
  averaged with zeros for unmatched entries.
- **Failure rate:** fraction of the `max(P,G)` instance slots that are unmatched
  or have IoU below 0.5.
- **Segmentation-view skip rate:** skipped instance-view optimizations divided
  by `detected_instances * K`; it is 1 when the initial detection fails.
- **NeRF:** PSNR, SSIM, LPIPS-VGG, and LPIPS-Alex, averaged over the fixed test
  views.

Random conditions are aggregated as mean and sample standard deviation across
the ten selection seeds. Because the ten seeds are not ten independent scenes,
formal significance testing should use scenes as the top-level sampling unit;
do not report a naive 480-sample p-value. Report per-scene/budget deltas and,
if a confidence interval is needed, use a hierarchical bootstrap that resamples
scenes first and random seeds second.

## Required output separation

- `aggregates/main_raw_results.csv` and `main_summary.csv`: primary claim.
- `aggregates/end_to_end_raw_results.csv` and `end_to_end_summary.csv`: auxiliary.
- `*_comparisons.csv`: per-scene/budget Proposed-minus-baseline deltas.
- `main_common_nerf_reconstruction.csv`: the four shared primary NeRFs.
- Common-NeRF reconstruction is reported once per scene in the primary table,
  because it is identical for every segmentation selection method and budget.
