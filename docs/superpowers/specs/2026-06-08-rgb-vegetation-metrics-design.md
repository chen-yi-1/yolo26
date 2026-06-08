# RGB Vegetation Metrics Script Design

## Context

The YOLO26 plant seedling health workflow needs a first RGB-only analysis step before training or rule-based health classification. The current repository has a YOLO26/Ultralytics training and inference wrapper, but no script for calculating RGB vegetation indices.

The approved first version should generate measurable RGB indicators and visual checks, not direct health diagnoses.

## Goals

- Add a batch script under `scripts/` for RGB image analysis.
- Calculate RGB vegetation indices that can be computed from normal camera images.
- Export one CSV row per image with summary statistics.
- Export vegetation mask images and mask overlays for manual quality checks.
- Keep health-state classification out of the first script, except for documenting how exported indicators should later be calibrated.

## Non-Goals

- Do not run YOLO inference inside this script.
- Do not require trained YOLO weights.
- Do not output final `healthy`, `subhealthy`, or `unhealthy` labels in version 1.
- Do not add speculative threshold presets for specific crops before real data is measured.

## Computable RGB Indicators

For each image, convert RGB values to float values in `[0, 1]`, then calculate:

- `ExG = 2G - R - B`
- `ExR = 1.4R - G`
- `ExGR = ExG - ExR`
- `NGRDI = (G - R) / (G + R)`
- `GLI = (2G - R - B) / (2G + R + B)`
- `VARI = (G - R) / (G + R - B)`
- `CIVE = 0.441R - 0.811G + 0.385B`

Division formulas should use a small epsilon to avoid division by zero.

## Outputs

Given an input image directory and an output directory, the script writes:

- `metrics.csv`
- `masks/<image_stem>_mask.png`
- `overlays/<image_stem>_overlay.png`

The CSV contains:

- Image metadata: path, width, height.
- Vegetation coverage ratio: masked vegetation pixels divided by total pixels.
- For each RGB index: full-image mean and standard deviation.
- For each RGB index: vegetation-mask mean and standard deviation.

## Mask Generation

Version 1 uses an `ExG` threshold to generate a binary vegetation mask:

```text
mask = ExG > threshold
```

The threshold should be configurable from the command line. A conservative default is acceptable, but users should validate it visually through the exported masks and overlays.

The overlay image should tint masked vegetation pixels green on top of the original image.

## How To Use Metrics For Health-State Analysis

The exported indicators should be analyzed in stages:

1. Collect images from each crop and growth day under consistent lighting where possible.
2. Export RGB metrics and overlays.
3. Manually label a small validation table with visible states such as `healthy`, `subhealthy`, `fungi_mold`, `wilted_yellowing`, `dead_rotten`, and `overgrown`.
4. Check whether mask overlays correctly isolate seedlings from tray, soil, paper, or substrate. If masks are poor, adjust the `ExG` threshold before interpreting health metrics.
5. Compare metric distributions by label:
   - Higher vegetation coverage, higher `ExG`, higher `GLI`, and higher `NGRDI` generally support healthy growth.
   - Lower `GLI` and `NGRDI` can indicate yellowing or weak green response.
   - Higher `ExR` can support yellow, brown, dry, or rotten-region detection, but should not be used alone.
   - Fast drops in coverage, `GLI`, or `NGRDI` across time can support early `subhealthy` detection.
6. Choose thresholds from the real dataset only after plotting or tabulating label-wise distributions. Initial thresholds should be crop- and camera-specific, because RGB values depend strongly on lighting, background, camera white balance, and growth stage.

The first practical classification layer should be rule-assisted, not rule-only:

```text
healthy: high vegetation coverage and stable/high green indices
subhealthy: mild decline in green indices or slow coverage growth
wilted_yellowing: low GLI/NGRDI with visible yellowing or wilting
dead_rotten: low vegetation signal plus high red/brown support and visible dead tissue
fungi_mold: usually not reliable from these green indices alone; needs visual texture/color detection or YOLO labels
overgrown: better measured with area, density, height/proxy shape, or time metadata than RGB indices alone
```

## CLI Shape

The script should support:

```bash
python scripts/rgb_vegetation_metrics.py --input datasets/images/val --output runs/rgb_metrics
```

Useful options:

- `--threshold`: ExG mask threshold.
- `--recursive`: process nested image directories.
- `--overlay-alpha`: opacity for the green overlay.

## Error Handling

- Skip unreadable files with a warning.
- Fail clearly if the input directory does not exist.
- Create output directories when missing.
- Write an empty CSV with headers if no supported images are found.

## Testing

Add focused tests for:

- RGB index formulas on a small synthetic image.
- Mask coverage calculation.
- CSV row generation for one temporary image.

Tests should avoid requiring YOLO weights, datasets, or GPU.
