import argparse
import csv
from pathlib import Path

import numpy as np


GREEN_FEATURES = ("ExG_veg_mean", "GLI_veg_mean", "NGRDI_veg_mean", "VARI_veg_mean")
REQUIRED_FIELDS = ("image_path", "vegetation_coverage", "ExR_veg_mean", *GREEN_FEATURES)


def parse_float(row, field):
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot parse numeric field '{field}' in row for {row.get('image_path', '<unknown>')}") from exc


def percentile_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.array([], dtype=np.float64)
    if values.size == 1:
        return np.array([1.0], dtype=np.float64)

    order = np.argsort(values)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks / (values.size - 1)


def calculate_scores(rows):
    coverage = np.asarray([parse_float(row, "vegetation_coverage") for row in rows], dtype=np.float64)
    exr = np.asarray([parse_float(row, "ExR_veg_mean") for row in rows], dtype=np.float64)

    green_ranks = []
    for field in GREEN_FEATURES:
        values = [parse_float(row, field) for row in rows]
        green_ranks.append(percentile_ranks(values))

    return {
        "coverage_rank": percentile_ranks(coverage),
        "red_rank": percentile_ranks(exr),
        "green_score": np.mean(np.vstack(green_ranks), axis=0),
    }


def classify_row(row, scores, index):
    coverage = parse_float(row, "vegetation_coverage")
    coverage_rank = float(scores["coverage_rank"][index])
    green_score = float(scores["green_score"][index])
    red_rank = float(scores["red_rank"][index])

    if coverage_rank <= 0.20 and green_score <= 0.25 and red_rank >= 0.60:
        status = "dead_rotten"
        confidence = max(green_score, 1.0 - coverage_rank, red_rank)
        reason = "low vegetation coverage, weak green indices, and high ExR red/brown signal"
    elif green_score <= 0.35 and red_rank >= 0.45:
        status = "wilted_yellowing"
        confidence = (1.0 - green_score + red_rank) / 2.0
        reason = "low GLI/NGRDI/VARI/ExG green score with elevated ExR"
    elif coverage_rank >= 0.90 and green_score >= 0.45:
        status = "overgrown"
        confidence = (coverage_rank + green_score) / 2.0
        reason = "very high canopy coverage in this batch"
    elif coverage_rank >= 0.35 and green_score >= 0.55 and red_rank <= 0.75:
        status = "healthy"
        confidence = (coverage_rank + green_score + (1.0 - red_rank)) / 3.0
        reason = "good canopy coverage, strong green indices, and no strong red/brown signal"
    else:
        status = "subhealthy"
        confidence = 0.55
        reason = "intermediate RGB index pattern; review overlay or calibrate with labels"

    classified = dict(row)
    classified.update(
        {
            "health_status": status,
            "confidence": round(float(confidence), 4),
            "coverage_rank": round(coverage_rank, 4),
            "green_score": round(green_score, 4),
            "red_rank": round(red_rank, 4),
            "reason": reason,
        }
    )
    if coverage <= 0:
        classified["reason"] = "no vegetation pixels in mask; check threshold or image"
    return classified


def read_metrics_csv(input_csv):
    with Path(input_csv).open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if rows:
        missing = [field for field in REQUIRED_FIELDS if field not in rows[0]]
        if missing:
            raise ValueError(f"Missing required metrics columns: {', '.join(missing)}")
    return rows


def write_classification_csv(rows, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = list(REQUIRED_FIELDS) + [
            "health_status",
            "confidence",
            "coverage_rank",
            "green_score",
            "red_rank",
            "reason",
        ]

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def classify_metrics_rows(rows):
    if not rows:
        return []
    scores = calculate_scores(rows)
    return [classify_row(row, scores, index) for index, row in enumerate(rows)]


def classify_metrics_csv(input_csv, output_csv):
    rows = read_metrics_csv(input_csv)
    classified = classify_metrics_rows(rows)
    write_classification_csv(classified, output_csv)
    return classified


def parse_args():
    parser = argparse.ArgumentParser(description="Classify seedling health from RGB vegetation metrics CSV.")
    parser.add_argument("--input", required=True, help="Input metrics.csv from rgb_vegetation_metrics.py.")
    parser.add_argument("--output", required=True, help="Output CSV with health_status, confidence, and reason.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = classify_metrics_csv(args.input, args.output)
    print(f"Wrote {len(rows)} classified rows to {args.output}")


if __name__ == "__main__":
    main()
