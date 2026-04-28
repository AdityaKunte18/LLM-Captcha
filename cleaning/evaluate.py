from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from cleaning.pipeline import CleaningConfig, clean_image
from cleaning.reference import render_clean_reference


def _resolve_dataset_dir(dataset_dir: Path) -> tuple[Path, Path]:
    candidates = [
        (dataset_dir, dataset_dir / "metadata.json"),
        (dataset_dir / "captcha_dataset", dataset_dir / "captcha_dataset" / "metadata.json"),
    ]
    for image_dir, metadata_path in candidates:
        if metadata_path.exists():
            return image_dir, metadata_path
    raise FileNotFoundError(f"Could not find metadata.json under {dataset_dir}")


def _load_metadata(metadata_path: Path) -> list[dict]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _render_reference(expression: str) -> np.ndarray:
    reference = render_clean_reference(expression)
    return np.array(reference.convert("L"))


def _compute_ssim(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float64)
    second = second.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_first = cv2.GaussianBlur(first, (11, 11), 1.5)
    mu_second = cv2.GaussianBlur(second, (11, 11), 1.5)

    mu_first_sq = mu_first * mu_first
    mu_second_sq = mu_second * mu_second
    mu_cross = mu_first * mu_second

    sigma_first_sq = cv2.GaussianBlur(first * first, (11, 11), 1.5) - mu_first_sq
    sigma_second_sq = cv2.GaussianBlur(second * second, (11, 11), 1.5) - mu_second_sq
    sigma_cross = cv2.GaussianBlur(first * second, (11, 11), 1.5) - mu_cross

    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (mu_first_sq + mu_second_sq + c1) * (sigma_first_sq + sigma_second_sq + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-12)
    return float(ssim_map.mean())


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask > 0


def _mask_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    candidate_mask = _foreground_mask(candidate)
    reference_mask = _foreground_mask(reference)

    tp = float(np.logical_and(candidate_mask, reference_mask).sum())
    fp = float(np.logical_and(candidate_mask, ~reference_mask).sum())
    fn = float(np.logical_and(~candidate_mask, reference_mask).sum())
    union = float(np.logical_or(candidate_mask, reference_mask).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / union if union else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
    }


def _difficulty_bucket(difficulty: float) -> str:
    if difficulty < 0.3:
        return "low"
    if difficulty < 0.5:
        return "medium"
    return "high"


def _evaluate_variant(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    scores = _mask_metrics(candidate, reference)
    scores["ssim"] = _compute_ssim(candidate, reference)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate raw and cleaned CAPTCHA images against regenerated clean references")
    parser.add_argument("--dataset-dir", default="captcha_dataset", help="Path containing metadata.json and image files")
    parser.add_argument("--cleaned-dir", default="cleaned_dataset", help="Directory created by run_batch.py")
    parser.add_argument("--output", default=None, help="Optional explicit output path for the evaluation JSON")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a quick smoke test")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    image_dir, metadata_path = _resolve_dataset_dir(dataset_dir)
    metadata = _load_metadata(metadata_path)
    if args.limit is not None:
        metadata = metadata[:args.limit]

    cleaned_dir = Path(args.cleaned_dir)
    gray_dir = cleaned_dir / "cleaned_gray"
    binary_dir = cleaned_dir / "cleaned_binary"
    if not gray_dir.exists() or not binary_dir.exists():
        raise FileNotFoundError("Cleaned dataset directories were not found. Run cleaning/run_batch.py first.")

    aggregate: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_bucket: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    by_noise_presence: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    per_item: list[dict] = []

    default_config = CleaningConfig()
    no_component_config = CleaningConfig(enable_component_filter=False)
    no_denoise_config = CleaningConfig(enable_denoise=False)

    for index, item in enumerate(metadata, start=1):
        raw_path = image_dir / item["filename"]
        gray_path = gray_dir / item["filename"]
        binary_path = binary_dir / item["filename"]

        raw = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
        gray_clean = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
        binary_clean = cv2.imread(str(binary_path), cv2.IMREAD_GRAYSCALE)
        if raw is None or gray_clean is None or binary_clean is None:
            raise FileNotFoundError(f"Missing evaluation input for {item['filename']}")

        raw_rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        no_component = clean_image(raw_rgb, no_component_config)["binary_clean"]
        no_denoise = clean_image(raw_rgb, no_denoise_config)["binary_clean"]
        reference = _render_reference(item["expression"])

        variants = {
            "raw_noisy": raw,
            "gray_clean": gray_clean,
            "binary_clean": binary_clean,
            "no_component_filter": no_component,
            "no_denoise": no_denoise,
        }

        item_result = {
            "filename": item["filename"],
            "difficulty": item["difficulty"],
            "bucket": _difficulty_bucket(item["difficulty"]),
            "noise_flags": {
                "gaussian": item["params"]["gaussian_sigma"] > 0,
                "salt_pepper": item["params"]["sp_ratio"] > 0,
                "blur": item["params"]["blur_radius"] > 0,
                "multi_line": item["params"]["line_count"] >= 2,
            },
            "metrics": {},
        }

        for name, candidate in variants.items():
            scores = _evaluate_variant(candidate, reference)
            aggregate[name].append(scores)
            by_bucket[item_result["bucket"]][name].append(scores)
            item_result["metrics"][name] = scores

            for noise_name, present in item_result["noise_flags"].items():
                key = "present" if present else "absent"
                by_noise_presence[f"{noise_name}_{key}"][name].append(scores)

        per_item.append(item_result)

        if index % 100 == 0 or index == len(metadata):
            print(f"Evaluated {index}/{len(metadata)} images")

    def summarize(groups: dict[str, list[dict[str, float]]]) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for name, items in groups.items():
            summary[name] = {
                metric: round(float(np.mean([row[metric] for row in items])), 6)
                for metric in ("ssim", "precision", "recall", "iou")
            }
        return summary

    output = {
        "count": len(per_item),
        "overall": summarize(aggregate),
        "by_difficulty": {bucket: summarize(groups) for bucket, groups in by_bucket.items()},
        "by_noise_presence": {name: summarize(groups) for name, groups in by_noise_presence.items()},
        "items": per_item,
        "ablation_configs": {
            "default": default_config.__dict__,
            "no_component_filter": no_component_config.__dict__,
            "no_denoise": no_denoise_config.__dict__,
        },
    }

    output_path = Path(args.output) if args.output else cleaned_dir / "evaluation_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"Saved evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
