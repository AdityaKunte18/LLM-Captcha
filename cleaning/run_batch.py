from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from cleaning.pipeline import CleaningConfig, clean_image


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-clean a CAPTCHA dataset")
    parser.add_argument("--dataset-dir", default="captcha_dataset", help="Path containing metadata.json and image files")
    parser.add_argument("--output-dir", default="cleaned_dataset", help="Directory to store cleaned outputs")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a quick smoke test")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    image_dir, metadata_path = _resolve_dataset_dir(dataset_dir)
    metadata = _load_metadata(metadata_path)
    if args.limit is not None:
        metadata = metadata[:args.limit]

    output_dir = Path(args.output_dir)
    gray_dir = output_dir / "cleaned_gray"
    binary_dir = output_dir / "cleaned_binary"
    gray_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)

    config = CleaningConfig()
    metrics: list[dict] = []

    for index, item in enumerate(metadata, start=1):
        image_path = image_dir / item["filename"]
        source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(f"Could not read image {image_path}")

        result = clean_image(cv2.cvtColor(source, cv2.COLOR_BGR2RGB), config)

        gray_filename = gray_dir / item["filename"]
        binary_filename = binary_dir / item["filename"]
        cv2.imwrite(str(gray_filename), result["gray_clean"])
        cv2.imwrite(str(binary_filename), result["binary_clean"])

        metrics.append(
            {
                "filename": item["filename"],
                "expression": item["expression"],
                "difficulty": item["difficulty"],
                "source_params": item["params"],
                "cleaning_stats": result["stats"],
                "outputs": {
                    "gray_clean": str(gray_filename),
                    "binary_clean": str(binary_filename),
                },
            }
        )

        if index % 100 == 0 or index == len(metadata):
            print(f"Processed {index}/{len(metadata)} images")

    summary = {
        "count": len(metrics),
        "config": metrics[0]["cleaning_stats"]["config"] if metrics else {},
        "items": metrics,
    }
    metrics_path = output_dir / "cleaning_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved cleaned dataset to {output_dir}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
