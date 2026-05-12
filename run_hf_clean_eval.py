import argparse
import json
import re
import time
from pathlib import Path
import torch
from PIL import Image
from run_hf_eval import (MODELS,SOLVE_PROMPT,TRANSCRIBE_PROMPT,ask_model,get_expression_from_output,load_model,normalize_expression,write_summary)


DATASET_DIR = Path("captcha_dataset")
NOISY_PREDICTIONS_PATH = Path("outputs/open_vlm/predictions.jsonl")
OUTPUT_ROOT = Path("outputs/open_vlm_cleaned")
CLEANED_VARIANTS = [
    ("cleaned_binary", Path("cleaned_dataset/cleaned_binary")),
    ("cleaned_gray", Path("cleaned_dataset/cleaned_gray")),
]

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command", required=True)
subparsers.add_parser("run")
args = parser.parse_args()

if args.command == "run":
    metadata_path = DATASET_DIR / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata_by_filename = {}
    for row in metadata:
        metadata_by_filename[row["filename"]] = row

    selected_filenames = []
    already_seen = set()

    with NOISY_PREDICTIONS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                filename = json.loads(line)["filename"]
                if filename not in already_seen:
                    selected_filenames.append(filename)
                    already_seen.add(filename)

    def get_file_number(filename):
        match = re.search(r"\d+", filename)
        if match:
            return int(match.group())
        return 0

    selected_filenames = sorted(selected_filenames, key=get_file_number)

    samples = []
    for filename in selected_filenames:
        row = metadata_by_filename[filename]
        samples.append(
            {
                "filename": row["filename"],
                "expression": row["expression"],
                "label": int(row["label"]),
                "difficulty": float(row["difficulty"]),
                "params": row["params"],
            }
        )

    print(f"Using {len(samples)} images from {NOISY_PREDICTIONS_PATH}")

    for variant_name, image_dir in CLEANED_VARIANTS:
        output_dir = OUTPUT_ROOT / variant_name
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions_path = output_dir / "predictions.jsonl"
        summary_path = output_dir / "summary.csv"
        predictions_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

        print(f"Running {variant_name}: {len(samples)} samples on {len(MODELS)} models")

        for model_name, model_id in MODELS:
            print(f"[{variant_name}][{model_name}] loading {model_id}")

            processor, model = load_model(model_id)

            for i, sample in enumerate(samples, start=1):
                image_path = image_dir / sample["filename"]
                image = Image.open(image_path).convert("RGB")

                transcribe_start = time.perf_counter()
                transcribe_output, _, _ = ask_model(
                    processor,
                    model,
                    image,
                    TRANSCRIBE_PROMPT,
                    max_new_tokens=24,
                )
                transcribe_latency = time.perf_counter() - transcribe_start

                solve_start = time.perf_counter()
                solve_output, _, _ = ask_model(
                    processor,
                    model,
                    image,
                    SOLVE_PROMPT,
                    max_new_tokens=16,
                )
                solve_latency = time.perf_counter() - solve_start

                expected_expression = normalize_expression(sample["expression"])
                predicted_expression = get_expression_from_output(transcribe_output)

                numbers = re.findall(r"[-+]?\d+", solve_output.replace(",", ""))

                if numbers is not None and len(numbers) > 0:
                    parsed = int(numbers[-1])
                else:
                    parsed = None

                transcription_correct = predicted_expression == expected_expression
                solve_correct = parsed == sample["label"]

                row = {
                    "image_variant": variant_name,
                    "model": model_name,
                    "model_id": model_id,
                    "filename": sample["filename"],
                    "expression": sample["expression"],
                    "label": sample["label"],
                    "difficulty": sample["difficulty"],
                    "params": sample["params"],
                    "transcribe_raw_output": transcribe_output,
                    "expected_expression": expected_expression,
                    "predicted_expression": predicted_expression,
                    "transcription_correct": transcription_correct,
                    "solve_raw_output": solve_output,
                    "raw_output": solve_output,
                    "parsed_answer": parsed,
                    "solve_correct": solve_correct,
                    "correct": solve_correct,
                    "transcribe_latency_seconds": transcribe_latency,
                    "solve_latency_seconds": solve_latency,
                    "latency_seconds": transcribe_latency + solve_latency,
                }

                with predictions_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")

                if i % 25 == 0:
                    print(
                        f"[{variant_name}][{model_name}] "
                        f"finished {i}/{len(samples)} samples"
                    )

            del model
            del processor
            torch.cuda.empty_cache()

        write_summary(predictions_path, summary_path)

        print(f"Wrote {predictions_path}")
        print(f"Wrote {summary_path}")