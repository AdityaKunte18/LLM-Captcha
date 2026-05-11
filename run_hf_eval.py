import argparse
import csv
import json
import random
import re
import time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


TRANSCRIBE_PROMPT = """
Read the arithmetic expression in this CAPTCHA image.
Reply with only the expression, no answer.
"""

SOLVE_PROMPT = """
Read the arithmetic expression in this CAPTCHA image and compute its value.
Reply with only the final integer, no words.
"""

MODELS = [
    ("qwen2_vl_2b", "Qwen/Qwen2-VL-2B-Instruct"),
    ("smolvlm_500m", "HuggingFaceTB/SmolVLM-500M-Instruct"),
    ("smolvlm2_2b", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
]

DATASET_DIR = "captcha_dataset"
OUTPUT_DIR = "outputs/open_vlm"


def load_samples(dataset_dir):
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    samples = []

    for row in metadata:
        samples.append(
            {
                "filename": row["filename"],
                "expression": row["expression"],
                "label": int(row["label"]),
                "difficulty": float(row["difficulty"]),
                "params": row["params"],
                "image_path": dataset_dir / row["filename"],
            }
        )

    def get_file_number(sample):
        match = re.search(r"\d+", sample["filename"])
        if match:
            return int(match.group())
        return 0

    #sort samples by their name instead of metadata
    samples = sorted(samples, key=get_file_number)

    #take a random sample, but use the same seed so every model gets the same images
    random.seed(42)
    if len(samples) > 200:
        samples = random.sample(samples, 200)

    samples = sorted(samples, key=get_file_number)
    return samples


def load_model(model_id):
    processor = AutoProcessor.from_pretrained(model_id)

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }

    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    model.eval()

    return processor, model


def ask_model(processor, model, image, prompt, max_new_tokens=16):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    generate_args = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }

    with torch.no_grad():
        output = model.generate(**generate_args)

    answer_ids = output[:, input_len:]

    answer = processor.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return answer, inputs, output

# replace common symbols and whitespace
def normalize_expression(text):
    text = str(text).lower().strip()
    text = text.replace("×", "*")
    text = text.replace("x", "*")
    text = text.replace("−", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace(" ", "")
    text = text.replace("=", "")
    text = text.replace(".", "")
    text = text.replace(":", "")
    text = text.replace(";", "")
    text = text.replace(",", "")
    text = text.replace("'", "")
    text = text.replace('"', "")
    return text


def get_expression_from_output(text):
    text = normalize_expression(text)

    #3+4, 6-6+1, 8*3
    matches = re.findall(r"\d+(?:[+\-*]\d+)+", text)

    if len(matches) > 0:
        return matches[0]

    return text



def write_summary(predictions_path, summary_path):
    rows = []

    with Path(predictions_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    fieldnames = [
        "model",
        "total",
        "transcription_correct",
        "transcription_accuracy",
        "solve_correct",
        "solve_accuracy",
        "average_transcribe_latency_seconds",
        "average_solve_latency_seconds",
        "average_total_latency_seconds",
    ]

    summary_rows = []

    for model_name, _ in MODELS:
        model_rows = [row for row in rows if row["model"] == model_name]
        if len(model_rows) == 0:
            continue

        transcription_correct = sum(1 for row in model_rows if row["transcription_correct"])
        solve_correct = sum(1 for row in model_rows if row["solve_correct"])
        total = len(model_rows)
        avg_transcribe_latency = sum(row["transcribe_latency_seconds"] for row in model_rows) / total
        avg_solve_latency = sum(row["solve_latency_seconds"] for row in model_rows) / total
        avg_total_latency = sum(row["latency_seconds"] for row in model_rows) / total

        summary_rows.append(
            {
                "model": model_name,
                "total": total,
                "transcription_correct": transcription_correct,
                "transcription_accuracy": f"{transcription_correct / total:.4f}",
                "solve_correct": solve_correct,
                "solve_accuracy": f"{solve_correct / total:.4f}",
                "average_transcribe_latency_seconds": f"{avg_transcribe_latency:.4f}",
                "average_solve_latency_seconds": f"{avg_solve_latency:.4f}",
                "average_total_latency_seconds": f"{avg_total_latency:.4f}",
            }
        )

    with Path(summary_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def run_benchmark():
    dataset_dir = Path(DATASET_DIR)
    output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.csv"

    predictions_path.unlink(missing_ok=True)
    summary_path.unlink(missing_ok=True)

    samples = load_samples(dataset_dir)

    print(f"Running {len(samples)} samples on {len(MODELS)} models")

    for model_name, model_id in MODELS:
        print(f"[{model_name}] loading {model_id}")

        processor, model = load_model(model_id)

        for i, sample in enumerate(samples, start=1):
            image = Image.open(sample["image_path"]).convert("RGB")

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
                print(f"[{model_name}] finished {i}/{len(samples)} samples")

        #free gpu memory
        del model
        del processor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary(predictions_path, summary_path)

    print(f"Wrote {predictions_path}")
    print(f"Wrote {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run")

    args = parser.parse_args()

    if args.command == "run":
        run_benchmark()


if __name__ == "__main__":
    main()
