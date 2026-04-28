import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


PROMPT = """
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
    return samples[:200]


def load_model(model_id, output_attentions=False):
    processor = AutoProcessor.from_pretrained(model_id)

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }

    if output_attentions:
        model_kwargs["attn_implementation"] = "eager" #for getting attention weights

    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    model.eval()

    return processor, model


def ask_model(processor, model, image, max_new_tokens=16, output_attentions=False):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
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

    if output_attentions:
        generate_args["return_dict_in_generate"] = True
        generate_args["output_attentions"] = True

    with torch.no_grad():
        output = model.generate(**generate_args)

    if output_attentions:
        sequences = output.sequences
    else:
        sequences = output

    answer_ids = sequences[:, input_len:]

    answer = processor.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return answer, inputs, output


def write_summary(predictions_path, summary_path):
    rows = []

    with Path(predictions_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    fieldnames = [
        "model",
        "total",
        "correct",
        "accuracy",
        "average_latency_seconds",
    ]

    summary_rows = []

    for model_name, _ in MODELS:
        model_rows = [row for row in rows if row["model"] == model_name]
        if len(model_rows) == 0:
            continue

        correct = sum(1 for row in model_rows if row["correct"])
        total = len(model_rows)
        avg_latency = sum(row["latency_seconds"] for row in model_rows) / total

        summary_rows.append(
            {
                "model": model_name,
                "total": total,
                "correct": correct,
                "accuracy": f"{correct / total:.4f}",
                "average_latency_seconds": f"{avg_latency:.4f}",
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
            start = time.perf_counter()

            image = Image.open(sample["image_path"]).convert("RGB")
            raw_output, _, _ = ask_model(processor, model, image)

            latency = time.perf_counter() - start

            numbers = re.findall(r"[-+]?\d+", raw_output.replace(",", ""))

            if numbers:
                parsed = int(numbers[-1])
            else:
                parsed = None

            row = {
                "model": model_name,
                "model_id": model_id,
                "filename": sample["filename"],
                "expression": sample["expression"],
                "label": sample["label"],
                "difficulty": sample["difficulty"],
                "params": sample["params"],
                "raw_output": raw_output,
                "parsed_answer": parsed,
                "correct": parsed == sample["label"],
                "latency_seconds": latency,
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


def run_attention():
    dataset_dir = Path(DATASET_DIR)
    output_dir = Path(OUTPUT_DIR) / "attention"

    output_dir.mkdir(parents=True, exist_ok=True)

    model_name, model_id = MODELS[1]

    samples = load_samples(dataset_dir)
    samples_by_file = {sample["filename"]: sample for sample in samples}

    predictions_path = Path(OUTPUT_DIR) / "predictions.jsonl"

    correct_rows = []

    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if row["model"] == model_name and row["correct"]:
                correct_rows.append(row)

    selected_samples = []

    for row in correct_rows[:12]:
        selected_samples.append(samples_by_file[row["filename"]])

    processor, model = load_model(model_id, output_attentions=True)

    results = []

    for sample in selected_samples:
        image = Image.open(sample["image_path"]).convert("RGB")

        raw_output, inputs, output = ask_model(
            processor,
            model,
            image,
            max_new_tokens=4,
            output_attentions=True,
        )

        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[-1]

        tokens = processor.tokenizer.convert_ids_to_tokens(
            input_ids[0].detach().cpu().tolist()
        )

        image_positions = []

        for i, token in enumerate(tokens):
            token = token.lower()

            if "image" in token or "vision" in token or "img" in token:
                image_positions.append(i)

        last_step_attention = output.attentions[-1]
        last_layer_attention = last_step_attention[-1]

        #get average over attention heads
        attention = (last_layer_attention[0, :, -1, :].mean(dim=0).detach().float().cpu().numpy())

        attention = attention[:input_len]
        image_attention = attention[image_positions]

        stem = Path(sample["filename"]).stem

        npz_path = output_dir / f"{stem}_{model_name}_attention.npz"
        heatmap_path = output_dir / f"{stem}_{model_name}_heatmap.png"

        np.savez_compressed(
            npz_path,
            attention_to_input=attention,
            image_token_positions=np.asarray(image_positions),
            attention_to_image_tokens=image_attention,
            filename=sample["filename"],
            model=model_name,
        )

        heatmap_result = None

        token_count = len(image_attention)

        if token_count > 0:
            cols = math.ceil(math.sqrt(token_count))
            rows = math.ceil(token_count / cols)

            padded = np.full(rows * cols, np.nan, dtype=float)
            padded[:token_count] = image_attention

            grid = padded.reshape(rows, cols)

            plt.figure(figsize=(4, 4))
            plt.imshow(grid, cmap="magma")
            plt.title(f"{model_name}: {sample['filename']}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(heatmap_path)
            plt.close()

            heatmap_result = str(heatmap_path)

        results.append(
            {
                "filename": sample["filename"],
                "raw_output": raw_output,
                "npz": str(npz_path),
                "heatmap": heatmap_result,
                "image_token_count": len(image_positions),
            }
        )

    note = {
        "model": model_name,
        "model_id": model_id,
        "samples_processed": len(results),
        "results": results,
    }

    note_path = output_dir / "attention_note.json"
    note_path.write_text(json.dumps(note, indent=2), encoding="utf-8")

    del model
    del processor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Wrote attention files to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run")
    subparsers.add_parser("attention")

    args = parser.parse_args()

    if args.command == "run":
        run_benchmark()
    elif args.command == "attention":
        run_attention()


if __name__ == "__main__":
    main()