import re
from pathlib import Path
import json
import csv

TRANSCRIBE_PROMPT = """
Read the arithmetic expression in this CAPTCHA image.
Reply with only the expression, no answer.
"""

SOLVE_PROMPT = """
Read the arithmetic expression in this CAPTCHA image and compute its value.
Reply with only the final integer, no words.
"""

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

    for model_name in ['gpt_4o', 'gemini']:
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
