#Please set your environment variable for OPENAI_API_KEY and GEMINI_API_KEY first

import argparse
import json
import re
import time
from pathlib import Path
from tqdm import tqdm
from utils import *

DATASET_DIR = "/Users/rdi/Desktop/LLM-Captcha/captcha_dataset/captcha_dataset"
OUTPUT_DIR = "outputs/Frontier_model_outputs"

### Helped with Quick Start from OpenAI: https://developers.openai.com/api/docs/quickstart?lang=python ### 
def ask_GPT(current_img_path, question):
    from openai import OpenAI
    import base64

    client = OpenAI()

    with open(current_img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    current_img_path_url = f"data:image/png;base64,{b64}"

    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": question,
                    },
                    {
                        "type": "input_image",
                        "image_url": current_img_path_url
                    }
                ]
            }
        ]
    )

    return response.output_text

### Helped with Quick Start from Google API: https://ai.google.dev/gemini-api/docs/image-understanding ### 
def ask_Gemini(current_img_path, question):
    from google import genai
    from google.genai import types

    client = genai.Client()

    with open(current_img_path, 'rb') as f:
        current_img_bytes = f.read()

    # Create the prompt with text and multiple images
    response = client.models.generate_content(

        model="gemini-3-flash-preview",
        contents=[
            question,
            types.Part.from_bytes(
                data=current_img_bytes,
                mime_type='image/png'
            )
        ]
    )

    return response.text

def run_benchmark(provider=None, write_summary_only=None):
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(DATASET_DIR)

    which_thing_to_run = None
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.csv"

    if write_summary_only:
        if not predictions_path.exists():
            raise FileNotFoundError(f"Prediction does not exist yet. Run benchmark first.")
        write_summary(predictions_path, summary_path)
        print("Finish Write Summary Only.")
        return
    
    if provider is None:
        which_thing_to_run = ["gpt_4o", "gemini"]
    else:    
        which_thing_to_run = [provider]
    
    for model in which_thing_to_run:
        for sample in tqdm(samples, desc=f"[{model}]"):

            current_img_path = sample["image_path"]

            transcribe_start = time.perf_counter()
            if model == "gpt_4o":
                understanding_from_model = ask_GPT(current_img_path, TRANSCRIBE_PROMPT)
            else:
                understanding_from_model = ask_Gemini(current_img_path, TRANSCRIBE_PROMPT)
            transcribe_latency = time.perf_counter() - transcribe_start

            transcribe_start = time.perf_counter()
            if model == "gpt_4o":
                answer_from_model = ask_GPT(current_img_path, SOLVE_PROMPT)
            else:
                answer_from_model = ask_Gemini(current_img_path, SOLVE_PROMPT)
            solve_latency = time.perf_counter() - transcribe_start

            expected_expression = normalize_expression(sample["expression"])
            predicted_expression = get_expression_from_output(understanding_from_model)

            numbers = re.findall(r"[-+]?\d+", answer_from_model.replace(",", ""))

            if numbers is not None and len(numbers) > 0:
                parsed = int(numbers[-1])
            else:
                parsed = None

            transcription_correct = predicted_expression == expected_expression
            solve_correct = parsed == sample["label"]

            row = {
                "model": model,
                "filename": sample["filename"],
                "expression": sample["expression"],
                "label": sample["label"],
                "difficulty": sample["difficulty"],
                "params": sample["params"],
                "transcribe_raw_output": understanding_from_model,
                "expected_expression": expected_expression,
                "predicted_expression": predicted_expression,
                "transcription_correct": transcription_correct,
                "solve_raw_output": answer_from_model,
                "raw_output": answer_from_model,
                "parsed_answer": parsed,
                "solve_correct": solve_correct,
                "correct": solve_correct,
                "transcribe_latency_seconds": transcribe_latency,
                "solve_latency_seconds": solve_latency,
                "latency_seconds": transcribe_latency + solve_latency,
            }

            with predictions_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    write_summary(predictions_path, summary_path)

    print(f"Wrote {predictions_path}")
    print(f"Wrote {summary_path}")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--provider",
        choices=["gpt_4o", "gemini"],
        default=None,
        help="Run only one provider's models (default: run all)."
    )
    run_parser.add_argument(
        "--write_summary_only",
        action="store_true",
        help="Skip the benchmark and just rebuild summary.csv from existing predictions.jsonl.",
    )
    args = parser.parse_args()
 
    if args.command == "run":
        run_benchmark(provider=args.provider, write_summary_only=args.write_summary_only)     

if __name__ == "__main__":
    main()
