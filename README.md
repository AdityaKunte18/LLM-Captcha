# LLM-Captcha

## Environment & Prerequisites
- **Python Version**: `3.11.15`
- **Core Dependencies**:
  - `numpy`: For numerical operations and noise generation
  - `pillow`: For image rendering and filter applications
  - `matplotlib`: For system font management and path resolution
  - `opencv-python`: For cleaning, morphology, connected-component analysis, and proxy evaluation

### Dataset Generation Features
The generator creates CAPTCHA images by interleaving mathematical expressions with multiple layers of visual distortion.


1. Arithmetic Expressions
     - Operands: Single-digit integers (1-9).
    - Terms: 2-term (e.g., a + b) or 3-term (e.g., a * b - c) expressions.
    - Operators: Addition (+), Subtraction (-), and Multiplication (*).
    - Ground Truth: Calculated following the standard order of operations.

2. Visual Noise & Augmentation
      - Geometric Transformation: Sinusoidal warping to distort character shapes.
      - Gaussian Noise: Simulates electronic sensor interference.
      - Salt & Pepper Noise: Random pixel-level impulse noise.
      - Line Noise: Randomly drawn lines to obstruct character recognition.

3. Difficulty Scaling
    -  The difficulty parameter (ranging from 0.1 to 0.7) acts as a Global Multiplier. This parameter allows you to adjust the intensity of all noise types simultaneously, enabling identification of the model's failure threshold.


Generated samples are saved in the captcha_dataset/ directory
  - .png files: The visual CAPTCHA samples.
  - metadata.json: A comprehensive log for Evaluation containing:
    - filename: Associated image file.
    - expression: The raw mathematical string.
    - label: The evaluated Ground Truth.
    - difficulty: The noise intensity level used.
    - params: Exact hyperparameters for every noise type.

## Cleaning Pipeline
The `cleaning/` package adds a classical CV preprocessing pipeline for denoising and normalizing noisy CAPTCHA images.

- `cleaning/pipeline.py`: Single-image cleaner that outputs grayscale and binary cleaned images plus per-image stats.
- `cleaning/run_batch.py`: Batch processor that writes `cleaned_gray/`, `cleaned_binary/`, and `cleaning_metrics.json`.
- `cleaning/reference.py`: Clean-reference renderer used for proxy evaluation without modifying the generator.
- `cleaning/evaluate.py`: Proxy evaluator that regenerates clean references from the stored expression and compares raw vs cleaned variants.

### Run the pipeline
```bash
python3 -m cleaning.run_batch --dataset-dir captcha_dataset --output-dir cleaned_dataset
python3 -m cleaning.evaluate --dataset-dir captcha_dataset --cleaned-dir cleaned_dataset
```

### Run the Frontier Model Evaluation
- `legacy_model_evaluation.py`: run GPT-4o or Gemini-3-flash-preview to output the evaluation. 
```bash
python legacy_model_evaluation.py run --provider gemini
python legacy_model_evaluation.py run --provider gpt_4o
```


### For experiment with HF models

1. !pip install -q -U transformers accelerate bitsandbytes qwen-vl-utils pillow
2. Clone the repo / mount
3. Unzip the dataset first so `captcha_dataset/metadata.json` and the PNG files are present

```
python run_hf_eval.py run
```

The benchmark asks each model to do two tasks:
- transcribe the CAPTCHA expression
- solve the CAPTCHA expression
