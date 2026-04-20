import os
import random
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from matplotlib import font_manager

OUTPUT_DIR = "captcha_dataset"
IMG_SIZE = (260, 100)
FONTS = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load font (1. Hevetica, 2. Arial, 3. default)
def get_font_path():
    if not FONTS:
        return None

    target_fonts = ["Helvetica", "helvetica", "Arial", "arial"]
    for target in target_fonts:
        for f in FONTS:
            if target in f:
                return f
    return FONTS[0]

SELECTED_PATH = get_font_path()

def load_font(size):
    if not SELECTED_PATH:
        return ImageFont.load_default()
    return ImageFont.truetype(SELECTED_PATH, size)

# Randomly generates an arithmetic expression and calculates its Ground Truth for later evaluation
def generate_expression():
    num_terms = random.choice([2, 3]) # num op num or num op num op num
    numbers = [random.randint(1, 9) for _ in range(num_terms)]
    op_candidate = [' - ', ' + ', ' * ']
    expression = str(numbers[0])

    for i in range(1, num_terms):
        op = random.choice(op_candidate)
        expression += op + str(numbers[i])

    result = eval(expression) # Ground truth
    return expression, result

# Calculates the Bounding Box dimensions (width and height) of a given string before it is drawn
def get_text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except:
        w, h = draw.textsize(text, font=font)
    return w, h

# Dynamically determines the maximum Font Size that fits within the image boundaries to maximize Readability
def get_max_font(text):
    for size in range(90, 30, -2):
        font = load_font(size)
        dummy = Image.new("RGB", IMG_SIZE)
        draw = ImageDraw.Draw(dummy)
        w, h = get_text_size(draw, text, font)

        if w < IMG_SIZE[0] * 0.9 and h < IMG_SIZE[1] * 0.8:
            return font

    return load_font(30)

# Renders the expression at the exact center of the image based on calculated coordinates
def draw_centered_text(image, text, font):
    draw = ImageDraw.Draw(image)
    w, h = get_text_size(draw, text, font)

    x = (IMG_SIZE[0] - w) // 2
    y = (IMG_SIZE[1] - h) // 2

    draw.text((x, y), text, font=font, fill=(0, 0, 0))

# Defines the Hyperparameters for noise and distortion based on the selected Difficulty level
def get_params(level):
    return {
        "gaussian_sigma": round(5 + level * 20, 2),
        "sp_ratio": round(0.01 + level * 0.05, 4),
        "blur_radius": round(0.2 + level * 1.0, 2),
        "line_count": int(1 + level * 3),
        "warp_amplitude": round(3 + level * 6, 2),
        "warp_frequency": round(0.02 + level * 0.05, 3)
    }

# Introduces Gaussian Noise across the image to simulate visual interference
def add_gaussian_noise(image, sigma):
    np_img = np.array(image).astype(np.float32)
    noise = np.random.normal(0, sigma, np_img.shape)
    noisy = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

# Applies Salt & Pepper Noise by randomly assigning pixels to extreme black or white values
def add_salt_pepper_noise(image, prob):
    np_img = np.array(image)

    rnd = np.random.rand(*np_img.shape[:2])

    # salt & pepper 
    np_img[rnd > 1 - prob] = 255
    np_img[rnd < prob] = 0

    return Image.fromarray(np_img)

# Draws random lines to obstruct the character shapes and challenge the model's Visual Encoder
def add_lines(draw, count):
    for _ in range(count):
        x1 = random.randint(0, IMG_SIZE[0])
        y1 = random.randint(0, IMG_SIZE[1])
        x2 = random.randint(0, IMG_SIZE[0])
        y2 = random.randint(0, IMG_SIZE[1])
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=1)

# Performs a Geometric Transformation by warping the image into a wave pattern
def geometric_transformation(image, amplitude, frequency):
    np_img = np.array(image)
    h, w = np_img.shape[:2]

    new_img = np.zeros_like(np_img)

    for y in range(h):
        shift = int(amplitude * np.sin(2 * np.pi * frequency * y))
        new_img[y] = np.roll(np_img[y], shift, axis=0)

    return Image.fromarray(new_img)

# Uses the Standard Deviation of pixel values to verify that the image maintains a minimum threshold of Readability
def is_readable(image, threshold=25):
    gray = np.array(image.convert("L"))
    return gray.std() > threshold

# Executes the end-to-end Pipeline from expression generation to the application of various noise types
def generate_captcha():
    while True:
        level = random.uniform(0.1, 0.7)
        params = get_params(level)
        
        expression, result = generate_expression()
        image = Image.new("RGB", IMG_SIZE, (255, 255, 255))

        # draw text
        font = get_max_font(expression)
        draw_centered_text(image, expression, font)

        # apply visual distortion
        image = geometric_transformation(
            image,
            params["warp_amplitude"],
            params["warp_frequency"]
        )

        # add line noise
        draw = ImageDraw.Draw(image)
        add_lines(draw, params["line_count"])

        # add Gaussian Noise
        if random.random() < 0.7:
            image = add_gaussian_noise(image, params["gaussian_sigma"])
        else:
            params["gaussian_sigma"] = 0  # if not applied

        # add Salt & Pepper Noise
        if random.random() < 0.5:
            image = add_salt_pepper_noise(image, params["sp_ratio"])
        else:
            params["sp_ratio"] = 0

        # apply Blur
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=params["blur_radius"]))
        else:
            params["blur_radius"] = 0

        if is_readable(image):
            meta = {
                "expression": expression,
                "label": result, # Ground Truth 
                "difficulty": round(level, 3),
                "params": params
            }
            return image, meta

# Generates the requested number of samples and saves the comprehensive Metadata in JSON
def create_dataset(NUM_SAMPLES):
    dataset_metadata = []

    for i in range(NUM_SAMPLES):
        img, meta = generate_captcha()

        filename = f"captcha_{i}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        img.save(path)

        meta["filename"] = filename
        dataset_metadata.append(meta)

        if i % 100 == 0:
            print(f"Generated {i} samples")

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f, indent=4)

    print(f"Dataset generation complete (Total: {NUM_SAMPLES})")
    print(f"Metadata saved to {os.path.join(OUTPUT_DIR, 'metadata.json')}")

if __name__ == "__main__":
    create_dataset(NUM_SAMPLES=1000)