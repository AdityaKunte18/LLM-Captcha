from __future__ import annotations

from PIL import Image

from generate_captcha import IMG_SIZE, draw_centered_text, get_max_font


def render_clean_reference(expression: str, image_size: tuple[int, int] = IMG_SIZE) -> Image.Image:
    image = Image.new("RGB", image_size, (255, 255, 255))
    font = get_max_font(expression)
    draw_centered_text(image, expression, font)
    return image
