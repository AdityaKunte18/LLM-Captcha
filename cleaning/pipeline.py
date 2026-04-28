from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

TARGET_SIZE = (260, 100)


@dataclass
class CleaningConfig:
    target_width: int = TARGET_SIZE[0]
    target_height: int = TARGET_SIZE[1]
    clahe_clip_limit: float = 2.2
    clahe_tile_grid_size: int = 8
    impulse_ratio_threshold: float = 0.012
    impulse_difference_threshold: int = 45
    denoise_strength: int = 14
    adaptive_block_size: int = 31
    adaptive_c: int = 11
    min_foreground_ratio: float = 0.02
    max_foreground_ratio: float = 0.55
    open_kernel_size: int = 2
    close_kernel_size: int = 2
    min_component_area: int = 6
    min_component_width: int = 1
    min_component_height: int = 1
    line_aspect_ratio: float = 14.0
    max_line_thickness: int = 2
    min_line_span: int = 36
    output_padding: int = 8
    enable_median_filter: bool = True
    enable_denoise: bool = True
    enable_component_filter: bool = True


def _normalize_config(config: CleaningConfig | dict[str, Any] | None) -> CleaningConfig:
    if config is None:
        return CleaningConfig()
    if isinstance(config, CleaningConfig):
        return config
    return CleaningConfig(**config)


def _to_gray_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("L"))

    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.uint8)
    if array.ndim == 3:
        return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    raise ValueError("Unsupported image format")


def _normalize_canvas(gray: np.ndarray, config: CleaningConfig) -> np.ndarray:
    target_size = (config.target_width, config.target_height)
    if gray.shape[::-1] == target_size:
        return gray.copy()
    return cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)


def _estimate_impulse_noise_ratio(gray: np.ndarray, config: CleaningConfig) -> tuple[float, np.ndarray]:
    median = cv2.medianBlur(gray, 3)
    extreme = (gray <= 12) | (gray >= 243)
    different = cv2.absdiff(gray, median) >= config.impulse_difference_threshold
    isolated = extreme & different
    return float(isolated.mean()), median


def _threshold_foreground(gray: np.ndarray, config: CleaningConfig) -> tuple[np.ndarray, str, float]:
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        config.adaptive_block_size,
        config.adaptive_c,
    )
    adaptive_ratio = float(np.count_nonzero(adaptive) / adaptive.size)

    if config.min_foreground_ratio <= adaptive_ratio <= config.max_foreground_ratio:
        return adaptive, "adaptive", adaptive_ratio

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_ratio = float(np.count_nonzero(otsu) / otsu.size)
    return otsu, "otsu", otsu_ratio


def _cleanup_mask(mask: np.ndarray, config: CleaningConfig) -> np.ndarray:
    open_kernel = np.ones((config.open_kernel_size, config.open_kernel_size), dtype=np.uint8)
    close_kernel = np.ones((config.close_kernel_size, config.close_kernel_size), dtype=np.uint8)

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
    return cleaned


def _keep_component(width: int, height: int, area: int, config: CleaningConfig) -> bool:
    if area < config.min_component_area:
        return False
    if width < config.min_component_width or height < config.min_component_height:
        return False

    thickness = min(width, height)
    aspect = max(width, height) / max(1, thickness)
    if (
        thickness <= config.max_line_thickness
        and aspect >= config.line_aspect_ratio
        and max(width, height) >= config.min_line_span
    ):
        return False

    fill_ratio = area / float(width * height)
    if fill_ratio < 0.06 and thickness <= 2 and max(width, height) >= config.min_line_span:
        return False

    return True


def _filter_components(mask: np.ndarray, config: CleaningConfig) -> tuple[np.ndarray, int, int]:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)

    kept = 0
    removed = 0
    for label in range(1, component_count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        if _keep_component(width, height, area, config):
            filtered[labels == label] = 255
            kept += 1
        else:
            removed += 1

    return filtered, kept, removed


def _crop_and_pad(gray: np.ndarray, mask: np.ndarray, config: CleaningConfig) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int] | None]:
    foreground = np.column_stack(np.where(mask > 0))
    if foreground.size == 0:
        empty_gray = np.full((config.target_height, config.target_width), 255, dtype=np.uint8)
        empty_binary = np.full((config.target_height, config.target_width), 255, dtype=np.uint8)
        return empty_gray, empty_binary, None

    top = int(foreground[:, 0].min())
    bottom = int(foreground[:, 0].max()) + 1
    left = int(foreground[:, 1].min())
    right = int(foreground[:, 1].max()) + 1

    cropped_gray = gray[top:bottom, left:right]
    cropped_mask = mask[top:bottom, left:right]

    inner_width = max(1, config.target_width - (2 * config.output_padding))
    inner_height = max(1, config.target_height - (2 * config.output_padding))
    crop_height, crop_width = cropped_gray.shape
    scale = min(1.0, inner_width / crop_width, inner_height / crop_height)

    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))

    resized_gray = cv2.resize(
        cropped_gray,
        (resized_width, resized_height),
        interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
    )
    resized_mask = cv2.resize(cropped_mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)

    gray_canvas = np.full((config.target_height, config.target_width), 255, dtype=np.uint8)
    binary_canvas = np.full((config.target_height, config.target_width), 255, dtype=np.uint8)

    y_offset = (config.target_height - resized_height) // 2
    x_offset = (config.target_width - resized_width) // 2

    gray_region = np.full_like(resized_gray, 255)
    gray_region[resized_mask > 0] = resized_gray[resized_mask > 0]

    gray_canvas[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = gray_region
    binary_canvas[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = 255 - resized_mask

    return gray_canvas, binary_canvas, (left, top, right - left, bottom - top)


def clean_image(image: Image.Image | np.ndarray, config: CleaningConfig | dict[str, Any] | None = None) -> dict[str, Any]:
    normalized_config = _normalize_config(config)

    gray = _normalize_canvas(_to_gray_array(image), normalized_config)
    impulse_ratio, median_candidate = _estimate_impulse_noise_ratio(gray, normalized_config)
    used_median = normalized_config.enable_median_filter and impulse_ratio >= normalized_config.impulse_ratio_threshold
    preprocessed = median_candidate if used_median else gray

    clahe = cv2.createCLAHE(
        clipLimit=normalized_config.clahe_clip_limit,
        tileGridSize=(normalized_config.clahe_tile_grid_size, normalized_config.clahe_tile_grid_size),
    )
    normalized = clahe.apply(preprocessed)

    denoised = normalized
    if normalized_config.enable_denoise:
        denoised = cv2.fastNlMeansDenoising(normalized, None, normalized_config.denoise_strength, 7, 21)

    binary_mask, threshold_method, threshold_ratio = _threshold_foreground(denoised, normalized_config)
    morphed_mask = _cleanup_mask(binary_mask, normalized_config)

    components_before = max(0, cv2.connectedComponents(morphed_mask, connectivity=8)[0] - 1)
    components_kept = components_before
    components_removed = 0
    filtered_mask = morphed_mask
    if normalized_config.enable_component_filter:
        filtered_mask, components_kept, components_removed = _filter_components(morphed_mask, normalized_config)

    gray_clean, binary_clean, bbox = _crop_and_pad(denoised, filtered_mask, normalized_config)
    foreground_ratio_after = float(np.count_nonzero(filtered_mask) / filtered_mask.size)

    return {
        "gray_clean": gray_clean,
        "binary_clean": binary_clean,
        "stats": {
            "config": asdict(normalized_config),
            "impulse_ratio": round(impulse_ratio, 6),
            "used_median_filter": used_median,
            "threshold_method": threshold_method,
            "foreground_ratio_initial": round(threshold_ratio, 6),
            "foreground_ratio_final": round(foreground_ratio_after, 6),
            "components_before_filter": int(components_before),
            "components_kept": int(components_kept),
            "components_removed": int(components_removed),
            "crop_bbox": bbox,
        },
    }
