"""Image loading helpers — directory scan and IMAGE-tensor unpacking, both
ending in a uint8 [N, S, S, 3] tile array ready for signature computation.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"}


def list_image_files(directory: str, recursive: bool, max_tiles: int) -> list[str]:
    """Return sorted list of image file paths in `directory`. `max_tiles=0`
    means unlimited."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Tile directory does not exist: {root}")

    paths: list[str] = []
    if recursive:
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if Path(f).suffix.lower() in _IMAGE_EXTS:
                    paths.append(os.path.join(dirpath, f))
    else:
        for entry in os.scandir(root):
            if entry.is_file() and Path(entry.name).suffix.lower() in _IMAGE_EXTS:
                paths.append(entry.path)

    paths.sort()
    if max_tiles and max_tiles > 0:
        paths = paths[:max_tiles]
    return paths


def _square_crop_center(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def _square_fit(img: Image.Image, fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Letterbox onto a square canvas, preserving aspect."""
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), fill)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def load_tiles_from_paths(
    paths: list[str],
    tile_size: int,
    crop_mode: str,
) -> tuple[np.ndarray, list[str]]:
    """Load and square-resize every path to `tile_size`. Skips files that
    fail to decode and prints one line per skip — better to keep going than
    abort on a single corrupt JPEG in a 5000-file folder."""
    rows: list[np.ndarray] = []
    names: list[str] = []
    for p in paths:
        try:
            with Image.open(p) as raw:
                img = ImageOps.exif_transpose(raw).convert("RGB")
        except Exception as e:
            print(f"[PhotoMosaic] skipping {p}: {e}")
            continue
        if crop_mode == "fit":
            img = _square_fit(img)
        else:
            img = _square_crop_center(img)
        img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        rows.append(np.asarray(img, dtype=np.uint8))
        names.append(os.path.basename(p))
    if not rows:
        raise RuntimeError("No usable images were loaded.")
    return np.stack(rows, axis=0), names


def tiles_from_image_tensor(
    images,  # torch.Tensor [B, H, W, 3] float32 0..1
    tile_size: int,
    crop_mode: str,
) -> tuple[np.ndarray, list[str]]:
    """Convert a ComfyUI IMAGE batch into the same uint8 [N, S, S, 3] format
    used by the directory loader."""
    arr = (images.detach().cpu().numpy().clip(0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    rows: list[np.ndarray] = []
    names: list[str] = []
    for i in range(arr.shape[0]):
        img = Image.fromarray(arr[i], mode="RGB")
        if crop_mode == "fit":
            img = _square_fit(img)
        else:
            img = _square_crop_center(img)
        img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        rows.append(np.asarray(img, dtype=np.uint8))
        names.append(f"img_{i}")
    return np.stack(rows, axis=0), names
