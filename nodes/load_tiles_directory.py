"""PhotoMosaic Tile Loader (Directory) — point at a folder of images, get a
TILE_LIBRARY back. The expected use is "drop your saved-discord-avatars
folder in here" or similar.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from comfy_api.latest import io

from ..utils.loading import list_image_files, load_tiles_from_paths
from ..utils.tile_library import TileLibrary, TileLibraryType, compute_signatures


def _preview_grid(tiles: np.ndarray, max_cells: int = 64) -> np.ndarray:
    """Pack up to max_cells tiles into a square-ish preview grid (uint8)."""
    n = min(int(tiles.shape[0]), max_cells)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    s = int(tiles.shape[1])
    canvas = np.zeros((rows * s, cols * s, 3), dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        canvas[r * s : (r + 1) * s, c * s : (c + 1) * s, :] = tiles[i]
    return canvas


class PhotoMosaicLoadTilesFromDirectory(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="nynxz.PhotoMosaic.LoadTilesFromDirectory",
            display_name="PhotoMosaic Load Tiles (Directory)",
            category="PhotoMosaic",
            description=(
                "Scan a folder of images and build a tile library for the "
                "PhotoMosaic node. Each image is square-cropped (or fit to "
                "square) and resized to tile_size."
            ),
            inputs=[
                io.String.Input(
                    "directory",
                    default="",
                    tooltip="Absolute path to a folder containing tile images.",
                ),
                io.Int.Input(
                    "tile_size",
                    default=64,
                    min=8,
                    max=512,
                    tooltip="Each tile is resized to this many pixels per side.",
                ),
                io.Combo.Input(
                    "crop_mode",
                    options=["center_crop", "fit"],
                    default="center_crop",
                    tooltip=(
                        "center_crop: take the largest centred square. "
                        "fit: letterbox onto a square (preserves the whole image)."
                    ),
                ),
                io.Boolean.Input(
                    "recursive",
                    default=False,
                    tooltip="Walk subdirectories.",
                ),
                io.Int.Input(
                    "max_tiles",
                    default=0,
                    min=0,
                    max=100000,
                    tooltip="0 = unlimited. Otherwise stop after this many files (sorted).",
                ),
            ],
            outputs=[
                TileLibraryType.Output(id="tile_library", display_name="tile_library"),
                io.Int.Output(id="tile_count", display_name="tile_count"),
                io.Image.Output(id="preview", display_name="preview"),
            ],
        )

    @classmethod
    def execute(
        cls,
        directory: str,
        tile_size: int = 64,
        crop_mode: str = "center_crop",
        recursive: bool = False,
        max_tiles: int = 0,
    ):
        directory = (directory or "").strip()
        if not directory:
            raise ValueError("PhotoMosaic: directory is empty.")

        paths = list_image_files(directory, recursive=recursive, max_tiles=max_tiles)
        if not paths:
            raise RuntimeError(f"PhotoMosaic: no images found in {directory}")

        tiles_uint8, names = load_tiles_from_paths(paths, tile_size, crop_mode)
        avg, quad = compute_signatures(tiles_uint8)
        library = TileLibrary(tiles=tiles_uint8, avg=avg, quad=quad, names=names)

        preview = _preview_grid(tiles_uint8)
        preview_tensor = torch.from_numpy(preview.astype(np.float32) / 255.0).unsqueeze(0)

        return io.NodeOutput(library, len(library), preview_tensor)
