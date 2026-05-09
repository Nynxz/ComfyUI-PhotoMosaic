"""TileLibrary — the bag of tiles plus their precomputed colour signatures
that the mosaic builder matches against. Exposed to ComfyUI as a typed
PHOTOMOSAIC_TILE_LIBRARY socket so two loader nodes can produce one and the
main PhotoMosaic node can consume it without re-scanning the source.
"""

from __future__ import annotations

import numpy as np
from comfy_api.latest._io import ComfyTypeIO, comfytype


class TileLibrary:
    """A square-cropped, fixed-size collection of tile images plus the colour
    fingerprints used for matching.

    Attributes:
        tiles:    uint8 [N, S, S, 3]   — tile pixels at S = tile_size.
        avg:      float32 [N, 3]       — per-tile mean RGB.
        quad:     float32 [N, 12]      — per-tile mean RGB of 4 quadrants
                                          (TL, TR, BL, BR), flattened.
        names:    list[str]            — original filenames (or "img_<i>"
                                          when sourced from an IMAGE batch).
    """

    def __init__(
        self,
        tiles: np.ndarray,
        avg: np.ndarray,
        quad: np.ndarray,
        names: list[str],
    ):
        self.tiles = tiles
        self.avg = avg
        self.quad = quad
        self.names = names

    @property
    def tile_size(self) -> int:
        return int(self.tiles.shape[1])

    def __len__(self) -> int:
        return int(self.tiles.shape[0])


def compute_signatures(tiles_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-tile mean RGB and per-quadrant mean RGB.

    Args:
        tiles_uint8: [N, S, S, 3] uint8.

    Returns:
        avg:  [N, 3]   float32 in 0..255.
        quad: [N, 12]  float32 in 0..255 — quadrants flattened TL,TR,BL,BR.
    """
    n, s, _, _ = tiles_uint8.shape
    f = tiles_uint8.astype(np.float32)
    avg = f.reshape(n, -1, 3).mean(axis=1)

    h = s // 2
    # Even when S is odd we drop the centre row/col — the few-pixel asymmetry
    # is irrelevant to a coarse colour signature.
    tl = f[:, :h, :h, :].reshape(n, -1, 3).mean(axis=1)
    tr = f[:, :h, h : 2 * h, :].reshape(n, -1, 3).mean(axis=1)
    bl = f[:, h : 2 * h, :h, :].reshape(n, -1, 3).mean(axis=1)
    br = f[:, h : 2 * h, h : 2 * h, :].reshape(n, -1, 3).mean(axis=1)
    quad = np.concatenate([tl, tr, bl, br], axis=1)
    return avg, quad


@comfytype(io_type="PHOTOMOSAIC_TILE_LIBRARY")
class TileLibraryType(ComfyTypeIO):
    Type = TileLibrary
