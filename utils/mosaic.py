"""Mosaic builder — given a source image and a TileLibrary, pick a tile per
grid cell and composite the result.

Matching is squared L2 in RGB (or RGB-per-quadrant for the "quadrant" mode).
Repeat avoidance is a Manhattan-radius constraint enforced greedily during
assignment. The colour-match knob shifts each placed tile's mean toward its
target cell mean while preserving the tile's internal structure.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .tile_library import TileLibrary


def _grid_cell_signatures(
    src_uint8: np.ndarray,
    grid_w: int,
    grid_h: int,
    quadrant: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize the source to a grid where each grid cell becomes either 1 px
    (for fast/avg matching) or 2 px (for quadrant matching), then read the
    pixel values directly. This is dramatically faster than averaging
    arbitrary-sized cell regions in Python and produces the same answer up
    to LANCZOS resampling.

    Returns:
        cell_avg:  [G, 3]   float32 0..255 where G = grid_w * grid_h
        cell_sig:  [G, K]   float32 0..255 where K = 12 (quadrant) or 3 (avg)
    """
    src_pil = Image.fromarray(src_uint8, mode="RGB")
    avg_arr = np.asarray(
        src_pil.resize((grid_w, grid_h), Image.Resampling.LANCZOS),
        dtype=np.float32,
    ).reshape(-1, 3)

    if not quadrant:
        return avg_arr, avg_arr

    quad_arr = np.asarray(
        src_pil.resize((grid_w * 2, grid_h * 2), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    # Read each cell's 4 quadrants (TL, TR, BL, BR) by simple striding.
    tl = quad_arr[0::2, 0::2, :].reshape(-1, 3)
    tr = quad_arr[0::2, 1::2, :].reshape(-1, 3)
    bl = quad_arr[1::2, 0::2, :].reshape(-1, 3)
    br = quad_arr[1::2, 1::2, :].reshape(-1, 3)
    sig = np.concatenate([tl, tr, bl, br], axis=1)
    return avg_arr, sig


def _assign_tiles(
    cell_sig: np.ndarray,    # [G, K]
    tile_sig: np.ndarray,    # [N, K]
    grid_w: int,
    grid_h: int,
    allow_repeats: bool,
    repeat_radius: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Greedy nearest-tile-per-cell with optional repeat constraints."""
    g = grid_w * grid_h
    n = tile_sig.shape[0]

    # Squared L2 distance, vectorised.
    # ||c - t||^2 = ||c||^2 + ||t||^2 - 2 c·t   →  but a single matmul is fine.
    c_sq = (cell_sig**2).sum(axis=1, keepdims=True)        # [G, 1]
    t_sq = (tile_sig**2).sum(axis=1, keepdims=True).T      # [1, N]
    dist = c_sq + t_sq - 2.0 * cell_sig @ tile_sig.T       # [G, N]

    if allow_repeats and repeat_radius <= 0:
        return np.argmin(dist, axis=1)

    # Walk cells in a randomised order so repeat constraints don't all bias
    # toward the top-left.
    order = rng.permutation(g)
    assignment = np.full(g, -1, dtype=np.int64)
    used_global: set[int] = set()
    placements: list[tuple[int, int, int]] = []  # (tile_idx, gx, gy)

    for cell in order:
        gy, gx = divmod(int(cell), grid_w)
        row = dist[cell]
        order_by_dist = np.argsort(row, kind="stable")
        chosen = -1
        for cand in order_by_dist:
            cand = int(cand)
            if not allow_repeats and cand in used_global:
                continue
            if repeat_radius > 0:
                ok = True
                for (used_t, ux, uy) in placements:
                    if used_t == cand and abs(ux - gx) + abs(uy - gy) <= repeat_radius:
                        ok = False
                        break
                if not ok:
                    continue
            chosen = cand
            break
        if chosen < 0:
            # Library too small to satisfy constraint — fall back to nearest.
            chosen = int(order_by_dist[0])
        assignment[cell] = chosen
        used_global.add(chosen)
        if repeat_radius > 0:
            placements.append((chosen, gx, gy))

        if not allow_repeats and len(used_global) >= n:
            # Every tile is now taken; remaining cells must allow repeats.
            allow_repeats = True

    return assignment


def _composite(
    library: TileLibrary,
    assignment: np.ndarray,
    cell_avg: np.ndarray,
    grid_w: int,
    grid_h: int,
    out_tile: int,
    color_match: float,
) -> np.ndarray:
    """Place each chosen tile (resized to out_tile) into the output canvas,
    optionally shifting its mean toward the target cell colour."""
    src_tiles = library.tiles
    s = library.tile_size
    if out_tile != s:
        # One bulk PIL resize per unique tile would be ideal, but assignment
        # often picks the same tile for many cells — resize on demand and
        # cache for the duration of this composite.
        cache: dict[int, np.ndarray] = {}

        def get_tile(idx: int) -> np.ndarray:
            t = cache.get(idx)
            if t is None:
                t = np.asarray(
                    Image.fromarray(src_tiles[idx]).resize(
                        (out_tile, out_tile), Image.Resampling.LANCZOS
                    ),
                    dtype=np.uint8,
                )
                cache[idx] = t
            return t
    else:
        def get_tile(idx: int) -> np.ndarray:
            return src_tiles[idx]

    out = np.empty((grid_h * out_tile, grid_w * out_tile, 3), dtype=np.uint8)
    for cell, tile_idx in enumerate(assignment):
        gy, gx = divmod(cell, grid_w)
        tile = get_tile(int(tile_idx))
        if color_match > 0.0:
            target = cell_avg[cell]                      # [3]
            tile_mean = tile.reshape(-1, 3).mean(axis=0)
            shift = color_match * (target - tile_mean)   # [3]
            tile = np.clip(
                tile.astype(np.float32) + shift, 0.0, 255.0
            ).astype(np.uint8)
        y0 = gy * out_tile
        x0 = gx * out_tile
        out[y0 : y0 + out_tile, x0 : x0 + out_tile, :] = tile
    return out


def build_mosaic(
    source_uint8: np.ndarray,        # [H, W, 3] uint8
    library: TileLibrary,
    grid_w: int,
    grid_h: int,
    out_tile: int,
    color_match: float,
    match_quality: str,              # "fast" or "quadrant"
    allow_repeats: bool,
    repeat_radius: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Return the mosaic uint8 [H', W', 3] plus a small info dict."""
    quadrant = match_quality == "quadrant"
    cell_avg, cell_sig = _grid_cell_signatures(
        source_uint8, grid_w, grid_h, quadrant
    )
    tile_sig = library.quad if quadrant else library.avg

    rng = np.random.default_rng(seed if seed > 0 else None)
    assignment = _assign_tiles(
        cell_sig=cell_sig,
        tile_sig=tile_sig,
        grid_w=grid_w,
        grid_h=grid_h,
        allow_repeats=allow_repeats,
        repeat_radius=max(0, repeat_radius),
        rng=rng,
    )
    out = _composite(
        library=library,
        assignment=assignment,
        cell_avg=cell_avg,
        grid_w=grid_w,
        grid_h=grid_h,
        out_tile=out_tile,
        color_match=color_match,
    )
    info = {
        "grid": (grid_w, grid_h),
        "cells": grid_w * grid_h,
        "unique_tiles_used": int(np.unique(assignment).size),
        "library_size": len(library),
        "out_size": (out.shape[1], out.shape[0]),
    }
    return out, info
