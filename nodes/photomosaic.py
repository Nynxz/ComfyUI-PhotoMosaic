"""PhotoMosaic — assemble the input image out of tile library entries."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ..utils.mosaic import build_mosaic
from ..utils.tile_library import TileLibrary, TileLibraryType


class PhotoMosaicNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="nynxz.PhotoMosaic.Build",
            display_name="PhotoMosaic",
            category="PhotoMosaic",
            description=(
                "Recreate the input image as a grid of tile images chosen "
                "from the tile library by colour similarity."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Source image to mosaic. Only the first frame is used.",
                ),
                TileLibraryType.Input("tile_library", display_name="tile_library"),
                io.Int.Input(
                    "grid_width",
                    default=80,
                    min=4,
                    max=2000,
                    tooltip="Number of tiles across the source image.",
                ),
                io.Int.Input(
                    "grid_height",
                    default=0,
                    min=0,
                    max=2000,
                    tooltip="Number of tiles vertically. 0 = derive from source aspect ratio.",
                ),
                io.Int.Input(
                    "output_tile_size",
                    default=0,
                    min=0,
                    max=512,
                    tooltip=(
                        "Pixel size of each tile in the output. 0 means use "
                        "the library's tile_size. Output resolution is "
                        "grid_width * tile × grid_height * tile."
                    ),
                ),
                io.Combo.Input(
                    "match_quality",
                    options=["fast", "quadrant"],
                    default="fast",
                    tooltip=(
                        "fast: match on tile mean colour (cheap, good enough "
                        "for most cases). quadrant: match on the four quadrant "
                        "means (better at preserving edges/structure)."
                    ),
                ),
                io.Float.Input(
                    "color_match",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip=(
                        "Shift each tile's mean toward the target cell colour. "
                        "0 = original tile colours, 1 = fully tinted to match "
                        "the source. ~0.3 is a nice sweet spot for crisp results."
                    ),
                ),
                io.Boolean.Input(
                    "allow_repeats",
                    default=True,
                    tooltip=(
                        "If False, every tile is used at most once. Falls back "
                        "to repeats automatically if the library is too small "
                        "for the grid."
                    ),
                ),
                io.Int.Input(
                    "repeat_radius",
                    default=0,
                    min=0,
                    max=64,
                    tooltip=(
                        "When repeats are allowed, prevent the same tile from "
                        "appearing within this Manhattan distance in cells. "
                        "0 = no constraint."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2**31 - 1,
                    tooltip="Tie-breaking seed for the assignment order. 0 = nondeterministic.",
                ),
            ],
            outputs=[
                io.Image.Output(id="mosaic", display_name="mosaic"),
                io.String.Output(id="info", display_name="info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        tile_library: TileLibrary,
        grid_width: int = 80,
        grid_height: int = 0,
        output_tile_size: int = 0,
        match_quality: str = "fast",
        color_match: float = 0.0,
        allow_repeats: bool = True,
        repeat_radius: int = 0,
        seed: int = 0,
    ):
        if image is None or image.shape[0] == 0:
            raise ValueError("PhotoMosaic: empty source IMAGE.")
        if not isinstance(tile_library, TileLibrary) or len(tile_library) == 0:
            raise ValueError("PhotoMosaic: tile_library is empty.")

        src = image[0].detach().cpu().numpy()
        src_uint8 = (src.clip(0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        src_h, src_w, _ = src_uint8.shape

        if grid_height <= 0:
            grid_height = max(1, round(grid_width * src_h / src_w))

        out_tile = output_tile_size if output_tile_size > 0 else tile_library.tile_size

        mosaic_uint8, info = build_mosaic(
            source_uint8=src_uint8,
            library=tile_library,
            grid_w=grid_width,
            grid_h=grid_height,
            out_tile=out_tile,
            color_match=float(color_match),
            match_quality=match_quality,
            allow_repeats=bool(allow_repeats),
            repeat_radius=int(repeat_radius),
            seed=int(seed),
        )

        out_tensor = torch.from_numpy(
            mosaic_uint8.astype(np.float32) / 255.0
        ).unsqueeze(0)

        info_str = (
            f"grid {info['grid'][0]}×{info['grid'][1]} ({info['cells']} cells), "
            f"{info['unique_tiles_used']} unique tiles used "
            f"of {info['library_size']}, "
            f"output {info['out_size'][0]}×{info['out_size'][1]} px"
        )
        return io.NodeOutput(out_tensor, info_str)
