"""PhotoMosaic Tile Loader (Images) — build a TILE_LIBRARY from any IMAGE
batch coming out of an upstream node (LoadImage Batch, custom loader, etc.).
"""

from __future__ import annotations

from comfy_api.latest import io

from ..utils.loading import tiles_from_image_tensor
from ..utils.tile_library import TileLibrary, TileLibraryType, compute_signatures


class PhotoMosaicLoadTilesFromImages(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="nynxz.PhotoMosaic.LoadTilesFromImages",
            display_name="PhotoMosaic Load Tiles (IMAGE Batch)",
            category="PhotoMosaic",
            description=(
                "Build a tile library from an IMAGE batch. Use this to feed "
                "the PhotoMosaic node from any upstream image source."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="An IMAGE batch — each frame becomes one tile.",
                ),
                io.Int.Input("tile_size", default=64, min=8, max=512),
                io.Combo.Input(
                    "crop_mode",
                    options=["center_crop", "fit"],
                    default="center_crop",
                ),
            ],
            outputs=[
                TileLibraryType.Output(id="tile_library", display_name="tile_library"),
                io.Int.Output(id="tile_count", display_name="tile_count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        tile_size: int = 64,
        crop_mode: str = "center_crop",
    ):
        if images is None or images.shape[0] == 0:
            raise ValueError("PhotoMosaic: empty IMAGE batch.")

        tiles_uint8, names = tiles_from_image_tensor(images, tile_size, crop_mode)
        avg, quad = compute_signatures(tiles_uint8)
        library = TileLibrary(tiles=tiles_uint8, avg=avg, quad=quad, names=names)
        return io.NodeOutput(library, len(library))
