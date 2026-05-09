"""ComfyUI-PhotoMosaic — turn an image into a photomosaic of other images."""

from comfy_api.latest import ComfyExtension, io

from .nodes.load_tiles_directory import PhotoMosaicLoadTilesFromDirectory
from .nodes.load_tiles_images import PhotoMosaicLoadTilesFromImages
from .nodes.photomosaic import PhotoMosaicNode


class PhotoMosaicExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PhotoMosaicLoadTilesFromDirectory,
            PhotoMosaicLoadTilesFromImages,
            PhotoMosaicNode,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    return PhotoMosaicExtension()
