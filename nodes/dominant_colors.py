"""PhotoMosaic Dominant Colors — extract the top N colours from an image and
emit them as an IMAGE batch. Output mode controls whether each frame is a
solid-fill swatch or the per-cluster region of the source.
"""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io
from PIL import Image


class PhotoMosaicDominantColors(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="nynxz.PhotoMosaic.DominantColors",
            display_name="PhotoMosaic Dominant Colors",
            category="PhotoMosaic",
            description=(
                "Median-cut quantization of an image into its top N colours, "
                "emitted as an IMAGE batch you can save or feed elsewhere."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Int.Input(
                    "n_colors",
                    default=5,
                    min=1,
                    max=64,
                    tooltip="How many dominant colours to extract.",
                ),
                io.Combo.Input(
                    "mode",
                    options=["swatches", "layers"],
                    default="swatches",
                    tooltip=(
                        "swatches: N solid-fill frames at source resolution, "
                        "ordered by frequency (most-common first).\n"
                        "layers: N frames, each one shows only the source "
                        "pixels assigned to that colour (others black)."
                    ),
                ),
                io.Int.Input(
                    "swatch_size",
                    default=0,
                    min=0,
                    max=2048,
                    tooltip=(
                        "[swatches mode only] Force swatch frames to this "
                        "square size. 0 = use source resolution."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(id="images", display_name="images"),
                io.String.Output(id="palette_hex", display_name="palette_hex"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        n_colors: int = 5,
        mode: str = "swatches",
        swatch_size: int = 0,
    ):
        if image is None or image.shape[0] == 0:
            raise ValueError("PhotoMosaicDominantColors: empty IMAGE.")

        src = image[0].detach().cpu().numpy()
        h, w, _ = src.shape
        src_uint8 = (src.clip(0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        pil = Image.fromarray(src_uint8, mode="RGB")

        quant = pil.quantize(colors=int(n_colors))  # median-cut by default
        palette_full = quant.getpalette() or []
        palette_full = (palette_full + [0] * (256 * 3))[: 256 * 3]

        # Sort palette indices by frequency (most-common first).
        counts = sorted(quant.getcolors() or [], key=lambda c: -c[0])
        idx_order = [idx for _cnt, idx in counts]
        rgb = np.array(
            [palette_full[i * 3 : (i + 1) * 3] for i in idx_order],
            dtype=np.uint8,
        )
        n = int(rgb.shape[0])
        palette_hex = ", ".join("#{:02X}{:02X}{:02X}".format(*c) for c in rgb)

        if mode == "swatches":
            sh, sw = (swatch_size, swatch_size) if swatch_size > 0 else (h, w)
            out = np.broadcast_to(
                rgb[:, None, None, :], (n, sh, sw, 3)
            ).copy()
        elif mode == "layers":
            quant_idx = np.asarray(quant, dtype=np.int32)
            out = np.zeros((n, h, w, 3), dtype=np.uint8)
            for i, original_idx in enumerate(idx_order):
                mask = quant_idx == original_idx
                out[i][mask] = rgb[i]
        else:
            raise ValueError(f"unknown mode: {mode!r}")

        out_tensor = torch.from_numpy(out.astype(np.float32) / 255.0)
        return io.NodeOutput(out_tensor, palette_hex)
