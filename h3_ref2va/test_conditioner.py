#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Smoke-test the ref2va conditioner pieces that do not need the 33B DiT:
image normalization, prompt presentation + vision preprocessing, and the
reference VAE encode. Run with the h3_nv venv python."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref2va_conditioner as C  # noqa: E402

MODELS = os.path.expanduser("~/work/models/MiniMax-H3")


def synthetic_image(w=900, h=640):
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx / w * 255).astype(np.uint8)
    g = (yy / h * 255).astype(np.uint8)
    b = ((1 - xx / w) * 255).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1))


def main():
    from transformers import AutoTokenizer, AutoProcessor

    tok = AutoTokenizer.from_pretrained(os.path.join(MODELS, "tokenizer"), trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(os.path.join(MODELS, "processor"), trust_remote_code=True)

    img = synthetic_image()
    norm = C.normalize_image_reference(img)
    print(f"image {img.size} -> normalized {norm.size}")

    vision, counts = C.gather_image_vision(proc, [norm])
    print(f"vision token counts: {counts}")
    print(f"pixel_values {tuple(vision['pixel_values'].shape)} grid {vision['image_grid_thw'].tolist()}")

    prompt = "The character walks toward camera, cinematic lighting."
    token_ids, token_tags = C.build_presentation(tok, prompt, counts)
    print(f"presentation: {len(token_ids)} tokens, tags set {sorted(set(token_tags))}")
    # decode the start of the presentation for a sanity read
    print("presentation head:", repr(tok.decode(token_ids[:12])))
    print("presentation tail:", repr(tok.decode(token_ids[-12:])))

    mm = proc.create_mm_token_type_ids([token_ids])[0]
    print(f"mm_token_type_ids uniq {sorted(set(mm))}, len {len(mm)}")

    # VAE encode (GPU, float32)
    from diffusers import AutoencoderKLMiniMaxH3

    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        os.path.join(MODELS, "vae"), torch_dtype=torch.float32
    ).to("cuda")
    vae.eval()
    lat = C.encode_image_reference(vae, norm)
    print(f"condition latent {tuple(lat.shape)} dtype {lat.dtype} "
          f"mean {lat.mean().item():.4f} std {lat.std().item():.4f}")
    assert lat.shape[0] == 1 and lat.shape[1] == 24 and lat.shape[2] == 1
    print("OK")


if __name__ == "__main__":
    main()
