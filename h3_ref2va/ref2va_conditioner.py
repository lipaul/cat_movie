#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MiniMax-H3 `ref2va` conditioner: reference normalization, prompt presentation
and the Qwen3-VL hidden state MiniMax-H3 conditions on.

Ports `MiniMaxH3Ref2VASetupStep` (image path), `MiniMaxH3Ref2VATextEncoderStep`
and `get_qwen3vl_prompt_embeds` from the diffusers reference. Only image
references are supported here (the character-sheet use case); video/audio
references would extend `_build_presentation` and the vision gathering.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from ref2va_layout import (
    CANVAS_MULTIPLE,
    KEYFRAME_ENCODE_SEED,
    PIXEL_MEAN,
    PIXEL_STD,
    REFERENCE_IMAGE_SHORT_EDGE,
    TEXT_TAG,
    VIDEO_TAG,
)


# ---------------------------------------------------------------------------
# image reference normalization  (MiniMaxH3Ref2VASetupStep, image branch)
# ---------------------------------------------------------------------------
def normalize_image_reference(image: Image.Image) -> Image.Image:
    """A reference image onto its own 2048 short edge, axes rounded to 32.

    Encoded at high detail, with upscaling and *no* area cap, unlike the target
    canvas. The released model was conditioned on a LANCZOS resize, so the PIL
    path is kept (an array would go through `F.interpolate` instead).
    """
    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"reference image must have a positive size, got {image.size}")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"reference image must be within 1:4 and 4:1, got {width}x{height}")
    scale = REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    target_h = max(CANVAS_MULTIPLE, round(height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    target_w = max(CANVAS_MULTIPLE, round(width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    if image.size != (target_w, target_h):
        image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return image


# ---------------------------------------------------------------------------
# prompt presentation  (MiniMaxH3Ref2VATextEncoderStep._build_presentation)
# ---------------------------------------------------------------------------
def build_presentation(tokenizer, prompt: str, image_token_counts: list[int]):
    """Tokenize `"<Picture i>: " + vision block ...` then the prompt verbatim.

    Returns `(token_ids, token_tags)`. Vision-block rows carry `video_tag` (0).
    """
    text_tag, video_tag = TEXT_TAG, VIDEO_TAG

    def text(value: str):
        ids = tokenizer(value, add_special_tokens=False)["input_ids"]
        return ids, [text_tag] * len(ids)

    def vision(pad_token: str, num_tokens: int):
        ids = ([tokenizer.convert_tokens_to_ids("<|vision_start|>")]
               + [tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
               + [tokenizer.convert_tokens_to_ids("<|vision_end|>")])
        return ids, [video_tag] * len(ids)

    token_ids, token_tags = [], []

    def emit(seg):
        token_ids.extend(seg[0])
        token_tags.extend(seg[1])

    for i, count in enumerate(image_token_counts):
        emit(text(f"<Picture {i + 1}>: "))
        emit(vision("<|image_pad|>", count))
    emit(text(prompt))
    return token_ids, token_tags


def gather_image_vision(processor, images: list[Image.Image]):
    """`pixel_values` / `image_grid_thw` and the per-image vision token count."""
    merge_size = processor.image_processor.merge_size ** 2
    feats = processor.image_processor(images=images, return_tensors="pt")
    counts = [int(grid.prod()) // merge_size for grid in feats["image_grid_thw"]]
    return {"pixel_values": feats["pixel_values"], "image_grid_thw": feats["image_grid_thw"]}, counts


# ---------------------------------------------------------------------------
# Qwen3-VL conditioner  (get_qwen3vl_prompt_embeds)
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_prompt_embeds(text_encoder, processor, token_ids, vision_inputs,
                         text_encoder_layer: int = 50):
    """The `(1, num_text_tokens, 5120)` hidden state after decoder layer 50."""
    device = next(text_encoder.parameters()).device
    dtype = text_encoder.dtype
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(
        processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device
    )
    vision_kwargs = {}
    for name, value in (vision_inputs or {}).items():
        vision_kwargs[name] = value.to(device, dtype) if name.startswith("pixel_") else value.to(device)
    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
        output_hidden_states=True,
        **vision_kwargs,
    )
    return outputs.hidden_states[text_encoder_layer].to(dtype=dtype)


# ---------------------------------------------------------------------------
# reference VAE encoding  (encode_vae_condition)
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_vae_condition(vae, pixels: torch.Tensor, encode_seed: int = KEYFRAME_ENCODE_SEED) -> torch.Tensor:
    """ImageNet-normalize, encode, *sample* the posterior at seed 42, fp16-round,
    then normalize by the VAE's own latent stats. Returns a CPU float32 tensor."""
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    pixel_mean = torch.tensor(PIXEL_MEAN, device=pixels.device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD, device=pixels.device).view(1, -1, 1, 1, 1)

    pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
    posterior = vae.encode(pixels, return_dict=False)[0]
    latents = posterior.sample(generator=torch.Generator(device="cpu").manual_seed(encode_seed))
    latents = latents.to(torch.float16).float().cpu()
    return (latents - latents_mean) / latents_std


@torch.no_grad()
def encode_image_reference(vae, image: Image.Image) -> torch.Tensor:
    """A normalized reference image -> `(1, 24, 1, H, W)` normalized latent."""
    device = next(vae.parameters()).device
    pixels = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None, :, None]
    return encode_vae_condition(vae, pixels)
