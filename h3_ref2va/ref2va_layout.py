#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Native (torch-only) MiniMax-H3 `ref2va` layout + schedule.

A faithful port of the diffusers reference (`diffusers/modular_pipelines/
minimax_h3/`): the packed-sequence layout, its fp64 rotary grid, the
row-timestep plan and the two rectified-flow schedulers. No diffusers import,
no model — `verify_layout.py` checks every output against the oracle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

# ---- released-checkpoint contract -----------------------------------------
FPS = 24
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_CHANNELS = 2
AUDIO_LATENT_CHANNELS = 32
TEXT_TAG, VIDEO_TAG, AUDIO_TAG = 1, 0, 2
MODALITY_NUM = 3
VAE_FRAMES_PER_CHUNK = 17
VAE_LATENTS_PER_CHUNK = 5
VAE_SPATIAL_RATIO = 16
CANVAS_MULTIPLE = 32
CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 768 * 1344
REFERENCE_IMAGE_SHORT_EDGE = 2048
FLOW_SHIFT_VIDEO = 12.0
FLOW_SHIFT_AUDIO = 3.0
KEYFRAME_NOISE_AUG = 0.999
KEYFRAME_ENCODE_SEED = 42
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)

_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


# ---- geometry helpers ------------------------------------------------------
def align_num_frames(n: int) -> int:
    while n % VAE_FRAMES_PER_CHUNK != VAE_LATENTS_PER_CHUNK:
        n += 1
    return n


def video_latent_num_frames(num_frames: int) -> int:
    return (num_frames - VAE_LATENTS_PER_CHUNK) // VAE_FRAMES_PER_CHUNK * VAE_LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))


def resolve_canvas_size(width: float, height: float) -> tuple[int, int]:
    """MiniMax-H3's own canvas rule: 768 short edge, 768*1344 area cap, /32."""
    ratio = width / height
    if ratio >= 1:
        w, h = CANVAS_SHORT_EDGE * ratio, float(CANVAS_SHORT_EDGE)
    else:
        w, h = float(CANVAS_SHORT_EDGE), CANVAS_SHORT_EDGE / ratio
    if w * h > CANVAS_MAX_PIXELS:
        sc = (CANVAS_MAX_PIXELS / (w * h)) ** 0.5
        w, h = w * sc, h * sc
    return (
        max(CANVAS_MULTIPLE, round(w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _frame_position_grid(latent_height, latent_width, patch_h, patch_w):
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
         for i in range(num_latent_frames)],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _video_span(num_latent_frames: int) -> float:
    # sequential sum, exactly as the reference (not the pairwise variant)
    return sum(
        _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
        for i in range(num_latent_frames)
    )


def _fill_audio_positions(position_ids, rows, num_audio_latents, rotary_time, width_grid, audio_channels):
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(audio_channels)
    position_ids[rows, 2] = torch.cat([
        torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
        torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
    ])


# ---- reference descriptors (mirror diffusers' reference dataclasses) -------
@dataclass
class Ref:
    kind: str                 # "image" | "video" | "audio"
    has_audio: bool = False
    # geometry of the *encoded* latent (image/video only)
    latent_frames: int = 1
    latent_height: int = 0
    latent_width: int = 0
    audio_latents: int = 0    # per channel

    @staticmethod
    def image(latent_frames, h, w):
        return Ref("image", False, latent_frames, h, w, 0)

    @staticmethod
    def audio(audio_latents):
        return Ref("audio", True, 0, 0, 0, audio_latents)

    @staticmethod
    def video(latent_frames, h, w, audio_latents=0):
        return Ref("video", audio_latents > 0, latent_frames, h, w, audio_latents)


# ---- packed layout ---------------------------------------------------------
def build_ref2va_packed_sequence(
    text_token_tags: torch.Tensor,
    references: list[Ref],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size=(1, 2, 2),
    audio_channels: int = AUDIO_CHANNELS,
    audio_tag: int = AUDIO_TAG,
    video_tag: int = VIDEO_TAG,
):
    """Port of `MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence`.

    Geometry comes from each reference's *encoded latent shape* (carried on
    `Ref`), not from the raw media.
    """
    _, patch_h, patch_w = patch_size
    num_text_tokens = int(text_token_tags.shape[0])
    num_target_video_rows = num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
    num_target_audio_rows = num_audio_latents * audio_channels

    num_reference_video_rows = sum(
        r.latent_frames * (r.latent_height // patch_h) * (r.latent_width // patch_w)
        for r in references if r.kind in ("image", "video")
    )
    num_reference_audio_rows = sum(
        r.audio_latents * audio_channels
        for r in references if r.kind == "audio" or (r.kind == "video" and r.has_audio)
    )
    sequence_length = (
        num_text_tokens + num_reference_video_rows + num_reference_audio_rows
        + num_target_audio_rows + num_target_video_rows
    )

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    target_frame_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    video_indices, audio_indices = [], []
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for r in references:
        if r.kind == "image":
            num_video_rows = r.latent_frames * (r.latent_height // patch_h) * (r.latent_width // patch_w)
            rows = slice(cursor, cursor + num_video_rows)
            cursor = rows.stop
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(r.latent_height, r.latent_width, patch_h, patch_w)
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            rotary_time += 1.0
        elif r.kind == "audio":
            num_audio_rows = r.audio_latents * audio_channels
            rows = slice(cursor, cursor + num_audio_rows)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(position_ids, rows, r.audio_latents, rotary_time,
                                  target_width_grid, audio_channels)
            rotary_time += float(r.audio_latents)
        elif r.kind == "video":
            num_audio_rows = r.audio_latents * audio_channels if r.has_audio else 0
            num_video_rows = r.latent_frames * (r.latent_height // patch_h) * (r.latent_width // patch_w)
            audio_rows = slice(cursor, cursor + num_audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + num_video_rows)
            cursor = video_rows.stop
            if num_audio_rows:
                audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(torch.arange(video_rows.start, video_rows.stop))

            frame_grid, width_grid = _frame_position_grid(r.latent_height, r.latent_width, patch_h, patch_w)
            if num_audio_rows:
                _fill_audio_positions(position_ids, audio_rows, r.audio_latents, rotary_time,
                                      width_grid, audio_channels)
            frame_time = _temporal_position_grid(r.latent_frames, rotary_time)
            position_ids[video_rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
            position_ids[video_rows, 1:] = frame_grid.repeat(r.latent_frames, 1)
            rotary_time += max(float(r.audio_latents), _video_span(r.latent_frames))
        else:
            raise ValueError(f"unknown reference kind {r.kind!r}")

    audio_start = cursor
    video_start = audio_start + num_target_audio_rows
    _fill_audio_positions(position_ids, slice(audio_start, video_start),
                          num_audio_latents, rotary_time, target_width_grid, audio_channels)
    frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = frame_time.repeat_interleave(target_frame_grid.shape[0])
    position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)

    video_indices = torch.cat(video_indices + [torch.arange(video_start, sequence_length)])
    audio_indices = torch.cat(audio_indices + [torch.arange(audio_start, video_start)])
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = audio_tag
    token_tags[video_indices] = video_tag

    return (position_ids, token_tags, video_indices, audio_indices, text_indices,
            num_reference_video_rows, num_reference_audio_rows)


# ---- row timesteps ---------------------------------------------------------
def build_row_timesteps(
    video_indices, audio_indices, num_condition_video_rows, num_condition_audio_rows,
    num_text_tokens, video_timestep, audio_timestep, condition_video_timestep,
    condition_audio_timestep,
):
    sequence_length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)
    row_timesteps = torch.full((sequence_length,), float(video_timestep), dtype=torch.float32)
    if num_condition_video_rows:
        row_timesteps[video_indices[:num_condition_video_rows]] = condition_video_timestep
    row_timesteps[audio_indices[num_condition_audio_rows:]] = audio_timestep
    if num_condition_audio_rows:
        row_timesteps[audio_indices[:num_condition_audio_rows]] = condition_audio_timestep
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


def row_timestep_plan(video_indices, audio_indices, num_condition_video_rows,
                      num_condition_audio_rows, num_text_tokens, timesteps, audio_timesteps):
    return [
        tuple(t.to(torch.float32) if False else t for t in build_row_timesteps(
            video_indices, audio_indices, num_condition_video_rows, num_condition_audio_rows,
            num_text_tokens, float(tv), float(ta), max(float(tv), KEYFRAME_NOISE_AUG), 1.0,
        ))
        for tv, ta in zip(timesteps, audio_timesteps)
    ]


# ---- rectified-flow scheduler (port of MiniMaxH3Scheduler) -----------------
class H3Scheduler:
    def __init__(self, shift: float):
        self.shift = float(shift)
        self.sigmas = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int, device="cpu"):
        base = torch.linspace(1.0, 0.0, int(num_inference_steps), dtype=torch.float32)
        sig = self.shift * base / (1 + (self.shift - 1) * base)
        self.sigmas = torch.unique_consecutive(sig).to(device)
        self.timesteps = (1.0 - self.sigmas[:-1]).to(device)

    @staticmethod
    def scale_noise(sample, timestep, noise):
        t = timestep
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=sample.dtype, device=sample.device)
        t = t.to(device=sample.device, dtype=sample.dtype)
        while t.ndim < sample.ndim:
            t = t.unsqueeze(-1)
        return t * sample + (1.0 - t) * noise

    def step(self, sample, model_output, timestep, index):
        sigma_from_t = 1 - (timestep if isinstance(timestep, torch.Tensor)
                            else torch.tensor(timestep, dtype=sample.dtype))
        sigma_from_t = sigma_from_t.to(device=sample.device, dtype=sample.dtype)
        while sigma_from_t.ndim < sample.ndim:
            sigma_from_t = sigma_from_t.unsqueeze(-1)
        denoised = sample + sigma_from_t * model_output
        compute_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
        sigma = self.sigmas[index].to(device=sample.device, dtype=compute_dtype)
        sigma_next = self.sigmas[index + 1].to(device=sample.device, dtype=compute_dtype)
        ratio = sigma_next / sigma
        return (ratio * sample.to(dtype=compute_dtype) + (1.0 - ratio) * denoised.to(dtype=compute_dtype)).to(sample.dtype)
