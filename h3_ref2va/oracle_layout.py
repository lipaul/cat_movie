#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Golden layout tensors for MiniMax-H3 `ref2va`, straight from the diffusers
reference implementation. Runs no model: only the pure layout/position helpers,
so the native port in `ref2va_layout.py` can be checked bit-for-bit.

Usage:
    .venv/bin/python oracle_layout.py            # prints a summary
    .venv/bin/python oracle_layout.py --dump DIR # also writes .npz goldens
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3Ref2VAPrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    _frame_position_grid,
    _temporal_position_grid,
)
from diffusers.modular_pipelines.minimax_h3.references import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)

# released-checkpoint constants
PATCH = (1, 2, 2)
AUDIO_CHANNELS = 2
TEXT_TAG, VIDEO_TAG, AUDIO_TAG = 1, 0, 2
FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 17, 5


def video_latent_num_frames(num_frames: int) -> int:
    return (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int, fps: int = 24, lat_per_s: int = 40) -> int:
    return int(round(num_frames / fps * lat_per_s))


def _img_ref():
    # geometry comes from the encoded latent shape, not the image; a 1x1 dummy is fine
    return MiniMaxH3ImageReference(image=np.zeros((8, 8, 3), dtype=np.uint8))


def _vid_ref(with_audio: bool):
    ref = MiniMaxH3VideoReference(frames=np.zeros((4, 8, 8, 3), dtype=np.uint8), fps=24.0)
    if with_audio:
        ref.audio = torch.zeros(2, 16000)
        ref.sample_rate = 32000
    return ref


def _aud_ref():
    return MiniMaxH3AudioReference(audio=torch.zeros(2, 16000), sample_rate=32000)


CASES = {
    # name: (references, cond_latent_shapes, audio_cond_row_counts,
    #        num_latent_frames, latent_h, latent_w, num_audio_latents)
    "image_only": (
        [_img_ref()],
        [(1, 24, 1, 16, 16)],
        [],
        31, 48, 84, 104,
    ),
    "image_x2": (
        [_img_ref(), _img_ref()],
        [(1, 24, 1, 16, 16), (1, 24, 1, 20, 20)],
        [],
        31, 48, 84, 104,
    ),
    "image_audio": (
        [_img_ref(), _aud_ref()],
        [(1, 24, 1, 16, 16)],
        [64],
        31, 48, 84, 104,
    ),
    "image_video_audio": (
        [_img_ref(), _vid_ref(with_audio=True)],
        [(1, 24, 1, 16, 16), (1, 24, 7, 16, 16)],
        [64],
        31, 48, 84, 104,
    ),
    "audio_first_video": (
        [_vid_ref(with_audio=True), _img_ref()],
        [(1, 24, 7, 16, 16), (1, 24, 1, 16, 16)],
        [64],
        31, 48, 84, 104,
    ),
}


def build_golden(name):
    refs, shapes, audio_rows, nlf, lh, lw, nal = CASES[name]
    condition_latents = [torch.zeros(*s) for s in shapes]
    audio_condition_latents = [torch.zeros(r, 32) for r in audio_rows]
    # text tokens: a handful, tags all text_tag (vision blocks are part of the real
    # presentation but the layout only reads the tag vector length)
    text_token_tags = torch.ones(7, dtype=torch.long)

    (
        position_ids, token_tags, video_indices, audio_indices, text_indices,
        n_cond_v, n_cond_a,
    ) = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        text_token_tags, refs, condition_latents, audio_condition_latents,
        nlf, lh, lw, nal, PATCH, AUDIO_CHANNELS, AUDIO_TAG, VIDEO_TAG,
    )
    return {
        "position_ids": position_ids.numpy(),
        "token_tags": token_tags.numpy(),
        "video_indices": video_indices.numpy(),
        "audio_indices": audio_indices.numpy(),
        "text_indices": text_indices.numpy(),
        "num_condition_video_rows": n_cond_v,
        "num_condition_audio_rows": n_cond_a,
        "num_latent_frames": nlf,
        "latent_height": lh,
        "latent_width": lw,
        "num_audio_latents": nal,
        "num_text_tokens": 7,
    }


def golden_row_timesteps():
    """build_row_timesteps for a synthetic schedule."""
    from diffusers.schedulers import MiniMaxH3Scheduler

    sv = MiniMaxH3Scheduler(shift=12.0)
    sa = MiniMaxH3Scheduler(shift=3.0)
    sv.set_timesteps(8)
    sa.set_timesteps(8)
    g = build_golden("image_video_audio")
    video_indices = torch.from_numpy(g["video_indices"])
    audio_indices = torch.from_numpy(g["audio_indices"])
    out = []
    for t, at in zip(sv.timesteps, sa.timesteps):
        uq, idx = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video_indices, audio_indices,
            int(g["num_condition_video_rows"]), int(g["num_condition_audio_rows"]),
            int(g["num_text_tokens"]),
            float(t), float(at), max(float(t), 0.999), 1.0,
        )
        out.append({"unique": uq.numpy(), "indices": idx.numpy()})
    return {
        "sigmas": sv.sigmas.numpy(),
        "timesteps": sv.timesteps.numpy(),
        "audio_sigmas": sa.sigmas.numpy(),
        "audio_timesteps": sa.timesteps.numpy(),
        "rows": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None, help="directory to write .npz goldens")
    args = ap.parse_args()

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)

    for name in CASES:
        g = build_golden(name)
        S = g["position_ids"].shape[0]
        print(f"[{name}] S={S} text={g['num_text_tokens']} "
              f"cond_v={g['num_condition_video_rows']} cond_a={g['num_condition_audio_rows']} "
              f"vid_rows={g['video_indices'].shape[0]} aud_rows={g['audio_indices'].shape[0]}")
        if args.dump:
            np.savez(os.path.join(args.dump, f"layout_{name}.npz"), **g)

    rt = golden_row_timesteps()
    print(f"[row_timesteps] steps={len(rt['timesteps'])} sigmas={rt['sigmas']}")
    if args.dump:
        np.savez(os.path.join(args.dump, "row_timesteps.npz"),
                 **{k: v for k, v in rt.items() if k != "rows"},
                 rows=np.array(rt["rows"], dtype=object), allow_pickle=True)
        with open(os.path.join(args.dump, "row_timesteps.json"), "w") as f:
            json.dump([{"unique": r["unique"].tolist(), "indices": r["indices"].tolist()}
                       for r in rt["rows"]], f, indent=1)
    print("ok")


if __name__ == "__main__":
    main()
