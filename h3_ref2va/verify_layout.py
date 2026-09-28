#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bit-exact check of the native `ref2va_layout` against the diffusers oracle.

Compares the packed layout, its fp64 position grid, the row indices, the row
timestep plan and the scheduler schedule for every synthetic case.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, ".")
import ref2va_layout as native  # noqa: E402
import oracle_layout as oracle  # noqa: E402

PATCH = (1, 2, 2)


def _refs_for(name):
    """Rebuild native Ref descriptors from the oracle case definition."""
    refs, shapes, audio_rows, *_ = oracle.CASES[name]
    native_refs, si, ai = [], 0, 0
    for r in refs:
        if r.kind == "image":
            s = shapes[si]; si += 1
            native_refs.append(native.Ref.image(s[2], s[3], s[4]))
        elif r.kind == "video":
            s = shapes[si]; si += 1
            nal = audio_rows[ai] // native.AUDIO_CHANNELS if r.has_audio else 0
            if r.has_audio:
                ai += 1
            native_refs.append(native.Ref.video(s[2], s[3], s[4], nal))
        else:  # audio
            nal = audio_rows[ai] // native.AUDIO_CHANNELS; ai += 1
            native_refs.append(native.Ref.audio(nal))
    return native_refs


def check_case(name):
    refs, shapes, audio_rows, nlf, lh, lw, nal = oracle.CASES[name]
    g = oracle.build_golden(name)
    native_refs = _refs_for(name)

    (pos, tags, vi, aii, ti, ncv, nca) = native.build_ref2va_packed_sequence(
        torch.ones(g["num_text_tokens"], dtype=torch.long),
        native_refs, nlf, lh, lw, nal, PATCH,
        native.AUDIO_CHANNELS, native.AUDIO_TAG, native.VIDEO_TAG,
    )
    errs = []
    if pos.shape != g["position_ids"].shape:
        errs.append(f"position_ids shape {tuple(pos.shape)} vs {g['position_ids'].shape}")
    else:
        d = (pos.numpy() - g["position_ids"])
        if d.any():
            errs.append(f"position_ids max|d|={np.abs(d).max():.3e} at {np.unravel_index(np.abs(d).argmax(), d.shape)}")
    for label, a, b in (
        ("token_tags", tags.numpy(), g["token_tags"]),
        ("video_indices", vi.numpy(), g["video_indices"]),
        ("audio_indices", aii.numpy(), g["audio_indices"]),
        ("text_indices", ti.numpy(), g["text_indices"]),
    ):
        if not np.array_equal(a, b):
            errs.append(f"{label} mismatch")
    for label, a, b in (
        ("num_condition_video_rows", ncv, g["num_condition_video_rows"]),
        ("num_condition_audio_rows", nca, g["num_condition_audio_rows"]),
    ):
        if int(a) != int(b):
            errs.append(f"{label} {a} vs {b}")
    return errs


def check_schedule():
    errs = []
    g = oracle.golden_row_timesteps()
    sv = native.H3Scheduler(native.FLOW_SHIFT_VIDEO)
    sa = native.H3Scheduler(native.FLOW_SHIFT_AUDIO)
    sv.set_timesteps(8); sa.set_timesteps(8)
    if not np.allclose(sv.sigmas.numpy(), g["sigmas"], rtol=0, atol=0):
        errs.append("video sigmas")
    if not np.allclose(sv.timesteps.numpy(), g["timesteps"], rtol=0, atol=0):
        errs.append("video timesteps")
    if not np.allclose(sa.sigmas.numpy(), g["audio_sigmas"], rtol=0, atol=0):
        errs.append("audio sigmas")

    gg = oracle.build_golden("image_video_audio")
    video_indices = torch.from_numpy(gg["video_indices"])
    audio_indices = torch.from_numpy(gg["audio_indices"])
    for i, gold in enumerate(g["rows"]):
        uq, idx = native.build_row_timesteps(
            video_indices, audio_indices, int(gg["num_condition_video_rows"]),
            int(gg["num_condition_audio_rows"]), int(gg["num_text_tokens"]),
            float(sv.timesteps[i]), float(sa.timesteps[i]),
            max(float(sv.timesteps[i]), native.KEYFRAME_NOISE_AUG), 1.0,
        )
        if not np.array_equal(uq.numpy(), gold["unique"]):
            errs.append(f"row_timesteps[{i}] unique")
        if not np.array_equal(idx.numpy(), gold["indices"]):
            errs.append(f"row_timesteps[{i}] indices")
    return errs


def main():
    ok = True
    for name in oracle.CASES:
        errs = check_case(name)
        print(f"[{'PASS' if not errs else 'FAIL'}] {name}" + ("" if not errs else f"  -> {errs}"))
        ok &= not errs
    errs = check_schedule()
    print(f"[{'PASS' if not errs else 'FAIL'}] schedule+row_timesteps" + ("" if not errs else f"  -> {errs}"))
    ok &= not errs
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
