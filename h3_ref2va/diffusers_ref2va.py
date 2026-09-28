#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MiniMax-H3 `ref2va` with the official diffusers pipeline (fully local).

Reference images -> character-consistent video + stereo audio. Uses the
diffusers-format `transformer_ref/` partition with int8 weight-only + group
offload, so it fits one 48GB card with the weights in host RAM.

Usage:
  .venv/bin/python diffusers_ref2va.py --prompt "..." \
      --ref refs/sheet/cat_front.png [--ref ...] \
      --height 544 --width 960 --num-frames 124 --steps 30 --seed 42 \
      --output video/cat_ref2va.mp4
"""
from __future__ import annotations

import argparse
import os
import time

import torch

MODEL = os.path.expanduser("~/work/models/MiniMax-H3")


def build_pipe(workflow: str):
    from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
    from diffusers.hooks import apply_group_offloading
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig

    subfolder = "transformer_ref" if workflow == "ref2va" else "transformer"
    comp = "transformer_ref" if workflow == "ref2va" else "transformer"

    pipe = ModularPipeline.from_pretrained(MODEL)
    pipe.update_components(**{
        comp: MiniMaxH3Transformer3DModel.from_pretrained(
            MODEL, subfolder=subfolder, dtype=torch.bfloat16,
            quantization_config=TorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=["proj_in", "audio_proj_in", "context_embedder",
                                        "time_embedder", "time_proj", "token_refiner",
                                        "norm_out", "proj_out", "audio_proj_out"],
            ),
        ),
        "text_encoder": Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL, subfolder="text_encoder", dtype=torch.bfloat16,
            quantization_config=TransformersTorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=["model.visual", "model.language_model.embed_tokens",
                                        "model.language_model.norm", "lm_head"],
            ),
        ),
    })
    pipe.load_components(workflow=workflow, dtype=torch.bfloat16)

    transformer = getattr(pipe, comp)
    transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    offload = dict(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=True)
    transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
    apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)
    pipe.vae.to("cuda")
    pipe.audio_vae.to("cuda")
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--ref", action="append", required=True)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--num-frames", type=int, default=124)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="ref2va.mp4")
    ap.add_argument("--audio-ref", default=None,
                    help="optional voice-timbre reference audio (wav/mp4)")
    args = ap.parse_args()

    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
    )
    from diffusers.utils.export_utils import encode_video

    t0 = time.time()
    pipe = build_pipe("ref2va")
    print(f"[ref2va] pipeline ready {time.time() - t0:.0f}s", flush=True)

    references = [MiniMaxH3ImageReference.from_file(p) for p in args.ref]
    if args.audio_ref:
        references.append(MiniMaxH3AudioReference.from_file(args.audio_ref))
    print(f"[ref2va] {len(references)} references ({len(args.ref)} image"
          f"{' + 1 audio' if args.audio_ref else ''}), canvas {args.width}x{args.height}, "
          f"{args.num_frames} frames, {args.steps} steps", flush=True)

    t1 = time.time()
    results = pipe(
        prompt=args.prompt,
        references=references,
        height=args.height, width=args.width, num_frames=args.num_frames,
        num_inference_steps=args.steps,
        generator=torch.Generator().manual_seed(args.seed),
        output=["videos", "audio", "sampling_rate"],
    )
    print(f"[ref2va] generated in {time.time() - t1:.0f}s", flush=True)

    out = args.output if args.output.endswith(".mp4") else args.output + ".mp4"
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    encode_video(results["videos"][0], fps=24, output_path=out,
                 audio=results["audio"][0], audio_sample_rate=results["sampling_rate"])
    print(f"[ref2va] wrote {out} ({os.path.getsize(out) / 2**20:.1f} MiB), "
          f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
