#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate the official diffusers MiniMax-H3 `t2va` pipeline on this machine.

Uses the int8 + group-offload recipe from the diffusers docs. Confirms the
runtime produces a real video from the diffusers-format `transformer/`.
"""
from __future__ import annotations

import os
import sys
import time

import torch

MODEL = os.path.expanduser("~/work/models/MiniMax-H3")


def main():
    from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
    from diffusers.hooks import apply_group_offloading
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig

    t0 = time.time()
    pipe = ModularPipeline.from_pretrained(MODEL)
    print(f"pipeline loaded {time.time() - t0:.0f}s")

    pipe.update_components(
        transformer=MiniMaxH3Transformer3DModel.from_pretrained(
            MODEL, subfolder="transformer", dtype=torch.bfloat16,
            quantization_config=TorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=["proj_in", "audio_proj_in", "context_embedder",
                                        "time_embedder", "time_proj", "token_refiner",
                                        "norm_out", "proj_out", "audio_proj_out"],
            ),
        ),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL, subfolder="text_encoder", dtype=torch.bfloat16,
            quantization_config=TransformersTorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=["model.visual", "model.language_model.embed_tokens",
                                        "model.language_model.norm", "lm_head"],
            ),
        ),
    )
    pipe.load_components(workflow="t2va", dtype=torch.bfloat16)
    print(f"components loaded {time.time() - t0:.0f}s")

    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    offload = dict(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=True)
    pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
    apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)
    pipe.vae.to("cuda")
    pipe.audio_vae.to("cuda")

    print("generating...")
    t1 = time.time()
    results = pipe(
        prompt="An anthropomorphic orange tabby cat turns its head to the camera and smiles, cinematic lighting.",
        height=544, width=960, num_frames=124,
        num_inference_steps=20,
        generator=torch.Generator().manual_seed(42),
        output=["videos", "audio", "sampling_rate"],
    )
    print(f"generated in {time.time() - t1:.0f}s")

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video")
    os.makedirs(outdir, exist_ok=True)
    from diffusers.utils.export_utils import encode_video

    out = os.path.abspath(os.path.join(outdir, "diffusers_t2va_544.mp4"))
    encode_video(results["videos"][0], fps=24, output_path=out,
                 audio=results["audio"][0], audio_sample_rate=results["sampling_rate"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
