#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LTX-2.5 image-to-video via the running ComfyUI server (API format).

Single-stage, Ada-friendly adaptation of ComfyUI's official
`video_ltx2_5_i2v` template:
  * bf16 transformer + `weight_dtype=fp8_e4m3fn` (no ltx_kernels / nvfp4)
  * `ltx-2.5-video-vae-conv-bf16` (the non-conv DiffVAE needs natten)
  * no `TextGenerateLTX2Prompt` (the small prompt-refiner Gemma is not local)
  * no latent upscale pass (single sampling stage)

Usage:
  env/comfy/bin/python h3_ref2va/comfy_ltx_i2v.py --image refs/pig/P1b_fourleg.png \
      --prompt "..." --width 960 --height 544 --frames 97 --seed 42 --out video/ltx_spike.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.request

HOST = "http://127.0.0.1:8188"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.join(ROOT, "ComfyUI")

NEG = "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, distorted anatomy, extra limbs, text, watermark"
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"


def build(image, prompt, width, height, frames, seed, prefix, strength, fps):
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
                         "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", "type": "ltxv"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.5-video-vae-conv-bf16.safetensors"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.5-audio-vae-bf16.safetensors"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["2", 0]}},
        "7": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": float(fps)}},
        "8": {"class_type": "LoadImage", "inputs": {"image": image}},
        "9": {"class_type": "LTXVPreprocess", "inputs": {"image": ["8", 0], "img_compression": 18}},
        "10": {"class_type": "EmptyLTXVLatentVideo",
               "inputs": {"width": width, "height": height, "length": frames, "batch_size": 1}},
        "11": {"class_type": "LTXVImgToVideoInplace",
               "inputs": {"vae": ["3", 0], "image": ["9", 0], "latent": ["10", 0],
                          "strength": strength, "bypass": False}},
        "12": {"class_type": "LTXVEmptyLatentAudio",
               "inputs": {"frames_number": frames, "frame_rate": float(fps),
                          "batch_size": 1, "audio_vae": ["4", 0]}},
        "13": {"class_type": "LTXVConcatAVLatent",
               "inputs": {"video_latent": ["11", 0], "audio_latent": ["12", 0]}},
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "15": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "16": {"class_type": "ManualSigmas", "inputs": {"sigmas": SIGMAS}},
        "17": {"class_type": "LTXVDualCFGGuider",
               "inputs": {"model": ["1", 0], "positive": ["7", 0], "negative": ["7", 1],
                          "video_cfg": 1.0, "audio_cfg": 1.0}},
        "18": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["14", 0], "guider": ["17", 0], "sampler": ["15", 0],
                          "sigmas": ["16", 0], "latent_image": ["13", 0]}},
        "19": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["18", 0]}},
        "20": {"class_type": "VAEDecodeTiled",
               "inputs": {"samples": ["19", 0], "vae": ["3", 0],
                          "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 16}},
        "21": {"class_type": "LTXVAudioVAEDecode",
               "inputs": {"samples": ["19", 1], "audio_vae": ["4", 0]}},
        "22": {"class_type": "CreateVideo", "inputs": {"images": ["20", 0], "fps": float(fps), "audio": ["21", 0]}},
        "23": {"class_type": "SaveVideo",
               "inputs": {"video": ["22", 0], "filename_prefix": prefix, "format": "auto"}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--frames", type=int, default=97)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--prefix", default="ltx_spike")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    name = os.path.basename(args.image)
    os.makedirs(os.path.join(COMFY, "input"), exist_ok=True)
    shutil.copyfile(args.image, os.path.join(COMFY, "input", name))

    graph = build(name, args.prompt, args.width, args.height, args.frames,
                  args.seed, args.prefix, args.strength, args.fps)
    data = json.dumps({"prompt": graph, "client_id": "ltx"}).encode()
    req = urllib.request.Request(HOST + "/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        resp = json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:2000])
        raise SystemExit(1)
    pid = resp["prompt_id"]
    print(f"queued {pid}", flush=True)

    t0 = time.time()
    while True:
        hist = json.load(urllib.request.urlopen(f"{HOST}/history/{pid}"))
        if pid in hist:
            break
        if time.time() - t0 > 7200:
            raise SystemExit("timeout")
        time.sleep(3)
    entry = hist[pid]
    st = entry.get("status", {})
    print(f"status: {st.get('status_str')} in {time.time() - t0:.0f}s", flush=True)
    if st.get("status_str") == "error":
        for m in st.get("messages", []):
            print("  ", str(m)[:800])
        raise SystemExit(1)
    for node in entry.get("outputs", {}).values():
        for key in ("images", "videos", "gifs"):
            for f in node.get(key, []):
                src = os.path.join(COMFY, "output", f.get("subfolder", ""), f["filename"])
                print("output:", src)
                if args.out:
                    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
                    shutil.copyfile(src, args.out)
                    print("copied to", args.out)


if __name__ == "__main__":
    main()
