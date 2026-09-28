#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LTX-2.5 first-and-last-frame to video — the continuity tool.

For a shot that must *connect* to the next one, generate it between two known
frames instead of letting the model invent both ends: shot N's last frame
becomes shot N+1's first frame, so the cut has no visual jump. Mirrors ComfyUI's
official `video_ltx2_5_flf2v` template: two `LTXVAddGuide` (frame_idx 0 and -1),
sampling on the guided latent, then `LTXVCropGuides` on the sampler's
`denoised_output` before decode.

Usage:
  env/comfy/bin/python h3_ref2va/comfy_ltx_flf2v.py \
      --first refs/story/a.png --last refs/story/b.png \
      --prompt "..." --frames 481 --seed 42 --out video/seg.mp4
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

NEG = ("blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
       "excessive noise, grainy texture, distorted anatomy, extra limbs, "
       "subtitles, captions, on-screen text, text overlay, chinese characters, letters, words, "
       "writing, typography, watermark, logo, signature")
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"


def build(first, last, prompt, width, height, frames, seed, prefix, strength, fps, neg=None):
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
                         "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", "type": "ltxv"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.5-video-vae-conv-bf16.safetensors"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.5-audio-vae-bf16.safetensors"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": neg or NEG, "clip": ["2", 0]}},
        "7": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": float(fps)}},
        "8": {"class_type": "LoadImage", "inputs": {"image": first}},
        "9": {"class_type": "LoadImage", "inputs": {"image": last}},
        "10": {"class_type": "LTXVPreprocess", "inputs": {"image": ["8", 0], "img_compression": 18}},
        "11": {"class_type": "LTXVPreprocess", "inputs": {"image": ["9", 0], "img_compression": 18}},
        "12": {"class_type": "EmptyLTXVLatentVideo",
               "inputs": {"width": width, "height": height, "length": frames, "batch_size": 1}},
        "13": {"class_type": "LTXVAddGuide",
               "inputs": {"positive": ["7", 0], "negative": ["7", 1], "vae": ["3", 0],
                          "latent": ["12", 0], "image": ["10", 0],
                          "frame_idx": 0, "strength": strength}},
        "14": {"class_type": "LTXVAddGuide",
               "inputs": {"positive": ["13", 0], "negative": ["13", 1], "vae": ["3", 0],
                          "latent": ["13", 2], "image": ["11", 0],
                          "frame_idx": -1, "strength": strength}},
        "15": {"class_type": "LTXVEmptyLatentAudio",
               "inputs": {"frames_number": frames, "frame_rate": float(fps),
                          "batch_size": 1, "audio_vae": ["4", 0]}},
        "16": {"class_type": "LTXVConcatAVLatent",
               "inputs": {"video_latent": ["14", 2], "audio_latent": ["15", 0]}},
        "17": {"class_type": "LTXVDualCFGGuider",
               "inputs": {"model": ["1", 0], "positive": ["14", 0], "negative": ["14", 1],
                          "video_cfg": 1.0, "audio_cfg": 1.0}},
        "18": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "19": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "20": {"class_type": "ManualSigmas", "inputs": {"sigmas": SIGMAS}},
        "21": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["18", 0], "guider": ["17", 0], "sampler": ["19", 0],
                          "sigmas": ["20", 0], "latent_image": ["16", 0]}},
        # denoised_output (index 1) carries the guide frames, which CropGuides trims
        "22": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["21", 1]}},
        "23": {"class_type": "LTXVCropGuides",
               "inputs": {"positive": ["14", 0], "negative": ["14", 1], "latent": ["22", 0]}},
        "24": {"class_type": "VAEDecodeTiled",
               "inputs": {"samples": ["23", 2], "vae": ["3", 0],
                          "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 16}},
        "25": {"class_type": "LTXVAudioVAEDecode",
               "inputs": {"samples": ["22", 1], "audio_vae": ["4", 0]}},
        "26": {"class_type": "CreateVideo",
               "inputs": {"images": ["24", 0], "fps": float(fps), "audio": ["25", 0]}},
        "27": {"class_type": "SaveVideo",
               "inputs": {"video": ["26", 0], "filename_prefix": prefix, "format": "auto"}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", required=True)
    ap.add_argument("--last", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--frames", type=int, default=481)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strength", type=float, default=0.7)
    ap.add_argument("--prefix", default="ltx_flf")
    ap.add_argument("--neg", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.makedirs(os.path.join(COMFY, "input"), exist_ok=True)
    names = []
    for tag, p in (("f", a.first), ("l", a.last)):
        n = f"flf_{tag}_{os.path.basename(p)}"
        shutil.copyfile(p, os.path.join(COMFY, "input", n))
        names.append(n)

    graph = build(names[0], names[1], a.prompt, a.width, a.height, a.frames,
                  a.seed, a.prefix, a.strength, a.fps, a.neg)
    data = json.dumps({"prompt": graph, "client_id": "flf"}).encode()
    req = urllib.request.Request(HOST + "/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:1500]); raise SystemExit(1)
    print(f"queued {pid}", flush=True)

    t0 = time.time()
    while True:
        hist = json.load(urllib.request.urlopen(f"{HOST}/history/{pid}"))
        if pid in hist:
            break
        if time.time() - t0 > 3600:
            raise SystemExit("timeout")
        time.sleep(3)
    st = hist[pid].get("status", {})
    print(f"status: {st.get('status_str')} in {time.time() - t0:.0f}s", flush=True)
    if st.get("status_str") == "error":
        for m in st.get("messages", []):
            print("  ", str(m)[:800])
        raise SystemExit(1)
    for node in hist[pid].get("outputs", {}).values():
        for k in ("images", "videos", "gifs"):
            for f in node.get(k, []):
                src = os.path.join(COMFY, "output", f.get("subfolder", ""), f["filename"])
                print("output:", src)
                if a.out:
                    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
                    shutil.copyfile(src, a.out)
                    print("copied to", a.out)


if __name__ == "__main__":
    main()
