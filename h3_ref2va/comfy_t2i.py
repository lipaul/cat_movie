#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FLUX.2-dev text-to-image via the running ComfyUI server (API format).

Usage:
  env/comfy/bin/python h3_ref2va/comfy_t2i.py --prompt "..." \
      --width 1024 --height 1024 --steps 20 --seed 42 --prefix cat_hero
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.request

HOST = "http://127.0.0.1:8188"
COMFY = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/ComfyUI"


def build(prompt, width, height, steps, seed, prefix, refs=None):
    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "flux2_dev_fp8mixed.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "mistral_3_small_flux2_fp8.safetensors", "type": "flux2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": 4.0}},
        "6": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["5", 0]}},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "9": {"class_type": "Flux2Scheduler",
              "inputs": {"steps": steps, "width": width, "height": height}},
        "10": {"class_type": "EmptyFlux2LatentImage",
               "inputs": {"width": width, "height": height, "batch_size": 1}},
        "11": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["7", 0], "guider": ["6", 0], "sampler": ["8", 0],
                          "sigmas": ["9", 0], "latent_image": ["10", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}},
    }
    if refs:
        # FLUX.2 multi-reference: each ref -> VAEEncode -> ReferenceLatent(conditioning) chained
        cond = ["5", 0]
        for i, r in enumerate(refs):
            li, ve, rl = f"1{i:02d}", f"2{i:02d}", f"3{i:02d}"
            g[li] = {"class_type": "LoadImage", "inputs": {"image": r}}
            g[ve] = {"class_type": "VAEEncode", "inputs": {"pixels": [li, 0], "vae": ["3", 0]}}
            g[rl] = {"class_type": "ReferenceLatent",
                     "inputs": {"conditioning": cond, "latent": [ve, 0]}}
            cond = [rl, 0]
        g["6"]["inputs"]["conditioning"] = cond
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="cat")
    ap.add_argument("--ref", action="append", default=None,
                    help="reference image path (repeatable; copied into ComfyUI/input)")
    ap.add_argument("--out", default=None, help="copy the first output image here")
    args = ap.parse_args()

    ref_names = None
    if args.ref:
        os.makedirs(os.path.join(COMFY, "input"), exist_ok=True)
        ref_names = []
        for p in args.ref:
            n = os.path.basename(p)
            shutil.copyfile(p, os.path.join(COMFY, "input", n))
            ref_names.append(n)

    graph = build(args.prompt, args.width, args.height, args.steps, args.seed,
                  args.prefix, ref_names)
    data = json.dumps({"prompt": graph, "client_id": "cat"}).encode()
    req = urllib.request.Request(HOST + "/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req))
    pid = resp["prompt_id"]
    print(f"queued {pid}", flush=True)

    t0 = time.time()
    while True:
        hist = json.load(urllib.request.urlopen(f"{HOST}/history/{pid}"))
        if pid in hist:
            break
        if time.time() - t0 > 3600:
            raise SystemExit("timeout")
        time.sleep(3)
    entry = hist[pid]
    status = entry.get("status", {})
    print(f"status: {status.get('status_str')} in {time.time() - t0:.0f}s", flush=True)
    if status.get("status_str") == "error":
        for m in status.get("messages", []):
            print("  ", m)
        raise SystemExit(1)

    images = []
    for node in entry.get("outputs", {}).values():
        images += node.get("images", [])
    for im in images:
        src = os.path.join(COMFY, "output", im.get("subfolder", ""), im["filename"])
        print("output:", src)
        if args.out and not os.path.exists(args.out):
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            shutil.copyfile(src, args.out)
            print("copied to", args.out)


if __name__ == "__main__":
    main()
