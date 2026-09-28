#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the location reference cards (scene bible) for the pig series.

One card per location, photoreal, no characters — used as an extra FLUX.2
reference when generating each shot's keyframe so the same place looks the same
across shots (the "场景/道具一致" layer of shot continuity).

Usage:
  env/comfy/bin/python h3_ref2va/gen_scenes.py           # all
  env/comfy/bin/python h3_ref2va/gen_scenes.py 03_classroom
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import json
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from comfy_t2i import build  # noqa: E402

ROOT = os.path.dirname(HERE)
COMFY = os.path.join(ROOT, "ComfyUI")
OUT = os.path.join(ROOT, "refs", "scene")
HOST = "http://127.0.0.1:8188"

STYLE = ("photorealistic cinematic still, empty location with no characters, "
         "consistent art direction, warm family-friendly children's story, "
         "natural lighting, shallow depth of field, film grain, no text, no watermark")

SCENES = [
    ("01_pigsty_morning",
     "the interior of a cozy rustic pigsty hut in the early morning, straw on the wooden floor, "
     "a big wooden window with golden sunlight streaming in, buckets and a feeding trough, warm and tidy"),
    ("02_kitchen_table",
     "a warm farmhouse kitchen interior, a small wooden dining table with wooden chairs and a bench, "
     "a bowl and a plate on the table, a window with soft daylight, hanging pots, homely"),
    ("03_village_road",
     "a countryside dirt road between green fields and trees in the morning, low wooden fences, "
     "wildflowers along the verge, soft sunlight, blue sky"),
    ("04_classroom",
     "a rustic village primary school classroom, rows of small wooden desks and chairs, "
     "a green chalkboard at the front, a teacher's desk, sunlight through the windows"),
    ("05_playground",
     "a village school playground with a dirt field and green grass, a low wooden fence, "
     "a big tree, bright afternoon sun, blue sky with small clouds"),
    ("06_pond",
     "a small farm pond in the late afternoon, calm water with lily pads and reeds, a grassy bank, "
     "golden light reflecting on the water, nearby bushes"),
    ("07_pigsty_night",
     "the interior of a rustic pigsty hut at night, a pile of straw, warm dim lantern light, "
     "moonlight through the wooden window, starry night sky outside, cozy and peaceful"),
]


def submit(graph):
    data = json.dumps({"prompt": graph, "client_id": "scene"}).encode()
    req = urllib.request.Request(HOST + "/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:1200]); raise


def main():
    only = sys.argv[1:] or None
    os.makedirs(OUT, exist_ok=True)
    for i, (name, desc) in enumerate(SCENES):
        if only and name not in only:
            continue
        graph = build(f"{desc}, {STYLE}", 1024, 576, 20, 100 + i, f"scene_{name}")
        pid = submit(graph)
        t0 = time.time()
        while True:
            hist = json.load(urllib.request.urlopen(f"{HOST}/history/{pid}"))
            if pid in hist:
                break
            if time.time() - t0 > 600:
                raise SystemExit(f"timeout on {name}")
            time.sleep(3)
        st = hist[pid].get("status", {}).get("status_str")
        imgs = [f for n in hist[pid].get("outputs", {}).values() for f in n.get("images", [])]
        if st != "success" or not imgs:
            print(f"[{name}] FAILED {st}"); continue
        src = os.path.join(COMFY, "output", imgs[0].get("subfolder", ""), imgs[0]["filename"])
        shutil.copyfile(src, os.path.join(OUT, f"{name}.png"))
        print(f"[{name}] ok {time.time() - t0:.0f}s -> refs/scene/{name}.png", flush=True)


if __name__ == "__main__":
    main()
