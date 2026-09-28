#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the 15 keyframes of《嘟嘟的一天》with FLUX.2 + multi-character refs.

Each shot is one 16:9 keyframe that LTX-2.5 then animates to ~20s.
Run against the running ComfyUI server.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from comfy_t2i import build  # noqa: E402

ROOT = os.path.dirname(HERE)
COMFY = os.path.join(ROOT, "ComfyUI")
HOST = "http://127.0.0.1:8188"
CH = os.path.join(ROOT, "refs", "char")
OUT = os.path.join(ROOT, "refs", "story")

STYLE = ("photorealistic cinematic still, real animals with natural anatomy, "
         "warm family-friendly children's story, soft natural lighting, shallow depth of field, "
         "highly detailed, film grain, no text, no watermark, no subtitles")

SHOTS = [
    ("01_wakeup", ["dudu", "mama"],
     "inside a cozy pigsty hut at early morning, a small pink piglet named Dudu lying on its back on "
     "straw with all four hooves up, still asleep, its snout twitching, warm golden sunlight streaming "
     "through the wooden window, the mother pig sitting nearby looking at him fondly"),
    ("02_wakeup_talk", ["dudu", "mama"],
     "inside the pigsty, the mother pig gently nuzzles the small pink piglet Dudu awake with her snout, "
     "Dudu just waking up, sleepy and dazed, straw around them, warm morning light"),
    ("03_breakfast", ["dudu", "mama"],
     "a small wooden dining table inside a farmhouse kitchen in the morning, the small pink piglet Dudu "
     "sitting on a little stool holding a wooden bowl with its front hooves, the mother pig placing a bowl "
     "of hot corn porridge and slices of apple on the table, steam rising, warm cozy light"),
    ("04_school_road", ["dudu", "lamb"],
     "a countryside dirt road in the morning, the small pink piglet Dudu trotting on four legs wearing a "
     "tiny school backpack, meeting a small white lamb on the road, green fields and trees, soft sunlight"),
    ("05_classroom", ["goose", "dudu", "lamb", "puppy", "duck"],
     "a rustic village primary school classroom, a white goose teacher standing at the blackboard, the "
     "small pink piglet Dudu at a wooden desk pushing the desk leg with its snout, a white lamb, a brown "
     "puppy and a yellow duckling sitting at their desks, warm daylight through windows"),
    ("06_naptime", ["dudu", "puppy", "lamb"],
     "inside the classroom at naptime, the animals resting with their heads on the wooden desks, the small "
     "pink piglet Dudu asleep on its back with hooves twitching, the brown puppy beside him looking sleepy, "
     "soft afternoon light"),
    ("07_playground", ["goose", "dudu", "lamb", "puppy", "duck"],
     "a village school playground with grass and a dirt field, the white goose teacher announcing free "
     "playtime, the small pink piglet Dudu looking up excitedly, a white lamb, a brown puppy and a yellow "
     "duckling gathered around, bright afternoon sun"),
    ("08_dig", ["dudu", "lamb", "puppy", "duck"],
     "on the school playground, the small pink piglet Dudu with its snout pushed into the dirt digging a "
     "little trench, a white lamb, a brown puppy and a yellow duckling watching and laughing around him, "
     "dust and dirt flying, sunny afternoon"),
    ("09_pond", ["duck", "dudu"],
     "at the edge of a small farm pond in the late afternoon, a yellow duckling calling out, the small pink "
     "piglet Dudu testing the water with one front hoof, reeds and water lilies, golden light on the water"),
    ("10_splash", ["dudu", "duck"],
     "the small pink piglet Dudu leaping into the pond with a big splash, all four legs paddling, water "
     "spraying from its snout, the yellow duckling beside it, sparkling water droplets in the sunlight"),
    ("11_swim", ["dudu", "duck", "lamb", "puppy"],
     "several animals playing together in the farm pond, the small pink piglet Dudu bobbing up and down "
     "with only its snout and ears above the water, a yellow duckling, a white lamb and a brown puppy "
     "splashing around, warm afternoon light"),
    ("12_dinner", ["dudu", "mama"],
     "back home in the farmhouse kitchen at dusk, the small pink piglet Dudu standing on the floor with wet "
     "muddy hooves talking excitedly to the mother pig, a wooden table with dinner set on it, warm lamp light"),
    ("13_eat", ["dudu"],
     "the small pink piglet Dudu eating a big bowl of pumpkin with its snout buried in the food, food on its "
     "face, sitting by the wooden table in a warm farmhouse kitchen, cozy lamp light"),
    ("14_bedtime", ["dudu", "mama"],
     "inside the pigsty at night, the small pink piglet Dudu climbing into a pile of straw and yawning, the "
     "mother pig beside him, warm dim lantern light, starry night visible through the window"),
    ("15_sleep", ["dudu"],
     "inside the pigsty at night, the small pink piglet Dudu curled up fast asleep in the straw, its snout "
     "resting, warm dim lantern light fading, peaceful and cozy, moonlight through the window"),
]


def submit(graph):
    data = json.dumps({"prompt": graph, "client_id": "kf"}).encode()
    req = urllib.request.Request(HOST + "/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:1500])
        raise


def main():
    only = sys.argv[1:] or None
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(COMFY, "input"), exist_ok=True)
    for name, chars, desc in SHOTS:
        if only and name not in only:
            continue
        refs = []
        for c in chars:
            src = os.path.join(CH, f"{c}.png")
            shutil.copyfile(src, os.path.join(COMFY, "input", f"char_{c}.png"))
            refs.append(f"char_{c}.png")
        prompt = f"{desc}, {STYLE}"
        graph = build(prompt, 1024, 576, 20, 5, f"kf_{name}", refs)
        pid = submit(graph)
        t0 = time.time()
        while True:
            hist = json.load(urllib.request.urlopen(f"{HOST}/history/{pid}"))
            if pid in hist:
                break
            if time.time() - t0 > 900:
                raise SystemExit(f"timeout on {name}")
            time.sleep(3)
        st = hist[pid].get("status", {}).get("status_str")
        imgs = [f for n in hist[pid].get("outputs", {}).values() for f in n.get("images", [])]
        if st != "success" or not imgs:
            print(f"[{name}] FAILED {st}"); continue
        src = os.path.join(COMFY, "output", imgs[0].get("subfolder", ""), imgs[0]["filename"])
        shutil.copyfile(src, os.path.join(OUT, f"{name}.png"))
        print(f"[{name}] ok {time.time() - t0:.0f}s -> refs/story/{name}.png", flush=True)


if __name__ == "__main__":
    main()
