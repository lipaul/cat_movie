#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Animate the 15 keyframes of《嘟嘟的一天》with LTX-2.5 (i2v, ~20s each).

Each shot carries its own dialogue / narration in the prompt, so LTX-2.5
generates the Chinese speech and lip movement together with the video.
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
from comfy_ltx_i2v import build  # noqa: E402

ROOT = os.path.dirname(HERE)
COMFY = os.path.join(ROOT, "ComfyUI")
HOST = "http://127.0.0.1:8188"
STORY = os.path.join(ROOT, "refs", "story")
OUT = os.path.join(ROOT, "video", "shots")

STYLE = ("photorealistic, natural animal motion, cinematic, warm family-friendly children's story, "
         "no text, no subtitles, no watermark")

# (keyframe, dialogue/narration spoken in the shot, motion note)
SHOTS = [
    ("01_wakeup",
     "A gentle warm Chinese narrator says slowly: 清晨，猪圈里的小猪嘟嘟还在做着甜甜的梦。",
     "the piglet sleeps on its back, snout twitching, breathing softly, warm sunlight moving slowly across the straw"),
    ("02_wakeup_talk",
     "The mother pig speaks in a warm adult female Chinese voice: 嘟嘟！太阳晒屁股啦！再不起来，早饭就变成晚饭了！ "
     "Then the small piglet Dudu answers in a sleepy cute child voice: 呜……还想再睡五分钟…… 好啦好啦……我起来了……",
     "the mother pig nuzzles the piglet awake, the piglet slowly rolls over, stretches and stands up unsteadily"),
    ("03_breakfast",
     "The mother pig says warmly in Chinese: 今天有热乎乎的玉米粥和苹果片！ "
     "The small piglet Dudu answers excitedly in a cute child voice: 香香香！",
     "the piglet holds the bowl with its front hooves, sniffs eagerly and eats, steam rising from the porridge"),
    ("04_school_road",
     "The small white lamb says cheerfully in a child Chinese voice: 嘟嘟早！你鼻子上还沾着粥呢！ "
     "The piglet Dudu answers in a cute child voice: 嘿嘿……好吃的留下的痕迹！走吧走吧！",
     "the piglet trots on four legs along the road with its backpack bouncing, the lamb walks beside it, morning light"),
    ("05_classroom",
     "The white goose teacher speaks in a clear adult Chinese voice: 今天我们学习分享……嘟嘟，你为什么一直在拱桌脚？ "
     "The piglet Dudu answers in a cute child voice: 老师，我在思考！思考怎么把好吃的分享给大家！",
     "the piglet pushes the desk leg with its snout, the other animals turn to look, the goose teacher stands at the board"),
    ("06_naptime",
     "The piglet Dudu whispers in a sleepy cute Chinese voice: 汪汪……我好想睡觉…… "
     "Then it snores softly, and the brown puppy sighs in a child voice: 又开始做泥巴梦了……",
     "the animals doze on the desks, the piglet's hooves twitch as it dreams, gentle afternoon light"),
    ("07_playground",
     "The goose teacher announces in a clear adult Chinese voice: 今天自由活动！可以跑步、跳跃、或者…… "
     "The piglet Dudu shouts excitedly in a cute child voice: 我要比赛谁拱得最快！",
     "the piglet jumps up excitedly on the grass, the other animals gather around, bright sunshine"),
    ("08_dig",
     "The small white lamb laughs in a child Chinese voice: 嘟嘟你又把操场拱成迷宫了！",
     "the piglet pushes its snout through the dirt digging a trench, dust flying, the other animals laugh and hop around"),
    ("09_pond",
     "The yellow duckling calls in a bright child Chinese voice: 来游泳呀！ "
     "The piglet Dudu answers hesitantly in a cute child voice: 凉凉的……",
     "the piglet touches the water with one front hoof, ripples spreading, the duckling paddles nearby"),
    ("10_splash",
     "The piglet Dudu shouts joyfully in a cute child Chinese voice: 噗噗噗！我是水里的小火箭！",
     "the piglet leaps into the pond with a big splash, paddling with all four legs, water spraying everywhere"),
    ("11_swim",
     "A gentle warm Chinese narrator says: 大家在池塘里玩得可开心啦。",
     "the animals splash and paddle together, the piglet bobs up and down, water sparkling in the sun"),
    ("12_dinner",
     "The mother pig asks warmly in Chinese: 今天玩得开心吗？ "
     "The piglet Dudu answers excitedly in a cute child voice: 开心！我拱了操场、游了泳、还差点把桌子拱走！",
     "the wet piglet stands by the table talking excitedly, the mother pig looks down at him, warm lamp light"),
    ("13_eat",
     "The mother pig laughs warmly in Chinese: 那明天继续努力……先把晚饭吃完！ "
     "The piglet Dudu answers with its mouth full in a cute child voice: 嗯嗯……今天的南瓜最好吃！",
     "the piglet buries its snout in the bowl of pumpkin and eats eagerly, food on its face"),
    ("14_bedtime",
     "The piglet Dudu yawns and asks in a sleepy cute Chinese voice: 妈妈……明天还要上学吗？ "
     "The mother pig answers gently in Chinese: 要的。",
     "the piglet climbs into the straw and yawns, the mother pig settles beside him, dim warm lantern light"),
    ("15_sleep",
     "The piglet Dudu murmurs sleepily in a fading cute Chinese voice: 那我……明天还要……拱…… "
     "Then a gentle Chinese narrator says softly: 就这样，嘟嘟带着甜甜的梦，睡着了。",
     "the piglet falls asleep in the straw, breathing slowly, the lantern light fades, moonlight through the window"),
]


def submit(graph):
    data = json.dumps({"prompt": graph, "client_id": "shot"}).encode()
    req = urllib.request.Request(HOST + "/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:1500]); raise


def main():
    only = sys.argv[1:] or None
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(COMFY, "input"), exist_ok=True)
    for i, (name, dialogue, motion) in enumerate(SHOTS):
        if only and name not in only:
            continue
        img = f"{name}.png"
        shutil.copyfile(os.path.join(STORY, img), os.path.join(COMFY, "input", img))
        prompt = f"{motion}. {dialogue} {STYLE}"
        graph = build(img, prompt, 960, 544, 481, 42 + i, f"shot_{name}", 1.0, 24)
        pid = submit(graph)
        t0 = time.time()
        while True:
            hist = json.load(urllib.request.urlopen(f"{HOST}/history/{pid}"))
            if pid in hist:
                break
            if time.time() - t0 > 1800:
                raise SystemExit(f"timeout on {name}")
            time.sleep(5)
        st = hist[pid].get("status", {}).get("status_str")
        outs = [f for n in hist[pid].get("outputs", {}).values()
                for k in ("images", "videos", "gifs") for f in n.get(k, [])]
        if st != "success" or not outs:
            print(f"[{name}] FAILED {st}"); continue
        src = os.path.join(COMFY, "output", outs[0].get("subfolder", ""), outs[0]["filename"])
        dst = os.path.join(OUT, f"{name}.mp4")
        shutil.copyfile(src, dst)
        print(f"[{name}] ok {time.time() - t0:.0f}s -> video/shots/{name}.mp4", flush=True)


if __name__ == "__main__":
    main()
