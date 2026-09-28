#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Post-production for《嘟嘟的一天》: concat 15 LTX shots, add one continuous
music-box BGM bed, an opening title card and bilingual subtitles.

Output: video/dudu_full.mp4
"""
from __future__ import annotations

import os
import subprocess
import wave
from datetime import timedelta

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = os.path.join(ROOT, "video", "shots")
OUT = os.path.join(ROOT, "video")
WORK = os.path.join(ROOT, "video", "post")
FONT = "Noto Sans CJK SC"

# (shot, [(start_s, end_s, zh, en), ...])   offsets are relative to the shot
SUBS = {
    "01_wakeup": [(0.8, 7.5, "清晨，猪圈里的小猪嘟嘟还在做着甜甜的梦。",
                   "Early in the morning, little pig Dudu is still dreaming sweet dreams.")],
    "02_wakeup_talk": [(0.6, 9.5, "妈妈猪：嘟嘟！太阳晒屁股啦！再不起来，早饭就变成晚饭了！",
                        "Mama Pig: Dudu! The sun is up! Get up, or breakfast will turn into dinner!"),
                       (10.5, 19.0, "嘟嘟：呜……还想再睡五分钟……好啦好啦……我起来了……",
                        "Dudu: Mmm... five more minutes... Okay, okay... I'm up...")],
    "03_breakfast": [(0.6, 9.0, "妈妈猪：今天有热乎乎的玉米粥和苹果片！",
                      "Mama Pig: Today we have warm corn porridge and apple slices!"),
                     (10.0, 18.0, "嘟嘟：香香香！", "Dudu: Yummy, yummy!")],
    "04_school_road": [(0.6, 9.0, "咩咩：嘟嘟早！你鼻子上还沾着粥呢！",
                        "Lamb: Morning, Dudu! You still have porridge on your nose!"),
                       (10.0, 19.0, "嘟嘟：嘿嘿……好吃的留下的痕迹！走吧走吧！",
                        "Dudu: Hehe... a trace of something tasty! Let's go!")],
    "05_classroom": [(0.6, 9.5, "老师：今天我们学习“分享”……嘟嘟，你为什么一直在拱桌脚？",
                      "Teacher: Today we learn about sharing... Dudu, why are you pushing the desk leg?"),
                     (10.5, 19.0, "嘟嘟：老师，我在思考！思考怎么把好吃的分享给大家！",
                      "Dudu: Teacher, I'm thinking! Thinking how to share the tasty food with everyone!")],
    "06_naptime": [(0.6, 9.0, "嘟嘟：汪汪……我好想睡觉……", "Dudu: Woof-woof... I'm so sleepy..."),
                   (10.0, 18.5, "汪汪：又开始做泥巴梦了……", "Puppy: He's having his mud dream again...")],
    "07_playground": [(0.6, 8.5, "老师：今天自由活动！", "Teacher: Free play time today!"),
                      (9.5, 19.0, "嘟嘟：我要比赛谁拱得最快！", "Dudu: I want to see who can dig the fastest!")],
    "08_dig": [(0.6, 18.5, "咩咩：嘟嘟你又把操场拱成迷宫了！",
                "Lamb: Dudu, you've turned the playground into a maze again!")],
    "09_pond": [(0.6, 9.0, "嘎嘎：来游泳呀！", "Duckling: Come swim!"),
                (10.0, 18.5, "嘟嘟：凉凉的……", "Dudu: It's cool...")],
    "10_splash": [(0.6, 18.5, "嘟嘟：噗噗噗！我是水里的小火箭！",
                   "Dudu: Splash, splash! I'm a little rocket in the water!")],
    "11_swim": [(0.6, 18.5, "旁白：大家在池塘里玩得可开心啦。",
                 "Narrator: Everyone is having so much fun in the pond.")],
    "12_dinner": [(0.6, 9.0, "妈妈猪：今天玩得开心吗？", "Mama Pig: Did you have fun today?"),
                  (10.0, 19.0, "嘟嘟：开心！我拱了操场、游了泳、还差点把桌子拱走！",
                   "Dudu: Yes! I dug up the playground, swam, and almost pushed the table away!")],
    "13_eat": [(0.6, 9.5, "妈妈猪：那明天继续努力……先把晚饭吃完！",
                "Mama Pig: Then keep it up tomorrow... finish your dinner first!"),
               (10.5, 19.0, "嘟嘟：嗯嗯……今天的南瓜最好吃！",
                "Dudu: Mmm... today's pumpkin is the tastiest!")],
    "14_bedtime": [(0.6, 9.5, "嘟嘟：妈妈……明天还要上学吗？", "Dudu: Mama... do we have school again tomorrow?"),
                   (10.5, 18.5, "妈妈猪：要的。", "Mama Pig: Yes.")],
    "15_sleep": [(0.6, 9.5, "嘟嘟：那我……明天还要……拱……", "Dudu: Then I... tomorrow I'll still... dig..."),
                 (10.5, 19.5, "旁白：就这样，嘟嘟带着甜甜的梦，睡着了。",
                  "Narrator: And so, with sweet dreams, Dudu fell asleep.")],
}
ORDER = ["01_wakeup", "02_wakeup_talk", "03_breakfast", "04_school_road", "05_classroom",
         "06_naptime", "07_playground", "08_dig", "09_pond", "10_splash", "11_swim",
         "12_dinner", "13_eat", "14_bedtime", "15_sleep"]

TITLE_ZH, TITLE_EN = "嘟嘟的一天", "Dodo's Day"


def dur(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    return float(out)


def ts(t):
    td = timedelta(seconds=t)
    return f"{int(td.total_seconds()//3600)}:{td.seconds//60:02d}:{td.seconds%60:02d}.{int(t*100)%100:02d}"


def make_bgm(path, total, sr=32000, bpm=76):
    """A soft music-box lullaby: C-major motif, sine partials, exponential decay."""
    beat = 60.0 / bpm
    # simple lullaby motif (semitone offsets from C4=261.63)
    motif = [0, 4, 7, 4, 5, 4, 2, 0]          # C E G E F E D C
    motif2 = [7, 9, 7, 4, 5, 7, 4, 0]
    scale = 2 ** (np.array(motif + motif2) / 12.0) * 261.63
    seq, i = [], 0
    while sum(beat * 0.5 for _ in seq) < total + 2:
        seq.append(scale[i % len(scale)] * (2 if (i // 16) % 2 else 1))
        i += 1
    n_per = int(beat * 0.5 * sr)
    buf = np.zeros(int(total * sr) + sr, dtype=np.float32)
    pos = 0
    for f in seq:
        if pos >= len(buf):
            break
        n = min(n_per * 2, len(buf) - pos)          # note rings past its slot
        t = np.arange(n) / sr
        env = np.exp(-3.2 * t)
        w = (np.sin(2 * np.pi * f * t) * 1.0
             + 0.35 * np.sin(2 * np.pi * 2 * f * t)
             + 0.14 * np.sin(2 * np.pi * 3 * f * t)
             + 0.06 * np.sin(2 * np.pi * 5 * f * t))
        buf[pos:pos + n] += (w * env).astype(np.float32) * 0.5
        pos += n_per
    # gentle pad an octave down
    t = np.arange(len(buf)) / sr
    pad = sum(np.sin(2 * np.pi * f * t) / (k + 1) for k, f in enumerate([65.4, 98.0, 130.8]))
    buf += (pad * 0.045).astype(np.float32)
    buf /= max(1e-6, np.abs(buf).max() * 1.15)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        pcm = (buf[:int(total * sr)] * 32767).astype("<i2")
        w.writeframes(np.repeat(pcm[:, None], 2, axis=1).tobytes())


def make_ass(path, starts, total):
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 960
PlayResY: 544
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: CN,{FONT},34,&H00FFFFFF,&H00FFFFFF,&H00202020,&H80000000,0,0,0,0,100,100,0,0,1,2.5,1.2,2,50,50,40,1
Style: EN,{FONT},20,&H00E8E8E8,&H00E8E8E8,&H00202020,&H80000000,0,0,0,0,100,100,0,0,1,2,0.8,2,50,50,18,1
Style: Title,{FONT},64,&H0030D8FF,&H0030D8FF,&H00101010,&H90000000,1,0,0,0,100,100,0,0,1,3,2,5,40,40,40,1
"""
    lines = [head, "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"]
    lines.append(f"Dialogue: 0,{ts(0)},{ts(4.5)},Title,,0,0,0,,{TITLE_ZH}\\N{{\\fs32\\c&H00FFFFFF&}}{TITLE_EN}\n")
    for name in ORDER:
        base = starts[name]
        for (a, b, zh, en) in SUBS[name]:
            txt = f"{zh}\\N{{\\rEN}}{en}"
            lines.append(f"Dialogue: 0,{ts(base + a)},{ts(base + b)},CN,,0,0,0,,{txt}\n")
    open(path, "w", encoding="utf-8").write("".join(lines))


def run(cmd):
    print(" ", " ".join(cmd[:6]), "...")
    subprocess.run(cmd, check=True)


def main():
    os.makedirs(WORK, exist_ok=True)
    # 1) concat
    concat_txt = os.path.join(WORK, "concat.txt")
    with open(concat_txt, "w") as f:
        for n in ORDER:
            f.write(f"file '{os.path.join(SHOTS, n + '.mp4')}'\n")
    concat_mp4 = os.path.join(WORK, "concat.mp4")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", concat_txt, "-c", "copy", concat_mp4])

    # 2) per-shot start times
    starts, t = {}, 0.0
    for n in ORDER:
        starts[n] = t
        t += dur(os.path.join(SHOTS, n + ".mp4"))
    total = t
    print(f"total {total:.2f}s")

    # 3) subtitles + bgm
    ass = os.path.join(WORK, "subs.ass")
    make_ass(ass, starts, total)
    bgm = os.path.join(WORK, "bgm.wav")
    make_bgm(bgm, total)
    print("  wrote subs.ass and bgm.wav")

    # 4) final: mix bgm (ducked low) + burn subs
    out = os.path.join(OUT, "dudu_full.mp4")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", concat_mp4, "-i", bgm,
         "-filter_complex",
         "[1:a]volume=0.20,afade=t=out:st=%.2f:d=4[bg];"
         "[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a];"
         "[0:v]ass=%s[v]" % (total - 4, ass.replace("\\", "/").replace(":", "\\:")),
         "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", out])
    print("wrote", out, f"{os.path.getsize(out)/2**20:.1f} MiB")


if __name__ == "__main__":
    main()
