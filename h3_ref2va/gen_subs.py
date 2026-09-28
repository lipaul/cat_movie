#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the bilingual ASS subtitles for《嘟嘟的一天》from the *actual* speech.

Why: LTX-2.5 decides its own speech timing, so hand-estimated cue times drift by
seconds. Timing here comes from an energy-based VAD on each shot's audio
(robust against Whisper hallucinating words over music/ambience); the cue *text*
is the script's.

The detected speech spans are fitted to the script lines: too many spans are
merged at their smallest gaps, too few are split at their longest span, so every
line gets a sensible window without ever inverting start/end.

Usage:
  env/comfy/bin/python h3_ref2va/gen_subs.py                # -> video/post/subs.ass
  env/comfy/bin/python h3_ref2va/gen_subs.py --debug        # print detected spans
"""
from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np

from story_data import ORDER, SHOT_LINES, TITLE_EN, TITLE_ZH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = os.path.join(ROOT, "video", "shots")

LEAD = 0.12      # cue appears slightly before the speech
TAIL = 0.45      # and lingers after
MERGE_GAP = 0.45  # silences shorter than this don't split an utterance
MIN_SPEECH = 0.14  # ignore blips shorter than this
SR = 16000


def ffprobe_dur(path: str) -> float:
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout.strip())


def read_audio(path: str) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def speech_spans(audio: np.ndarray, hop: int = 160, win: int = 320):
    """Energy VAD -> [(start, end), ...] in seconds."""
    if audio.size < win:
        return []
    n = 1 + (audio.size - win) // hop
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    frames = audio[idx]
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)

    noise = float(np.percentile(rms, 30))
    peak = float(np.percentile(rms, 95))
    thr = max(0.006, noise + 0.25 * (peak - noise))
    active = rms > thr

    spans, start = [], None
    for i, a in enumerate(active):
        t = i * hop / SR
        if a and start is None:
            start = t
        elif not a and start is not None:
            spans.append([start, t + win / SR])
            start = None
    if start is not None:
        spans.append([start, len(active) * hop / SR + win / SR])

    # merge short gaps, drop blips
    merged = []
    for s in spans:
        if merged and s[0] - merged[-1][1] < MERGE_GAP:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(s, e) for s, e in merged if e - s >= MIN_SPEECH]


def _gaps(spans):
    return [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]


def fit_lines(spans, n_lines, dur):
    """Merge/split detected spans until there are exactly `n_lines` of them."""
    spans = [list(s) for s in spans]
    if not spans:
        # no speech detected: spread the lines proportionally over the shot
        step = dur / max(1, n_lines)
        spans = [[i * step + 0.3, (i + 1) * step - 0.5] for i in range(n_lines)]
    while len(spans) > n_lines:
        g = _gaps(spans)
        i = int(np.argmin(g))
        spans[i][1] = spans[i + 1][1]
        del spans[i + 1]
    while len(spans) < n_lines:
        lens = [e - s for s, e in spans]
        i = int(np.argmax(lens))
        s, e = spans[i]
        mid = (s + e) / 2
        spans[i] = [s, mid]
        spans.insert(i + 1, [mid, e])
    return [(s, e) for s, e in spans]


def align(spans, lines, dur):
    fits = fit_lines(spans, len(lines), dur)
    cues = []
    for (s, e), (zh, en) in zip(fits, lines):
        s2 = max(0.0, s - LEAD)
        # cap the cue at a length estimated from the line (~4 chars/second)
        cap = max(1.2, len(zh) / 4.0 + 0.9)
        e2 = min(dur, min(e + TAIL, s2 + cap))
        if e2 - s2 < 0.7:
            e2 = min(dur, s2 + 0.7)
        cues.append((s2, e2, zh, en))
    return cues


def ts(t: float) -> str:
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


FONT = "Noto Sans CJK SC"
HEAD = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 960
PlayResY: 544
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: CN,{FONT},29,&H00FFFFFF,&H00FFFFFF,&H00181818,&H78000000,0,0,0,0,100,100,0,0,1,2.2,1.0,2,60,60,60,1
Style: EN,{FONT},22,&H00DCDCDC,&H00DCDCDC,&H00181818,&H78000000,0,0,0,0,100,100,0,0,1,1.8,0.8,2,60,60,26,1
Style: Title,{FONT},62,&H0030D8FF,&H0030D8FF,&H00101010,&H90000000,1,0,0,0,100,100,0,0,1,3,2,8,40,40,36,1
"""


def write_ass(path, cues, total):
    lines = [HEAD, "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"]
    lines.append(f"Dialogue: 0,{ts(0)},{ts(4.5)},Title,,0,0,0,,{TITLE_ZH}\\N"
                 f"{{\\fs32\\c&H00FFFFFF&}}{TITLE_EN}\n")
    for start, end, zh, en in cues:
        lines.append(f"Dialogue: 0,{ts(start)},{ts(end)},CN,,0,0,0,,{zh}\n")
        lines.append(f"Dialogue: 0,{ts(start)},{ts(end)},EN,,0,0,0,,{en}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    return path


def build(out_path=None, verbose=True, debug=False):
    out_path = out_path or os.path.join(ROOT, "video", "post", "subs.ass")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_cues, cur = [], 0.0
    for name in ORDER:
        video = os.path.join(SHOTS, f"{name}.mp4")
        d = ffprobe_dur(video)
        spans = speech_spans(read_audio(video))
        lines = SHOT_LINES.get(name, [])
        cues = align(spans, lines, d)
        if debug:
            print(f"  [{name}] spans={[(round(s, 2), round(e, 2)) for s, e in spans]}")
        if verbose:
            print(f"  [{name}] speech={len(spans)} lines={len(lines)} "
                  f"cue_dur={[round(e - s, 2) for s, e, _, _ in cues]}", flush=True)
        for s, e, zh, en in cues:
            all_cues.append((cur + s, cur + e, zh, en))
        cur += d
    all_cues.sort(key=lambda c: c[0])
    write_ass(out_path, all_cues, cur)
    if verbose:
        print(f"wrote {out_path}  ({len(all_cues)} cues, {cur:.2f}s)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    build(a.out, debug=a.debug)
