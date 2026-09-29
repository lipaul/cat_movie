#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Post-production for《嘟嘟的一天》.

  * concat the 15 LTX shots, with a per-shot luma match toward the film's mean
    (mild, clamped) plus a light grade — this is the cheap half of "shot
    continuity": it irons out small exposure/colour drift between shots
  * real-timed bilingual subtitles from `gen_subs.py` (Whisper-aligned)
  * one continuous music-box BGM bed, an opening title card

Output: video/dudu_full.mp4
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import wave

import numpy as np

from story_data import ORDER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = os.path.join(ROOT, "video", "shots")
OUT = os.path.join(ROOT, "video")
WORK = os.path.join(ROOT, "video", "post")


def dur(path: str) -> float:
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout.strip())


def ts(t: float) -> str:
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def deghost_video(src, dst, y0=396, thr=178, toph=22, dil=2, radius=5):
    """Inpaint LTX's baked-in text out of a shot (keeps the original audio)."""
    import cv2
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = dst + ".v.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    ktop = np.ones((7, 7), np.uint8)
    kdil = np.ones((3, 3), np.uint8)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        band = g[y0:, :]
        th = cv2.morphologyEx(band, cv2.MORPH_TOPHAT, ktop)
        m = ((band > thr) & (th > toph)).astype(np.uint8) * 255
        m = cv2.dilate(m, kdil, iterations=dil)
        mask = np.zeros(g.shape, np.uint8)
        mask[y0:, :] = m
        vw.write(cv2.inpaint(frame, mask, radius, cv2.INPAINT_TELEA))
        n += 1
    cap.release()
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-i", src,
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", dst],
                   check=True)
    os.remove(tmp)
    return n


def mean_luma(path: str) -> float:
    """Mean luma (0-255) of a shot, via signalstats YAVG (1 sample/second)."""
    out = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", path, "-vf",
         "fps=1,signalstats,metadata=print:key=lavfi.signalstats.YAVG",
         "-f", "null", "-"],
        capture_output=True, text=True).stderr
    vals = [float(m) for m in re.findall(r"YAVG=([\d.]+)", out)]
    return float(np.mean(vals)) if vals else 128.0


def make_bgm(path, total, sr=32000, bpm=76):
    beat = 60.0 / bpm
    motif = [0, 4, 7, 4, 5, 4, 2, 0, 7, 9, 7, 4, 5, 7, 4, 0]
    scale = 2 ** (np.array(motif) / 12.0) * 261.63
    n_per = int(beat * 0.5 * sr)
    buf = np.zeros(int(total * sr) + sr, dtype=np.float32)
    pos, i = 0, 0
    while pos < len(buf):
        f = scale[i % len(scale)] * (2 if (i // 16) % 2 else 1)
        n = min(n_per * 2, len(buf) - pos)
        t = np.arange(n) / sr
        w = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t)
             + 0.14 * np.sin(2 * np.pi * 3 * f * t) + 0.06 * np.sin(2 * np.pi * 5 * f * t))
        buf[pos:pos + n] += (w * np.exp(-3.2 * t)).astype(np.float32) * 0.5
        pos += n_per; i += 1
    t = np.arange(len(buf)) / sr
    pad = sum(np.sin(2 * np.pi * f * t) / (k + 1) for k, f in enumerate([65.4, 98.0, 130.8]))
    buf += (pad * 0.045).astype(np.float32)
    buf /= max(1e-6, np.abs(buf).max() * 1.15)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        pcm = (buf[:int(total * sr)] * 32767).astype("<i2")
        w.writeframes(np.repeat(pcm[:, None], 2, axis=1).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-grade", action="store_true", help="skip the per-shot luma match")
    ap.add_argument("--grade-strength", type=float, default=0.5,
                    help="0=none, 1=fully match the film mean (clamped to +-0.02)")
    ap.add_argument("--no-subs", action="store_true")
    ap.add_argument("--band", default="gradient", choices=["gradient", "solid", "none"],
                    help="subtitle bar: gradient (default) / solid black / none")
    ap.add_argument("--band-height", type=int, default=150,
                    help="bar height in pixels, measured from the bottom (default 150)")
    ap.add_argument("--band-alpha", type=float, default=0.96,
                    help="opacity at the bottom of the bar, 0..1 (default 0.96)")
    ap.add_argument("--deghost", action="store_true",
                    help="experimental: inpaint LTX's baked-in text out of each shot "
                         "(may leave smudges on textured backgrounds)")
    a = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    shots = [os.path.join(SHOTS, f"{n}.mp4") for n in ORDER]
    for s in shots:
        if not os.path.isfile(s):
            raise SystemExit(f"missing shot: {s}")

    if a.deghost:
        dh = os.path.join(WORK, "deghost")
        os.makedirs(dh, exist_ok=True)
        print("  deghosting (inpaint) ...")
        for i, n in enumerate(ORDER):
            src = os.path.join(SHOTS, f"{n}.mp4")
            dst = os.path.join(dh, f"{n}.mp4")
            nf = deghost_video(src, dst)
            print(f"    [{n}] {nf} frames", flush=True)
            shots[i] = dst

    total = sum(dur(s) for s in shots)
    print(f"total {total:.2f}s, {len(shots)} shots")

    # 1) subtitles (Whisper-aligned)
    subs = None
    if not a.no_subs:
        import gen_subs
        subs = gen_subs.build(os.path.join(WORK, "subs.ass"))

    # 2) BGM
    bgm = os.path.join(WORK, "bgm.wav")
    make_bgm(bgm, total)
    print(f"  bgm {os.path.getsize(bgm)/2**20:.1f} MiB")

    # 3) per-shot luma match
    grade = []
    if not a.no_grade:
        lumas = [mean_luma(s) for s in shots]
        target = float(np.mean(lumas))
        for y in lumas:
            d = float(np.clip((target - y) / 255.0 * a.grade_strength, -0.02, 0.02))
            grade.append(d)
        print("  luma:", " ".join(f"{y:.1f}" for y in lumas), f"-> target {target:.1f}")

    # 4) one pass: per-shot grade -> concat -> [bottom band] -> ass ; audio concat -> amix bgm
    N = len(shots)
    chain = []
    for i in range(N):
        eq = "eq=saturation=1.03" + (f":brightness={grade[i]:.5f}" if grade else "")
        chain.append(f"[{i}:v]{eq},format=yuv420p[v{i}]")
        chain.append(f"[{i}:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a{i}]")
    chain.append("".join(f"[v{i}]" for i in range(N)) + f"concat=n={N}:v=1:a=0[vcat]")
    chain.append("".join(f"[a{i}]" for i in range(N)) + f"concat=n={N}:v=0:a=1[acat]")

    vcur = "[vcat]"
    if a.band != "none" and a.band_height > 0:
        H = min(a.band_height, 544)
        top = 544 - H
        if a.band == "solid":
            band = f"drawbox=x=0:y={top}:w=iw:h={H}:color=black@{a.band_alpha:.3f}:t=fill"
        else:
            steps_n = 5
            step_h = max(1, H // steps_n)
            parts = []
            for k in range(steps_n):
                y = top + k * step_h
                h = (H - k * step_h) if k == steps_n - 1 else step_h
                al = a.band_alpha * (k + 1) / steps_n
                parts.append(f"drawbox=x=0:y={y}:w=iw:h={h}:color=black@{al:.3f}:t=fill")
            band = ",".join(parts)
        chain.append(f"{vcur}{band}[vband]")
        vcur = "[vband]"
    if subs:
        esc = subs.replace("\\", "/").replace(":", "\\:")
        chain.append(f"{vcur}ass={esc}[vsub]")
        vcur = "[vsub]"
    chain.append(f"[{N}:a]volume=0.20,afade=t=out:st={max(0, total - 4):.2f}:d=4[bg]")
    chain.append("[acat][bg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]")

    filter_complex = ";".join(chain)
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for s in shots:
        cmd += ["-i", s]
    cmd += ["-i", bgm, "-filter_complex", filter_complex,
            "-map", vcur, "-map", "[aout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            os.path.join(OUT, "dudu_full.mp4")]
    print("  encoding ...")
    subprocess.run(cmd, check=True)
    out = os.path.join(OUT, "dudu_full.mp4")
    print(f"wrote {out}  {os.path.getsize(out)/2**20:.1f} MiB  {total:.2f}s")


if __name__ == "__main__":
    main()
