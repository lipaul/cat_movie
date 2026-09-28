#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inspect a generated video: frame/audio stats and a Whisper transcription of
the speech track (to verify H3's native dialogue)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--transcribe", action="store_true")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--whisper", default=os.path.expanduser("~/work/models/whisper-small"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="CPU by default: the GPU is usually busy with ComfyUI/H3")
    args = ap.parse_args()

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height,nb_frames,duration",
         "-of", "default=nw=1", args.video],
        capture_output=True, text=True).stdout
    print(out.strip())

    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.video,
                    "-ac", "1", "-ar", "16000", tmp], check=True)
    w, sr = sf.read(tmp)
    print(f"audio: {w.shape} @ {sr}Hz  rms={np.sqrt((w**2).mean()):.4f} peak={np.abs(w).max():.4f}")

    if args.transcribe:
        import torch
        from transformers import pipeline
        use_cuda = args.device == "cuda" and torch.cuda.is_available()
        asr = pipeline("automatic-speech-recognition", model=args.whisper,
                       device=0 if use_cuda else -1,
                       dtype=torch.float16 if use_cuda else torch.float32)
        res = asr(tmp, generate_kwargs={"language": args.lang, "task": "transcribe"})
        print("transcription:", res["text"].strip())
    os.remove(tmp)


if __name__ == "__main__":
    main()
