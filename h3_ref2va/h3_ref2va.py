#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MiniMax-H3 `ref2va`: reference-image -> character-consistent video + audio.

Native int8 DiT (reused from the proven `h3_t2va.py`) driven by a faithful port
of the diffusers `ref2va` contract:

  * image references encoded by the video VAE (`encode_vae_condition`)
  * prompt presentation + Qwen3-VL hidden state[50] via HF transformers
  * packed sequence `[text | reference blocks | target audio | target video]`
  * anchors noised to t=0.999 and held there; generated rows step down their
    own schedules

Usage:
  .venv/bin/python h3_ref2va.py --prompt "..." --ref refs/sheet/cat_front.png \
      --num-frames 124 --steps 50 --seed 42 --output video/cat.mp4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ref2va_layout as L  # noqa: E402
import ref2va_conditioner as C  # noqa: E402

DEFAULT_MODEL_DIR = os.path.expanduser("~/work/models/MiniMax-H3")
H3_T2VA = os.path.expanduser("~/paul_arc/test/h3_nv/h3_t2va.py")


def _load_h3_t2va():
    """Import the proven t2va implementation as a module (it only runs main()
    under __main__)."""
    path = os.environ.get("H3_T2VA_PATH", H3_T2VA)
    if not os.path.isfile(path):
        raise SystemExit(f"h3_t2va.py not found at {path}; set H3_T2VA_PATH")
    spec = importlib.util.spec_from_file_location("h3_t2va", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h3_t2va"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="MiniMax-H3 ref2va (reference images)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--ref", action="append", required=True, help="reference image path (repeatable, <=9)")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--output", default="ref2va.mp4")
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--aspect-ratio", default="16:9")
    ap.add_argument("--num-frames", type=int, default=124)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--no-stream-adaln", action="store_true")
    ap.add_argument("--embeds-cache", default=None, help="path to cache/load prompt embeds (.pt)")
    args = ap.parse_args()

    if len(args.ref) > 9:
        raise SystemExit("at most 9 image references")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    h3 = _load_h3_t2va()
    from PIL import Image
    from transformers import AutoTokenizer, AutoProcessor, Qwen3VLForConditionalGeneration
    from diffusers import AutoencoderKLMiniMaxH3

    t0 = time.time()
    device = torch.device("cuda")
    model_dir = args.model_dir

    # ---- geometry ----
    h, w = args.height, args.width
    if h is None or w is None:
        h, w = L.resolve_canvas_size(*[float(x) for x in args.aspect_ratio.split(":")])
    for name, v in (("height", h), ("width", w)):
        if v % L.CANVAS_MULTIPLE:
            raise SystemExit(f"{name}={v} must be a multiple of {L.CANVAS_MULTIPLE}")
    nf = L.align_num_frames(args.num_frames)
    dur = nf / L.FPS
    if not (5.0 <= dur <= 15.0):
        raise SystemExit(f"duration {dur:.2f}s outside 5-15s")
    nlf = L.video_latent_num_frames(nf)
    na = L.audio_latent_num_frames(nf)
    lh, lw = h // L.VAE_SPATIAL_RATIO, w // L.VAE_SPATIAL_RATIO
    patch = (1, 2, 2)
    h3.log(f"canvas {w}x{h} | {nf} frames ({dur:.2f}s) | latents 24x{nlf}x{lh}x{lw} | audio {na}/ch")

    # ---- references + prompt embeds ----
    refs_norm = [C.normalize_image_reference(Image.open(p)) for p in args.ref]
    h3.log(f"{len(refs_norm)} references: " + ", ".join(str(r.size) for r in refs_norm))

    if args.embeds_cache and os.path.isfile(args.embeds_cache):
        blob = torch.load(args.embeds_cache, map_location="cpu")
        prompt_embeds, text_tags = blob["embeds"], blob["token_tags"]
        h3.log(f"loaded prompt embeds cache {tuple(prompt_embeds.shape)}")
    else:
        tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"), trust_remote_code=True)
        proc = AutoProcessor.from_pretrained(os.path.join(model_dir, "processor"), trust_remote_code=True)
        vision, counts = C.gather_image_vision(proc, refs_norm)
        token_ids, text_tags = C.build_presentation(tok, args.prompt, counts)
        h3.log(f"presentation {len(token_ids)} tokens | vision {counts}")
        quant = None
        try:
            from transformers import TorchAoConfig as TATorchAo
            from torchao.quantization import Int8WeightOnlyConfig
            quant = TATorchAo(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=["model.visual", "model.language_model.embed_tokens",
                                        "model.language_model.norm", "lm_head"],
            )
        except Exception:  # noqa: BLE001
            h3.log("torchao int8 unavailable; using bf16")
        te = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(model_dir, "text_encoder"), dtype=torch.bfloat16,
            quantization_config=quant, low_cpu_mem_usage=True,
        )
        te.eval()
        t1 = time.time()
        prompt_embeds = C.encode_prompt_embeds(te, proc, token_ids, vision, 50).cpu()
        h3.log(f"conditioner forward {time.time() - t1:.0f}s -> {tuple(prompt_embeds.shape)}")
        del te
        h3.free_cuda()
        if args.embeds_cache:
            torch.save({"embeds": prompt_embeds, "token_tags": torch.tensor(text_tags)}, args.embeds_cache)
            h3.log(f"saved embeds cache {args.embeds_cache}")

    text_token_tags = torch.tensor(text_tags, dtype=torch.long)

    # ---- encode references with the video VAE ----
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        os.path.join(model_dir, "vae"), torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    condition_latents = [C.encode_image_reference(vae, r) for r in refs_norm]
    del vae
    h3.free_cuda()
    h3.log("condition latents: " + ", ".join(str(tuple(c.shape)) for c in condition_latents))

    refs = [L.Ref.image(c.shape[2], c.shape[3], c.shape[4]) for c in condition_latents]

    # ---- layout ----
    (position_ids, token_tags, video_idx, audio_idx, text_idx,
     n_cond_v, n_cond_a) = L.build_ref2va_packed_sequence(
        text_token_tags, refs, nlf, lh, lw, na, patch,
        L.AUDIO_CHANNELS, L.AUDIO_TAG, L.VIDEO_TAG,
    )
    S = position_ids.shape[0]
    n_tgt_v = nlf * (lh // patch[1]) * (lw // patch[2])
    n_tgt_a = na * L.AUDIO_CHANNELS
    h3.log(f"packed S={S} | text {len(text_tags)} | cond_v {n_cond_v} | target_v {n_tgt_v} "
           f"| cond_a {n_cond_a} | target_a {n_tgt_a}")
    position_ids = position_ids.to(device)
    token_tags = token_tags.to(device)
    video_idx, audio_idx, text_idx = video_idx.to(device), audio_idx.to(device), text_idx.to(device)

    # ---- noise: conditioning first, then video, then audio ----
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    cond_rows = []
    for c in condition_latents:
        noise = torch.randn(c.shape, generator=gen, device=device, dtype=torch.float32)
        noised = L.H3Scheduler.scale_noise(c.to(device), L.KEYFRAME_NOISE_AUG, noise)
        cond_rows.append(h3.patchify(noised, patch))
    condition_rows = torch.cat(cond_rows)
    video_rows = torch.cat([condition_rows,
                            h3.patchify(torch.randn(1, 24, nlf, lh, lw, generator=gen, device=device), patch)])
    audio_rows = torch.randn(n_tgt_a, L.AUDIO_LATENT_CHANNELS, generator=gen, device=device)

    sv = L.H3Scheduler(L.FLOW_SHIFT_VIDEO)
    sa = L.H3Scheduler(L.FLOW_SHIFT_AUDIO)
    sv.set_timesteps(args.steps, device)
    sa.set_timesteps(args.steps, device)
    n_eval = len(sv.timesteps)
    h3.log(f"{n_eval} evals | video sigma {float(sv.sigmas[0]):.4f}->0")

    # ---- DiT ----
    model = h3.load_dit(os.path.join(model_dir, "Ref2VA", "transformer"), device,
                        quantize=not args.bf16, stream_adaln=not args.no_stream_adaln)
    te = model.encode_text(prompt_embeds.to(device))
    del prompt_embeds
    cos, sin = model.rope(position_ids)
    buf = te.new_zeros((1, S, model.hidden))
    CH = 8192
    t1 = time.time()
    for i in range(n_eval):
        t_v, t_a = sv.timesteps[i], sa.timesteps[i]
        unique_t, row_t_idx = L.build_row_timesteps(
            video_idx, audio_idx, n_cond_v, n_cond_a, len(text_tags),
            float(t_v), float(t_a), max(float(t_v), L.KEYFRAME_NOISE_AUG), 1.0,
        )
        unique_t, row_t_idx = unique_t.to(device), row_t_idx.to(device)
        temb_silu = torch.nn.functional.silu(model.time_mlp(model.time_proj(unique_t)))
        adaln_idx = row_t_idx * L.MODALITY_NUM + token_tags
        packed = model(buf, video_rows[None], audio_rows[None], te, temb_silu,
                       adaln_idx, cos, sin, video_idx, audio_idx, text_idx)
        x_final = model.head(packed, temb_silu, row_t_idx)
        del packed
        for s in range(n_cond_v, video_rows.shape[0], CH):
            e = min(s + CH, video_rows.shape[0])
            v = model.proj_out(x_final[0, video_idx[s:e]].float())
            video_rows[s:e] = sv.step(video_rows[s:e], v, t_v, i)
            del v
        for s in range(n_cond_a, audio_rows.shape[0], CH):
            e = min(s + CH, audio_rows.shape[0])
            v = model.audio_proj_out(x_final[0, audio_idx[s:e]].float())
            audio_rows[s:e] = sa.step(audio_rows[s:e], v, t_a, i)
            del v
        del x_final
        el = time.time() - t1
        if (i + 1) % max(1, n_eval // 10) == 0 or i == n_eval - 1:
            h3.log(f"  step {i + 1}/{n_eval} | {el:.0f}s, {el / (i + 1):.1f}s/step | "
                   f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    del model, te, buf
    h3.free_cuda()

    # ---- decode target rows only ----
    target_video_rows = video_rows[n_cond_v:]
    target_audio_rows = audio_rows[n_cond_a:]
    vcfg = json.load(open(os.path.join(model_dir, "Ref2VA", "video_vae", "config.json")))
    vmean = torch.tensor(vcfg["latents_mean"], device=device).view(1, -1, 1, 1, 1)
    vstd = torch.tensor(vcfg["latents_std"], device=device).view(1, -1, 1, 1, 1)
    lat = h3.unpatchify(target_video_rows, patch, 24, nlf, lh, lw) * vstd + vmean
    acfg = json.load(open(os.path.join(model_dir, "Ref2VA", "audio_vae", "config.json")))
    amean = torch.tensor(acfg["latents_mean"], device=device).view(1, -1, 1)
    astd = torch.tensor(acfg["latents_std"], device=device).view(1, -1, 1)
    a_lat = target_audio_rows.reshape(L.AUDIO_CHANNELS, na, L.AUDIO_LATENT_CHANNELS).permute(0, 2, 1) * astd + amean
    del video_rows, audio_rows, target_video_rows, target_audio_rows
    h3.free_cuda()

    vae = h3.load_bundled_vae(os.path.join(model_dir, "Ref2VA"), "video_vae", device)
    frames = h3.video_decode(vae, lat, nf)
    del vae, lat
    h3.free_cuda()
    avae = h3.load_bundled_vae(os.path.join(model_dir, "Ref2VA"), "audio_vae", device)
    wav = h3.audio_decode(avae, a_lat)
    del avae, a_lat
    h3.free_cuda()
    h3.log(f"decoded {frames.shape[0]} frames {frames.shape[2]}x{frames.shape[1]} | "
           f"audio {wav.shape[1] / 32000:.2f}s")

    out = args.output if args.output.endswith(".mp4") else args.output + ".mp4"
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    h3.write_mp4(out, (np.clip(frames, 0, 1) * 255 + 0.5).astype(np.uint8), L.FPS)
    h3.log(f"wrote {out} ({os.path.getsize(out) / 2**20:.1f} MiB) [video only; mux audio separately]")
    wav_path = os.path.splitext(out)[0] + ".wav"
    import soundfile as sf
    sf.write(wav_path, wav.T.astype(np.float32), 32000, subtype="PCM_16")
    h3.log(f"wrote {wav_path}")
    h3.log(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
