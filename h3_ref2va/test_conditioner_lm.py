#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load the 32B Qwen3-VL conditioner in int8 and run the ref2va forward,
reading `hidden_states[50]`. Reports VRAM, timing and the embedding shape."""
from __future__ import annotations

import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref2va_conditioner as C  # noqa: E402

MODELS = os.path.expanduser("~/work/models/MiniMax-H3")
LAYER = 50


def main():
    t0 = time.time()
    from transformers import AutoTokenizer, AutoProcessor, Qwen3VLForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(os.path.join(MODELS, "tokenizer"), trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(os.path.join(MODELS, "processor"), trust_remote_code=True)

    quant = None
    try:
        from transformers import TorchAoConfig as TransformersTorchAoConfig
        from torchao.quantization import Int8WeightOnlyConfig

        quant = TransformersTorchAoConfig(
            Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=[
                "model.visual", "model.language_model.embed_tokens",
                "model.language_model.norm", "lm_head",
            ],
        )
        print("using torchao int8 weight-only")
    except Exception as e:  # noqa: BLE001
        print(f"torchao int8 unavailable ({e}); falling back to bf16")

    print("loading text encoder...")
    te = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(MODELS, "text_encoder"),
        dtype=torch.bfloat16,
        quantization_config=quant,
        low_cpu_mem_usage=True,
    )
    te.eval()
    print(f"loaded in {time.time() - t0:.0f}s | VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    # one image reference
    img = Image.new("RGB", (900, 640), (200, 120, 40))
    norm = C.normalize_image_reference(img)
    vision, counts = C.gather_image_vision(proc, [norm])
    token_ids, token_tags = C.build_presentation(tok, "The character waves at the camera.", counts)
    print(f"presentation {len(token_ids)} tokens, vision counts {counts}")

    t1 = time.time()
    emb = C.encode_prompt_embeds(te, proc, token_ids, vision, text_encoder_layer=LAYER)
    torch.cuda.synchronize()
    print(f"forward {time.time() - t1:.0f}s | embeds {tuple(emb.shape)} dtype {emb.dtype} "
          f"| peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print(f"tags set {sorted(set(token_tags))}")

    np_save = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
    os.makedirs(np_save, exist_ok=True)
    torch.save({"embeds": emb.cpu(), "token_tags": torch.tensor(token_tags)}, os.path.join(np_save, "prompt_embeds_smoke.pt"))
    print("saved golden/prompt_embeds_smoke.pt")
    print("OK")


if __name__ == "__main__":
    main()
