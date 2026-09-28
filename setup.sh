#!/usr/bin/env bash
# =============================================================================
# setup.sh — 从零构建"本地生成视频"所需的 Python / uv 环境
#
# 做三件事：
#   1. env/comfy  : ComfyUI + torch(cu128) + 依赖  → 跑 FLUX.2 出图 与 LTX-2.5 出视频
#   2. env/h3     : diffusers + transformers + torchao → 跑 MiniMax-H3 ref2va（高质量对白）
#   3. 模型接线   : 把本机已有权重软链进 ComfyUI/models，并做一次环境自检
#
# 用法：
#   ./setup.sh                 # 建两个 venv + 接线 + 自检（模型已在本地时不下载）
#   ./setup.sh --no-h3         # 只建 ComfyUI/LTX 环境
#   ./setup.sh --download      # 额外下载缺失的模型（FLUX.2 / LTX-2.5 / whisper，很大）
#   ./setup.sh --no-link       # 不碰 ComfyUI/models
#   ./setup.sh --no-verify     # 跳过自检
#
# 可覆盖的环境变量：
#   WORK_MODELS=~/work/models   COMFY_REV=1568e6c    COMFY_PORT=8188
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128
#
# 幂等：可重复运行；已存在的东西会跳过，不会重下 torch。
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

WORK_MODELS="${WORK_MODELS:-$HOME/work/models}"
COMFY_ENV="$PROJECT_DIR/env/comfy"
H3_ENV="$PROJECT_DIR/env/h3"
COMFY_DIR="$PROJECT_DIR/ComfyUI"
COMFY_REV="${COMFY_REV:-1568e6c}"
COMFY_PORT="${COMFY_PORT:-8188}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

WITH_H3=1
DO_DOWNLOAD=0
DO_LINK=1
DO_VERIFY=1

# ---------------------------------------------------------------- helpers ---
log()  { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn ]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

py_has() {  # py_has <python> <module>
  "$1" -c "import $2" >/dev/null 2>&1
}

uvpip() {  # uvpip <venv-python> <args...>
  uv pip install --python "$1" "${@:2}"
}

usage() {
  sed -n '2,/^set -e/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  exit 0
}

while (($#)); do
  case "$1" in
    --with-h3)   WITH_H3=1 ;;
    --no-h3)     WITH_H3=0 ;;
    --download)  DO_DOWNLOAD=1 ;;
    --no-download) DO_DOWNLOAD=0 ;;
    --link)      DO_LINK=1 ;;
    --no-link)   DO_LINK=0 ;;
    --verify)    DO_VERIFY=1 ;;
    --no-verify) DO_VERIFY=0 ;;
    -h|--help)   usage ;;
    *) die "未知参数：$1（用 -h 看用法）" ;;
  esac
  shift
done

# ------------------------------------------------------------- 1. prereqs ---
step_prereqs() {
  log "1/6 前置检查"
  have uv   || die "缺少 uv：curl -LsSf https://astral.sh/uv/install.sh | sh"
  have git  || die "缺少 git"
  have ffmpeg || warn "缺少 ffmpeg（后期拼接/烧字幕需要）：sudo apt install ffmpeg"
  if have ffmpeg; then
    ffmpeg -hide_banner -h filter=subtitles >/dev/null 2>&1 \
      || warn "ffmpeg 没有 libass（subtitles 滤镜），烧中文字幕会失败"
  fi
  have nvidia-smi || die "缺少 nvidia-smi（本项目需要 NVIDIA GPU）"
  log "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
  log "  uv : $(uv --version)"
  local free_kb free_gb
  free_kb=$(df -Pk "$PROJECT_DIR" | awk 'NR==2{print $4}')
  free_gb=$((free_kb / 1024 / 1024))
  log "  磁盘可用: ${free_gb} GB"
  (( free_gb >= 80 )) || warn "磁盘可用不足 80GB；若用 --download 会不够"
}

# ------------------------------------------------------- 2. ComfyUI / LTX ---
step_comfy_env() {
  log "2/6 env/comfy（ComfyUI + LTX-2.5 路线）"
  if [[ ! -x "$COMFY_ENV/bin/python" ]]; then
    log "  创建 venv（python 3.12）"
    uv venv --python 3.12 "$COMFY_ENV"
  else
    log "  venv 已存在，复用"
  fi
  local PY="$COMFY_ENV/bin/python"

  if py_has "$PY" torch; then
    log "  torch 已安装：$("$PY" -c 'import torch;print(torch.__version__)')，跳过"
  else
    log "  安装 torch/torchvision（cu128，约 10 分钟）"
    uvpip "$PY" torch torchvision --index-url "$TORCH_INDEX"
  fi

  if [[ ! -d "$COMFY_DIR/.git" ]]; then
    log "  clone ComfyUI @ $COMFY_REV"
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
    git -C "$COMFY_DIR" checkout -q "$COMFY_REV" 2>/dev/null \
      || warn "  找不到 commit $COMFY_REV，使用当前 HEAD"
  else
    log "  ComfyUI 已存在（$(git -C "$COMFY_DIR" rev-parse --short HEAD)）"
  fi

  log "  安装 ComfyUI 依赖"
  uvpip "$PY" -r "$COMFY_DIR/requirements.txt"
  uvpip "$PY" soundfile huggingface_hub   # 抽检转写 / hf CLI
}

# ------------------------------------------------------------ 3. env/h3 -----
step_h3_env() {
  (( WITH_H3 )) || { log "3/6 跳过 env/h3（--no-h3）"; return; }
  log "3/6 env/h3（MiniMax-H3 ref2va 路线）"
  if [[ ! -x "$H3_ENV/bin/python" ]]; then
    uv venv --python 3.12 "$H3_ENV"
  else
    log "  venv 已存在，复用"
  fi
  local PY="$H3_ENV/bin/python"

  if py_has "$PY" torch; then
    log "  torch 已安装：$("$PY" -c 'import torch;print(torch.__version__)')，跳过"
  else
    log "  安装 torch/torchvision（cu128，约 10 分钟）"
    uvpip "$PY" torch torchvision --index-url "$TORCH_INDEX"
  fi
  log "  安装 diffusers / transformers / torchao ..."
  uvpip "$PY" diffusers transformers accelerate torchao einops soundfile av \
               safetensors pillow numpy huggingface_hub
}

# ------------------------------------------------------- 4. 模型接线 ---------
LINKED=0; MISSING=0
link_one() {  # link_one <src> <dest-dir>
  local src="$1" dst="$2"
  mkdir -p "$dst"
  if [[ -f "$src" ]]; then
    ln -sf "$src" "$dst/"
    LINKED=$((LINKED + 1))
  else
    warn "  缺少模型：$src"
    MISSING=$((MISSING + 1))
  fi
}

step_link_models() {
  (( DO_LINK )) || { log "4/6 跳过模型接线（--no-link）"; return; }
  log "4/6 模型接线 → ComfyUI/models"

  # FLUX.2（出图）
  local FX="$PROJECT_DIR/models/flux2/split_files"
  link_one "$FX/diffusion_models/flux2_dev_fp8mixed.safetensors"                 "$COMFY_DIR/models/diffusion_models"
  link_one "$FX/text_encoders/mistral_3_small_flux2_fp8.safetensors"             "$COMFY_DIR/models/text_encoders"
  link_one "$FX/vae/flux2-vae.safetensors"                                       "$COMFY_DIR/models/vae"
  link_one "$FX/loras/Flux2TurboComfyv2.safetensors"                             "$COMFY_DIR/models/loras"

  # LTX-2.5（出视频）；Ada 用 bf16 + fp8 与 -conv VAE
  local LT="$WORK_MODELS/ltx-2.5"
  link_one "$LT/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" "$COMFY_DIR/models/diffusion_models"
  link_one "$LT/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"          "$COMFY_DIR/models/text_encoders"
  link_one "$LT/vae/ltx-2.5-video-vae-conv-bf16.safetensors"                          "$COMFY_DIR/models/vae"
  link_one "$LT/vae/ltx-2.5-audio-vae-bf16.safetensors"                               "$COMFY_DIR/models/vae"
  link_one "$LT/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"                "$COMFY_DIR/models/loras"
  link_one "$LT/model_patches/ltx-2.5-duration-head-bf16.safetensors"                 "$COMFY_DIR/models/model_patches"
  mkdir -p "$COMFY_DIR/models/latent_upscale_models"
  link_one "$LT/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" "$COMFY_DIR/models/latent_upscale_models"

  log "  已链接 $LINKED 个权重，缺失 $MISSING 个"
  (( MISSING == 0 )) || warn "  缺失项可用 ./setup.sh --download 拉取，或放到 \$WORK_MODELS"
}

# ------------------------------------------------------- 5. 下载（可选） -----
step_download() {
  (( DO_DOWNLOAD )) || { log "5/6 跳过模型下载（--download 才下载）"; return; }
  log "5/6 下载缺失模型"

  if [[ ! -f "$HOME/.cache/huggingface/token" ]] && ! have hf; then
    warn "  建议先 hf auth login（LTX-2.5 是 gated 仓库）"
  fi
  local HF="$COMFY_ENV/bin/hf"
  [[ -x "$HF" ]] || HF="hf"

  # FLUX.2（非 gated）
  if [[ ! -f "$PROJECT_DIR/models/flux2/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors" ]]; then
    log "  下载 FLUX.2-dev（~57GB）"
    # 注意：hf download 的 --include 只吃一个 glob，逐个下
    for pat in "split_files/diffusion_models/*" "split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors" \
               "split_files/vae/*" "split_files/loras/Flux2TurboComfyv2.safetensors"; do
      "$HF" download Comfy-Org/flux2-dev --include "$pat" --local-dir models/flux2
    done
  else
    log "  FLUX.2 已存在，跳过"
  fi

  # LTX-2.5（gated，~188GB）
  if [[ ! -d "$WORK_MODELS/ltx-2.5/diffusion_models" ]]; then
    log "  下载 Lightricks/LTX-2.5（~188GB，需已接受许可）"
    "$HF" download Lightricks/LTX-2.5 --include "*" --local-dir "$WORK_MODELS/ltx-2.5" \
      || warn "  LTX-2.5 下载失败（大概率是未登录/未接受许可）"
  else
    log "  LTX-2.5 已存在，跳过"
  fi

  # whisper-small（抽检转写）
  if [[ ! -d "$WORK_MODELS/whisper-small" ]]; then
    log "  下载 whisper-small（~3.7GB）"
    "$HF" download openai/whisper-small --local-dir "$WORK_MODELS/whisper-small"
  else
    log "  whisper-small 已存在，跳过"
  fi

  note_minimax_h3
}

note_minimax_h3() {
  if [[ -d "$WORK_MODELS/MiniMax-H3/transformer" ]]; then
    log "  MiniMax-H3 已在 $WORK_MODELS/MiniMax-H3"
  else
    warn "  MiniMax-H3 缺失：ref2va 需 $WORK_MODELS/MiniMax-H3（约 465GB）"
    warn "    hf download MiniMaxAI/MiniMax-H3 --include 'transformer/*' --local-dir $WORK_MODELS/MiniMax-H3"
    warn "    hf download MiniMaxAI/MiniMax-H3 --include 'transformer_ref/*' --local-dir $WORK_MODELS/MiniMax-H3"
    warn "    （vae/ audio_vae/ text_encoder/ tokenizer/ processor/ scheduler/ audio_scheduler/ 也要）"
  fi
}

# ----------------------------------------------------------- 6. 自检 ---------
step_verify() {
  (( DO_VERIFY )) || { log "6/6 跳过自检（--no-verify）"; return; }
  log "6/6 自检"
  local PY="$COMFY_ENV/bin/python"
  "$PY" - <<'EOF' || true
import sys, torch, transformers
print(f"  env/comfy python {sys.version.split()[0]} | torch {torch.__version__} | transformers {transformers.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}" + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
EOF
  if (( WITH_H3 )) && [[ -x "$H3_ENV/bin/python" ]]; then
    "$H3_ENV/bin/python" -c "import torch,diffusers,transformers;print(f'  env/h3    torch {torch.__version__} | diffusers {diffusers.__version__}')" || true
  fi
  log "  模型就位情况："
  for p in \
    "FLUX.2 diffusion|$PROJECT_DIR/models/flux2/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors" \
    "FLUX.2 text enc|$PROJECT_DIR/models/flux2/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors" \
    "LTX-2.5 distill|$WORK_MODELS/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
    "LTX-2.5 gemma4 |$WORK_MODELS/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    "LTX-2.5 convVAE|$WORK_MODELS/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
    "MiniMax-H3 ref|$WORK_MODELS/MiniMax-H3/transformer_ref" \
    "whisper-small  |$WORK_MODELS/whisper-small" ; do
    local name="${p%%|*}" path="${p#*|}"
    if [[ -e "$path" ]]; then printf '    \033[32m✓\033[0m %s\n' "$name"; else printf '    \033[31m✗\033[0m %s  (%s)\n' "$name" "$path"; fi
  done
  log "自检完成。启动： ./run.sh serve"
}

step_prereqs
step_comfy_env
step_h3_env
step_link_models
step_download
step_verify
log "全部完成 ✅"
