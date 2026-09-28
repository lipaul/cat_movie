#!/usr/bin/env bash
# =============================================================================
# run.sh — 常用命令入口（配合 setup.sh）
#
#   ./run.sh serve                 启动 ComfyUI（后台，日志 /tmp/comfy.log）
#   ./run.sh stop                  停掉 ComfyUI（按 PID，安全）
#   ./run.sh status                GPU / 进程 / 队列
#   ./run.sh verify                打印环境与模型就位情况
#
#   ./run.sh t2i     <args...>     FLUX.2 文生图       → h3_ref2va/comfy_t2i.py
#   ./run.sh keyframes [names...]  批量关键帧           → h3_ref2va/gen_keyframes.py
#   ./run.sh ltx     <args...>     LTX-2.5 图生视频     → h3_ref2va/comfy_ltx_i2v.py
#   ./run.sh flf2v   <args...>     LTX-2.5 首尾帧(接续) → h3_ref2va/comfy_ltx_flf2v.py
#   ./run.sh shots   [names...]    批量逐镜动画         → h3_ref2va/gen_shots.py
#   ./run.sh subs                  重建字幕(语音对齐)   → h3_ref2va/gen_subs.py
#   ./run.sh post                  拼接+BGM+字幕+调色   → h3_ref2va/post_production.py
#   ./run.sh inspect <file> [...]  抽帧/转写抽检        → h3_ref2va/inspect_av.py
#   ./run.sh ref2va  <args...>     H3 ref2va（env/h3）  → h3_ref2va/diffusers_ref2va.py
#
# 环境变量：COMFY_PORT=8188
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

COMFY_PY="$PROJECT_DIR/env/comfy/bin/python"
H3_PY="$PROJECT_DIR/env/h3/bin/python"
COMFY_PORT="${COMFY_PORT:-8188}"
COMFY_LOG="${COMFY_LOG:-/tmp/comfy.log}"

log()  { printf '\033[1;36m[run]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[err ]\033[0m %s\n' "$*" >&2; exit 1; }

comfy_pid() { ps -eo pid,cmd | awk '/[m]ain\.py --listen/{print $1}' | head -1; }

need_comfy_py() { [[ -x "$COMFY_PY" ]] || die "缺少 $COMFY_PY，先跑 ./setup.sh"; }
need_h3_py()    { [[ -x "$H3_PY" ]]    || die "缺少 $H3_PY，先跑 ./setup.sh"; }

cmd_serve() {
  need_comfy_py
  local pid; pid="$(comfy_pid || true)"
  if [[ -n "$pid" ]]; then log "ComfyUI 已在运行（pid $pid）"; return; fi
  log "启动 ComfyUI (127.0.0.1:$COMFY_PORT) → $COMFY_LOG"
  nohup "$COMFY_PY" ComfyUI/main.py --listen 127.0.0.1 --port "$COMFY_PORT" >"$COMFY_LOG" 2>&1 &
  for _ in $(seq 1 90); do
    curl -fs "http://127.0.0.1:$COMFY_PORT/object_info" >/dev/null 2>&1 && { log "就绪"; return; }
    sleep 2
  done
  die "ComfyUI 启动超时，看 $COMFY_LOG"
}

cmd_stop() {
  local pid; pid="$(comfy_pid || true)"
  if [[ -z "$pid" ]]; then log "ComfyUI 未在运行"; return; fi
  kill "$pid" && log "已停止 pid $pid" || warn "kill 失败"
}

cmd_status() {
  echo "--- GPU ---";       nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  local pid; pid="$(comfy_pid || true)"
  echo "--- ComfyUI ---";   [[ -n "$pid" ]] && echo "running pid=$pid" || echo "stopped"
  if [[ -n "$pid" ]]; then
    curl -fs "http://127.0.0.1:$COMFY_PORT/queue" 2>/dev/null | \
      "$COMFY_PY" -c "import sys,json;d=json.load(sys.stdin);print('queue running',len(d.get('queue_running',[])),'pending',len(d.get('queue_pending',[])))" 2>/dev/null || true
  fi
}

cmd_verify() {
  bash "$PROJECT_DIR/setup.sh" --no-h3 --no-download --no-link --verify 2>/dev/null \
    || warn "（setup.sh 自检不可用，改为直接列版本）"
  [[ -x "$COMFY_PY" ]] && "$COMFY_PY" -c "import torch,transformers;print('env/comfy torch',torch.__version__,'| transformers',transformers.__version__)" || true
  [[ -x "$H3_PY" ]]    && "$H3_PY"    -c "import torch,diffusers;print('env/h3    torch',torch.__version__,'| diffusers',diffusers.__version__)" || true
}

cmd_t2i()       { need_comfy_py; "$COMFY_PY" h3_ref2va/comfy_t2i.py "$@"; }
cmd_keyframes() { need_comfy_py; "$COMFY_PY" h3_ref2va/gen_keyframes.py "$@"; }
cmd_ltx()       { need_comfy_py; "$COMFY_PY" h3_ref2va/comfy_ltx_i2v.py "$@"; }
cmd_flf2v()     { need_comfy_py; "$COMFY_PY" h3_ref2va/comfy_ltx_flf2v.py "$@"; }
cmd_shots()     { need_comfy_py; "$COMFY_PY" h3_ref2va/gen_shots.py "$@"; }
cmd_subs()      { need_comfy_py; "$COMFY_PY" h3_ref2va/gen_subs.py "$@"; }
cmd_post()      { need_comfy_py; "$COMFY_PY" h3_ref2va/post_production.py "$@"; }
cmd_inspect()   { need_comfy_py; "$COMFY_PY" h3_ref2va/inspect_av.py "$@"; }
cmd_ref2va()    { need_h3_py;    "$H3_PY"    h3_ref2va/diffusers_ref2va.py "$@"; }

usage() { sed -n '2,/^set -e/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

case "${1:-}" in
  serve)     shift; cmd_serve "$@" ;;
  stop)      shift; cmd_stop "$@" ;;
  status)    shift; cmd_status "$@" ;;
  verify)    shift; cmd_verify "$@" ;;
  t2i)       shift; cmd_t2i "$@" ;;
  keyframes) shift; cmd_keyframes "$@" ;;
  ltx)       shift; cmd_ltx "$@" ;;
  flf2v)     shift; cmd_flf2v "$@" ;;
  shots)     shift; cmd_shots "$@" ;;
  subs)      shift; cmd_subs "$@" ;;
  post)      shift; cmd_post "$@" ;;
  inspect)   shift; cmd_inspect "$@" ;;
  ref2va)    shift; cmd_ref2va "$@" ;;
  ""|-h|--help) usage ;;
  *) die "未知命令：$1（用 ./run.sh -h 看用法）" ;;
esac
