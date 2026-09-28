# 本地 AI 视频生成项目 · README

从零开始，用**本地模型**生成角色一致的短视频（含中文对白、音乐与双语字幕）。
本项目验证了两条视频路线（**LTX-2.5** 与 **MiniMax-H3**），并完成了一部 5 分钟短剧《嘟嘟的一天》。

---

## 0. 成果一览

| 产物 | 规格 | 备注 |
|---|---|---|
| `video/dudu_full.mp4` | 960×544, 300.7s (5 分钟) | 短剧《嘟嘟的一天》，15 镜，中英字幕 + BGM |
| `video/pig_moon.mp4` | 960×544, 14.4s | 写实小猪望月（H3 ref2va） |
| `video/peppa_moon.mp4` | 960×544, 14.4s | 佩奇粘土风望月（H3 ref2va） |
| `video/cat_speak_zh.mp4` | 960×544, 5.2s | 橘猫中文对白（H3 ref2va + 音频参考） |
| `refs/char/*.png` | 1024×1024 | 6 个角色参考图（FLUX.2） |
| `refs/story/*.png` | 1024×576 | 15 个镜头关键帧（FLUX.2 多参考） |

---

## 快速开始（一键）

```bash
git clone git@github.com:lipaul/cat_movie.git
cd cat_movie

./setup.sh            # 建 env/comfy + env/h3、接线模型、自检
./run.sh serve        # 后台启动 ComfyUI

# 出图 → 关键帧 → 出视频 → 后期
./run.sh t2i    --prompt "..." --width 1024 --height 576 --seed 5 --out refs/story/x.png
./run.sh ltx    --image refs/story/x.png --prompt "..." --frames 481 --out video/x.mp4
./run.sh shots               # 批量逐镜（按 gen_shots.py 的镜头表）
./run.sh post                # 拼接 + BGM + 中英字幕
./run.sh inspect video/x.mp4 --transcribe --lang zh
```

| 脚本 | 作用 |
|---|---|
| `setup.sh` | 从零建环境：`env/comfy`（ComfyUI+LTX）、`env/h3`（MiniMax-H3）、模型接线、自检 |
| `run.sh` | 命令入口：`serve / stop / status / verify / t2i / keyframes / ltx / shots / post / inspect / ref2va` |

**选项**：`./setup.sh --no-h3`（只建 ComfyUI/LTX）·`--download`（下载缺失模型，很大）·`--no-link`·`--no-verify`
**前置**：`uv`、`git`、`ffmpeg`（需含 `libass` 才能烧字幕）、NVIDIA GPU。
模型默认从 `$WORK_MODELS`（默认 `~/work/models`）**软链**，不重复下载。

---

## 1. 硬件与环境（实测）

| 项 | 值 |
|---|---|
| GPU | **NVIDIA RTX 6000 Ada Generation, 48GB**（SM 8.9 / Ada） |
| 驱动 / CUDA | 580.178.04 / 13.0 |
| CPU / RAM | 24 核 / 183GB |
| 磁盘 | 1.8T（生成时长期占用 ~1.3T，建议预留 ≥300GB） |
| Python | 3.12.13（用 `uv` 管理） |
| ComfyUI 环境 | `env/comfy`：torch **2.11.0+cu128**、transformers **5.17.0**、soundfile 0.14.0 |
| ComfyUI commit | `1568e6c` |
| 关键限制 | Ada 不支持 **nvfp4**、**FlashAttention-4**（Blackwell 专属）；`natten` / `ltx_kernels` 未安装 |

> **Ada 注意**：LTX-2.5 的官方 `int8-convrot` / `nvfp4` 权重分别依赖 `ltx_kernels` 与 Blackwell，本机不可用 → 走 **bf16 + fp8 weight_dtype**；官方非 conv 版 VAE 依赖 `natten` → 必须用 **`-conv` 版 VAE**。

---

## 2. 一次性准备

> 本章已由 **`./setup.sh`** 自动完成；下面是手动步骤，便于理解或按需定制。

### 2.1 模型

| 模型 | 路径 | 体积 | 来源 |
|---|---|---|---|
| FLUX.2-dev fp8 | `models/flux2/` | 56.6GB | `Comfy-Org/flux2-dev`（非 gated） |
| MiniMax-H3 | `~/work/models/MiniMax-H3` | 465GB | `MiniMaxAI/MiniMax-H3` |
| LTX-2.5 | `~/work/models/ltx-2.5` | 188GB | Lightricks（ComfyUI 格式） |
| whisper-small | `~/work/models/whisper-small` | 3.7GB | 抽检转写用 |

FLUX.2 下载（实测 ~13 分钟）：

```bash
cd /home/acm/paul_nv/cat
HF=env/comfy/bin/hf          # 或 ~/.local/bin/uv run hf
$HF download Comfy-Org/flux2-dev --include "split_files/diffusion_models/*" --local-dir models/flux2
$HF download Comfy-Org/flux2-dev --include "split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors" --local-dir models/flux2
$HF download Comfy-Org/flux2-dev --include "split_files/vae/*" --local-dir models/flux2
$HF download Comfy-Org/flux2-dev --include "split_files/loras/Flux2TurboComfyv2.safetensors" --local-dir models/flux2
```

> ⚠️ 坑：`hf download` 的 `--include` 只吃**一个**参数，多个 pattern 会被当成位置参数而忽略，导致漏下大文件。**一次一个 glob**。

### 2.2 ComfyUI（实测 ~12 分钟）

```bash
cd /home/acm/paul_nv/cat
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 env/comfy
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ComfyUI
uv pip install --python env/comfy/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python env/comfy/bin/python -r ComfyUI/requirements.txt
uv pip install --python env/comfy/bin/python soundfile     # 抽检脚本需要
```

### 2.3 软链模型到 ComfyUI

```bash
cd /home/acm/paul_nv/cat
# FLUX.2
ln -sf $PWD/models/flux2/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors ComfyUI/models/diffusion_models/
ln -sf $PWD/models/flux2/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors ComfyUI/models/text_encoders/
ln -sf $PWD/models/flux2/split_files/vae/flux2-vae.safetensors ComfyUI/models/vae/
ln -sf $PWD/models/flux2/split_files/loras/Flux2TurboComfyv2.safetensors ComfyUI/models/loras/
# LTX-2.5（注意用 -conv VAE）
L=$HOME/work/models/ltx-2.5
ln -sf $L/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors ComfyUI/models/diffusion_models/
ln -sf $L/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors ComfyUI/models/text_encoders/
ln -sf $L/vae/ltx-2.5-video-vae-conv-bf16.safetensors ComfyUI/models/vae/
ln -sf $L/vae/ltx-2.5-audio-vae-bf16.safetensors ComfyUI/models/vae/
```

### 2.4 启动 ComfyUI

```bash
cd /home/acm/paul_nv/cat
env/comfy/bin/python ComfyUI/main.py --listen 127.0.0.1 --port 8188 > /tmp/comfy.log 2>&1 &
for i in $(seq 1 90); do curl -s http://127.0.0.1:8188/object_info >/dev/null && break; sleep 2; done
```

---

## 3. 生成流程总览

```
① 角色参考图   FLUX.2 (ComfyUI)      refs/char/*.png
② 镜头关键帧   FLUX.2 多参考          refs/story/*.png
③ 逐镜动画     LTX-2.5 i2v / H3      video/shots/*.mp4
④ 后期         拼接 + BGM + 字幕      video/*_full.mp4
⑤ 抽检         抽帧 + Whisper 转写
```

### 两条视频路线对比

| | **LTX-2.5** | **MiniMax-H3** |
|---|---|---|
| 速度（960×544） | **~8s/秒视频**（20s 片段 162s） | ~10s/**帧**（20s 片段 ≈ 2 小时） |
| 单片段上限 | ≥20s | 14.375s（345 帧） |
| 中文对白 | ✅ 支持 | ✅ 更强、口型更好 |
| 角色一致性 | 靠关键帧锁定 | ref2va 多参考更强 |
| 显存 | ~42–44GB | ~30–48GB（int8 + offload） |
| 适用 | 大批量、动作/空镜、长片 | hero 镜、对白密集镜 |

**结论：长片优先 LTX-2.5；H3 只留给最关键的少数镜头。**

---

## 4. 逐步操作 + 实测耗时/资源

### ① 图像：FLUX.2 出角色参考图与关键帧

```bash
cd /home/acm/paul_nv/cat
# 单张（支持多角色参考，可重复 --ref）
env/comfy/bin/python h3_ref2va/comfy_t2i.py \
  --prompt "photorealistic ... , no text, no watermark" \
  --width 1024 --height 576 --steps 20 --seed 5 \
  --ref refs/char/dudu.png --ref refs/char/mama.png \
  --prefix kf_01_wakeup --out refs/story/01_wakeup.png

# 批量关键帧
env/comfy/bin/python h3_ref2va/gen_keyframes.py            # 全部
env/comfy/bin/python h3_ref2va/gen_keyframes.py 13_eat     # 指定镜
```

| 任务 | 规格 | 实测 |
|---|---|---|
| 单张（无参考） | 1024×1024, 20 步 | **48–60s** |
| 单张（1–2 参考） | 1024×576, 20 步 | **30–55s** |
| 单张（5 参考） | 1024×576, 20 步 | **~130s** |
| 6 张角色图 | 1024×1024 | **~5 分钟** |
| 15 张关键帧 | 1024×576 | **~40 分钟** |

资源：显存峰值 ~35–45GB（FLUX.2 常驻，跑视频前**必须停掉 ComfyUI**）。

### ② 视频路线 A：LTX-2.5（推荐）

```bash
# 单镜（image → ~20s 视频），481 帧 = 20.04s
env/comfy/bin/python h3_ref2va/comfy_ltx_i2v.py \
  --image refs/story/01_wakeup.png \
  --prompt "the piglet sleeps on its back ... no text, no subtitles, no watermark" \
  --width 960 --height 544 --frames 481 --fps 24 --seed 42 \
  --out video/shots/01_wakeup.mp4

# 批量（含台词/旁白）
env/comfy/bin/python h3_ref2va/gen_shots.py
```

实测（960×544，8 sigmas，`euler_ancestral`，fp8）：

| 帧数 | 时长 | 耗时 | 显存峰值 |
|---|---|---|---|
| 97 | 4.0s | **50s** | ~42GB |
| 193 | 8.0s | **60s** | ~42GB |
| 241 | 10.0s | 69s | ~42GB |
| 345 | 14.4s | 105s | ~43GB |
| **481** | **20.0s** | **162s** | **~44GB** |

- 15 镜 × 20s：**总计 ~42 分钟**
- 中文台词写在 prompt 里即可（如 `说话内容`），LTX 原生生成语音 + 口型
- 关键节点：`UNETLoader(bf16+fp8_e4m3fn) → CLIPLoader(type=ltxv) → LTXVConditioning → LTXVImgToVideoInplace → LTXVConcatAVLatent → LTXVDualCFGGuider + ManualSigmas → SamplerCustomAdvanced → VAEDecodeTiled + LTXVAudioVAEDecode`

### ③ 视频路线 B：MiniMax-H3（高质量对白；**venv 已丢失，需重建**）

H3 依赖 diffusers 的 modular pipeline（`MiniMaxH3ModularPipeline`）+ torchao int8 + group offload：

```bash
# 重建环境（约 10 分钟；torch 用 cu130）
uv venv --python 3.12 env/h3
uv pip install --python env/h3/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python env/h3/bin/python diffusers transformers accelerate torchao einops soundfile av
# 还需 ~/work/models/MiniMax-H3/transformer_ref （diffusers 格式，66GB，用于 ref2va）
```

```bash
# ref2va（参考图 → 视频，≤9 图 + 可选音频参考）
env/h3/bin/python h3_ref2va/diffusers_ref2va.py \
  --prompt "$(cat prompts/pig_moon.txt)" \
  --ref refs/pig/P1_sitting.png --ref refs/pig/P2_family.png \
  --height 544 --width 960 --num-frames 345 --steps 25 --seed 42 \
  --output video/pig_moon.mp4
```

实测（960×544，25 步，int8 + offload）：

| 任务 | 帧数/时长 | 参考 | 生成 | 含加载总耗时 | 显存峰值 |
|---|---|---|---|---|---|
| t2va 20 步 | 124 / 5.2s | 0 | — | **438s** | ~26GB |
| ref2va 20 步 | 124 / 5.2s | 1 图 | 314s + 加载 | **812s** | ~33GB |
| ref2va 25 步 | 158 / 6.6s | 3 图 + 1 音频 | 1373s | **1608s** | — |
| ref2va 25 步 | 345 / 14.4s | 2 图 | 2776s | **2966s** | 48.0/49.1GB |
| ref2va 25 步 | 345 / 14.4s | 3 图 | 3502s | **3699s** | 38GB |

- 加载固定开销 **175–186s**
- 单帧约 **10s**；345 帧单条 **50–62 分钟**
- 帧数必须为 `17n+5`，时长 ≤15s → **最大 345 帧 = 14.375s**（362 帧 = 15.083s 会被拒）
- 3 张参考图 345 帧时序列 ~78k tokens，**主要瓶颈是时间不是显存**

### ④ 后期：拼接 + BGM + 双语字幕

```bash
env/comfy/bin/python h3_ref2va/post_production.py
```

- concat 15 镜（`-c copy`）
- numpy 合成八音盒 BGM（~300s）
- ASS 中英双语字幕 + 标题卡（Noto Sans CJK）
- 最终 `libx264 crf18` 编码

实测：**~1–2 分钟**（5 分钟成片）

### ⑤ 抽检

```bash
# 抽帧
ffmpeg -ss 25 -i video/dudu_full.mp4 -frames:v 1 /tmp/f.jpg
# 转写（默认 CPU，避免与 ComfyUI 抢显存）
env/comfy/bin/python h3_ref2va/inspect_av.py video/shots/02_wakeup_talk.mp4 --transcribe --lang zh
```

---

## 5. 端到端时间预算

| 阶段 | 耗时（实测） |
|---|---|
| 一次性：ComfyUI + FLUX.2 下载 | ~25 分钟 |
| 一次性：LTX-2.5 已就位（无需下载） | — |
| 6 角色图 + 15 关键帧 | ~45 分钟 |
| **15 镜 LTX 动画（5 分钟成片）** | **~42 分钟** |
| 后期 | ~2 分钟 |
| **合计** | **≈ 1.5 小时**（纯 H3 同长度需 ~20 小时） |

---

## 6. 坑与注意事项

1. **显存互斥**：ComfyUI 常驻 ~41GB，跑 H3/其他大模型前必须停掉。安全停法：
   ```bash
   PID=$(ps -eo pid,cmd | awk '/[m]ain\.py --listen/{print $1}'); [ -n "$PID" ] && kill $PID
   ```
2. **`pkill -f "ComfyUI/main.py"` 会杀掉自己**（匹配到执行它的 shell 命令行）。
3. **`hf download --include` 只吃一个 pattern**，多个会漏文件。
4. **Ada 适配**：LTX-2.5 必须用 bf16+fp8 与 `-conv` VAE；`int8-convrot`/`nvfp4` 及非 conv VAE 在本机不可用。
5. **H3 帧数**必须 `17n+5` 且 ≤345（14.375s）。
6. **ref2va 与首帧互斥**：参考图路线不能同时给 `first_frame`；可用 prompt 把某张参考图声明为「[Shot 1] 开场锚点」。
7. **多角色易漏**：prompt 里明确「有几个角色、各自位置」；参考图里角色太小/太偏容易被忽略。
8. **长序列慢**：LTX 20s 片段 ~162s；H3 345 帧 ~50–62 分钟。
9. **对白验证**：Whisper 会把「圆」听成「远」（近音），属正常。

---

## 7. 目录结构

```
/home/acm/paul_nv/cat/
├── README.md                    # 本文件
├── setup.sh                     # 一键建环境（env/comfy + env/h3 + 模型接线 + 自检）
├── run.sh                       # 命令入口（serve/stop/status/t2i/ltx/shots/post/inspect/ref2va）
├── ComfyUI/                     # ComfyUI（git clone）
├── env/comfy/                   # ComfyUI venv（torch cu128）
├── models/flux2/                # FLUX.2 fp8 权重
├── h3_ref2va/                   # 本项目脚本
│   ├── comfy_t2i.py             # FLUX.2 文生图（多参考）
│   ├── gen_keyframes.py         # 15 镜关键帧
│   ├── comfy_ltx_i2v.py         # LTX-2.5 i2v（Ada 适配）
│   ├── gen_shots.py             # 逐镜动画（含台词）
│   ├── post_production.py       # 拼接+BGM+字幕
│   ├── diffusers_ref2va.py      # H3 ref2va（参考图→视频）
│   ├── diffusers_t2va.py        # H3 t2va 验证
│   ├── inspect_av.py            # 抽帧/转写抽检
│   └── (原生 ref2va 端口: ref2va_layout.py / verify_layout.py / oracle_layout.py)
├── refs/
│   ├── char/                    # 角色参考图
│   ├── story/                   # 镜头关键帧
│   ├── hero/ sheet/ peppa/ pig/ voice/
├── prompts/                     # H3 六段式提示词
├── video/                       # 产物 + shots/ + post/
└── docs/status.md               # 详细制作日志与实测数据
```

---

## 8. 常用命令速查

```bash
# 起服务
env/comfy/bin/python ComfyUI/main.py --listen 127.0.0.1 --port 8188 &

# 出图
env/comfy/bin/python h3_ref2va/comfy_t2i.py --prompt "..." --width 1024 --height 576 --seed 5 --prefix x --out refs/story/x.png

# 出视频（LTX，20s）
env/comfy/bin/python h3_ref2va/comfy_ltx_i2v.py --image refs/story/x.png --prompt "..." --frames 481 --out video/x.mp4

# 后期
env/comfy/bin/python h3_ref2va/post_production.py

# 抽检
env/comfy/bin/python h3_ref2va/inspect_av.py video/x.mp4 --transcribe --lang zh
```

---

## 9. 分镜连贯性（跨镜头一致）

跨镜头一致性拆成 5 类，各有对应手段：

| 类型 | 手段 | 状态 |
|---|---|---|
| **角色一致** | 固定角色参考图 + FLUX.2 多参考出关键帧 | ✅ `refs/char/` |
| **场景/道具一致** | 每个地点一张场景定妆图 + prompt 固定道具清单 | 建议补（见下） |
| **光线/色彩一致** | 固定光照词 + 后期按镜头亮度向全片均值微调（±0.02） | ✅ `post_production.py` |
| **动作/轴线连续** | **首尾帧续接**（`comfy_ltx_flf2v.py`）+ 导演规则 | ✅ 工具已备 |
| **音频连续** | 全片单轨 BGM + 字幕按真实语音对齐 | ✅ `gen_subs.py` |

### 关键做法

1. **关键帧驱动**：每镜先用 FLUX.2 出静帧，再 i2v 动画 —— 构图 100% 由关键帧锁定，模型不自由发挥
2. **要接续的镜头用首尾帧**：
   ```bash
   ./run.sh flf2v --first <上一镜尾帧.png> --last <本镜目标帧.png> \
                  --prompt "..." --frames 481 --strength 0.7 --out video/seg.mp4
   ```
   （`LTXVAddGuide(frame_idx=0/-1)` → 采样 → `LTXVCropGuides` 裁掉 guide 帧）
   不需要接续的镜头，**硬切**即可 —— 短剧天然是多镜头硬切
3. **导演规则**（写进分镜表，零成本）：180° 轴线一致 · 视线匹配 · 景别递进（远→中→近）· 切点落在动作/台词边界
4. **字幕时序**：`gen_subs.py` 用**能量 VAD** 检测真实语音段，再把剧本台词映射上去
   （不用 Whisper 定时 —— 它会在音乐/环境音上幻觉出整段"语音"）

### ⚠️ 坑：LTX 会把 prompt 里的中文"画成字幕"

只要 prompt 里出现中文台词，LTX-2.5 就有很大概率把它**当字幕烧进画面**（且是乱码，如"炖乎平的主玉来和和片"），与后期字幕叠加成"双字幕"。

实测结论（960×544 / 481 帧，同一镜反复验证）：

| 做法 | 画面 | 语音 |
|---|---|---|
| 中文台词 + 强化负面（含 `subtitles, chinese characters…`） | ❌ 仍出字 | ✅ 正确 |
| **负面里再写入该句中文** | ❌ 仍出字 | ✅ 正确 |
| 去掉冒号/说话人标签，仅留中文 | ❌ 仍出字 | ✅ 正确 |
| 换 seed（43/44/45/46） | ❌ 四连出字 | ✅ 正确 |
| **prompt 完全不含中文** | ✅ 干净 | ❌ 说胡话 |

即：**要正确的中文语音就必须把中文写进 prompt，也就难免被烧成画面文字**。

当前采用**确定性兜底**：

- `post_production.py` 默认在画面底部叠一条深色渐变**字幕栏**（`--no-band` 可关），用来盖住 LTX 烧进去的文字，同时给双语字幕一个稳定的可读背景
- 更彻底的方案（未实施）：用 `LTXVReferenceAudio` 传入参考语音，让 prompt 完全不含中文 → 干净画面 + 指定台词（代价是每镜两遍生成）

判断某镜是否被烧字：抽帧看**下三分之一**有没有非预期文字。

---

*所有耗时/显存均为本机（RTX 6000 Ada 48GB）实测；数据来源见 `docs/status.md`。*
