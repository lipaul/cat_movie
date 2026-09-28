# 项目状态 — 拟人胖橘猫（角色一致性图像 + 图生视频）

项目目录：`/home/acm/paul_nv/cat`
更新：2026-09-24

## ✅ 端到端已跑通

```
FLUX.2-dev (ComfyUI)  →  英雄图 + 8 视角设定集  →  MiniMax-H3 ref2va  →  角色一致视频+音频
```

最终产物：

| 产物 | 路径 |
|---|---|
| 英雄图（正面全身） | `refs/hero/cat_hero.png` |
| 设定集 8 视角 | `refs/sheet/cat_{side,threequarter,head,back,tpose,sitting,happy,surprised}.png` |
| 设定集拼图 | `refs/sheet/contact_sheet.png` |
| **最终视频**（960x544, 124帧, 5.17s, h264+aac） | `video/cat_ref2va_final.mp4` |
| **中文对白视频**（同上规格，H3 原生生成中文语音+口型） | `video/cat_speak_zh.mp4` |
| **多轮中文对白 + 音色参考**（158帧/6.58s，3 分镜） | `video/cat_speak_zh_v2.mp4` |

视频内容：橘猫站立、抬爪挥手、张嘴微笑、微转身——角色与设定集一致（橘色虎斑、胖、圆脸、琥珀眼）。

中文对白视频：`[Shot 1]` 抬爪挥手说「大家好，我是胖橘！」，3 秒处切近景说「今天也要开开心心的哦！」。Whisper 转写验证：**「大家好 我是胖菊今天也要开开心心的哦」**（"胖菊"为同音字，发音正确）——H3 原生中文语音与口型同步成功。

多轮对白 v2（`prompts/cat_speak_zh_v2.txt`）：3 分镜 / 3 句 —— 「嗨，又见面啦！」→「我还是那只胖橘猫。」→「记得每天都要开心哦！」；并用 `<Audio 1>` = `refs/voice/cat_voice_zh.wav`（从 v1 抽取的猫声）作**音色参考**，保持两版音色一致。Whisper 转写：**「嗨！又见面啦！我还是那只胖橘猫！记得每天都要开心哦！」**（完全正确）。用时 1608s（加载 228s + 生成 1373s）。

音频参考用法：`diffusers_ref2va.py --audio-ref <wav/mp4>`（作为 `<Audio N>` timbre reference，≤3 段，须与图像/视频参考搭配）。

对白 prompt 格式（官方 ref2va 规范）：六段式 `subject_definitions` / `summary` / `retention_analysis` / `detailed_description` / `overall_soundscape` / `non_diegetic_music`；对白用 `<d>[Chinese] 台词</d>`，说话人用 `(S1)`；分镜 `[Shot 1]` / `[Shot 2] At 00:03.000, ...`。模板见 `prompts/cat_speak_zh.txt`。

---

## 一、当前可用（已实测）

### 1. 视频：本地 MiniMax-H3 `ref2va`（参考图 → 角色一致视频 + 音频）✅

**关键结论：原生 `h3_t2va.py` 是坏的，官方 diffusers pipeline 是好的。**

| 运行时 | 结果 |
|---|---|
| `paul_arc/test/h3_nv/h3_t2va.py`（自实现 int8 DiT） | ❌ 纯噪声（连用户此前的 `t2va_768p.mp4` 也是噪声） |
| 官方 `diffusers` `MiniMaxH3ModularPipeline`（int8 + group offload） | ✅ 清晰、角色一致、带音频 |

工作命令：

```bash
cd /home/acm/paul_nv/cat
VENV=/home/acm/paul_arc/test/h3_nv/.venv
$VENV/bin/python h3_ref2va/diffusers_ref2va.py \
  --prompt "The anthropomorphic orange tabby cat turns its head toward the camera and smiles. <Picture 1> is the character reference. Cinematic lighting, detailed orange fur." \
  --ref refs/test_ref.png \
  --height 544 --width 960 --num-frames 124 --steps 20 --seed 42 \
  --output video/diff_ref2va_test.mp4
```

产物：`video/diff_ref2va_test.mp4`（960x544, 124 帧, h264+aac）。输出是写实橘猫，随参考图保持一致，会转头/张嘴。

- 组件：`transformer_ref/`（66GB，已下载）、`vae/`、`audio_vae/`、`text_encoder/`、`tokenizer/`、`processor/`、`scheduler/`。
- 配方：torchao **int8 weight-only** + transformer **block_level group offload** + text encoder **leaf_level offload**；权重驻留 host RAM。
- 实测：t2va 544x960/20 步 = 438s；ref2va 1 张参考图/20 步 = 629s（含 181s 加载）。

### 2. 图像：FLUX.2-dev fp8 ✅ 已下载，ComfyUI 安装中

- `models/flux2/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors`（35.45GB）
- `models/flux2/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors`（18.03GB）
- `models/flux2/split_files/vae/flux2-vae.safetensors`（0.33GB）
- `models/flux2/split_files/loras/Flux2TurboComfyv2.safetensors`（2.76GB，加速）

---

## 二、代码资产（`h3_ref2va/`）

| 文件 | 用途 | 状态 |
|---|---|---|
| `ref2va_layout.py` | 原生 ref2va 布局/调度端口 | ✅ 与 diffusers **逐位对齐**（`verify_layout.py` 全 PASS） |
| `oracle_layout.py` | 从 diffusers 纯函数生成 golden | ✅ |
| `verify_layout.py` | 布局对拍 | ✅ ALL PASS |
| `ref2va_conditioner.py` | 图像归一化(短边2048) + 呈现 + VAE 编码 | ✅ 实测通过 |
| `h3_ref2va.py` | 原生 ref2va 全流程 | ⚠️ 可跑但依赖**坏掉的原生 DiT**，出噪声 → 弃用 |
| `diffusers_t2va.py` | 官方 t2va 验证 | ✅ 出图 |
| `diffusers_ref2va.py` | **官方 ref2va（当前主路径）** | ✅ 出片 |

> 原生 layout 端口的价值：它证明了打包契约，且 `ref2va_conditioner.py` 的归一化/呈现逻辑可复用；但执行路径改为 diffusers。

---

## 三、模型资产（`~/work/models`，661GB+62GB）

| 路径 | 大小 | 说明 |
|---|---|---|
| `MiniMax-H3/FL2VA/`、`Ref2VA/` | 各 144GB | 原始格式（原生代码用，已弃用） |
| `MiniMax-H3/transformer/`、`transformer_ref/` | 各 66GB | **diffusers 格式（当前使用）** |
| `MiniMax-H3/text_encoder/` | 66.7GB | Qwen3-VL-32B |
| `MiniMax-H3/vae/`、`audio_vae/` | 10.4/0.6GB | diffusers 格式 |
| `ltx-2.5/` | 188GB | 备选视频路线 |
| `Qwen3.8-27B(-MixedInt4)` | 52/20GB | LLM/VLM |

---

## 四、复现步骤（端到端）

```bash
cd /home/acm/paul_nv/cat
HV=/home/acm/paul_arc/test/h3_nv/.venv      # H3/diffusers 环境
CV=/home/acm/paul_nv/cat/env/comfy          # ComfyUI 环境

# 1) 起 ComfyUI（图像阶段）
$CV/bin/python ComfyUI/main.py --listen 127.0.0.1 --port 8188 &

# 2) 英雄图
$CV/bin/python h3_ref2va/comfy_t2i.py --prompt "<角色提示词>" \
  --width 1024 --height 1024 --steps 20 --seed 42 --prefix cat_hero --out refs/hero/cat_hero.png

# 3) 设定集（用英雄图作多参考）
$CV/bin/python h3_ref2va/comfy_t2i.py --ref refs/hero/cat_hero.png \
  --prompt "<角色提示词>, full body, side profile view, facing right" \
  --width 1024 --height 1024 --steps 20 --seed 42 --prefix sheet_side --out refs/sheet/cat_side.png
# …其余视角同理

# 4) ⚠️ 跑视频前必须停掉 ComfyUI 释放显存
pkill -f "ComfyUI/main.py"; sleep 5

# 5) 角色一致视频（本地 diffusers MiniMax-H3 ref2va）
$HV/bin/python h3_ref2va/diffusers_ref2va.py \
  --prompt "… <Picture 1> … <Picture 2> … <Picture 3> are the character references …" \
  --ref refs/hero/cat_hero.png --ref refs/sheet/cat_side.png --ref refs/sheet/cat_head.png \
  --height 544 --width 960 --num-frames 124 --steps 25 --seed 42 \
  --output video/cat_ref2va_final.mp4
```

## 五、经验教训

- **先验证基线**：动手扩展前应先确认原生 t2va 能出正常视频。之前的 `t2va_768p.mp4` 是噪声，被当成了"已跑通"。
- 官方 diffusers 是最可靠的参考实现，且本机组件齐全（除 `transformer_ref/`，已补下）。
- int8 + offload 配方在 48GB 卡上工作良好。
- **显存互斥**：ComfyUI 常驻约 41GB，跑 H3 前必须停掉 ComfyUI，否则 CUDA OOM。
- **`pkill -f` 会误杀自己**：`pkill -f "ComfyUI/main.py"` 会匹配到执行它的那个 shell 自己的命令行（脚本里含该字符串），导致整条命令被杀死。安全做法：先单独用 `ps -eo pid,cmd | grep "[m]ain.py --listen"` 找到 PID 再 `kill <pid>`，或把 `pkill` 放在单独的一次调用里。
- 参考图短边 2048，每张约 4–6k conditioning 行，参考图越多越慢（3 张/25 步/124 帧 ≈ 22 分钟；158 帧 ≈ 30 分钟）。

---

# 附：小猪望月（14.4s 长视频）

同一套管线（FLUX.2 + H3 ref2va），验证了 **345 帧 / 14.375s** 的长视频上限。

## 产物

| 产物 | 路径 | 说明 |
|---|---|---|
| 佩奇粘土版 | `video/peppa_moon.mp4` | 960×544，14.375s；卡通粘土，佩奇+苏西（**乔治未出现**） |
| 写实猪版 | `video/pig_moon.mp4` | 960×544，14.375s；**写实四足猪 + 拟人动作 + 布景**，猪一家三口 |
| 参考图（佩奇） | `refs/peppa/P1b_closeup.png`（首帧锚点）、`P2_charsheet.png` | |
| 参考图（写实猪） | `refs/pig/P1_sitting.png`（坐姿捧杯锚点）、`P1b_fourleg.png`（四足）、`P2_family.png`（一家三口） | |

写实猪版镜头：写实猪坐姿捧粉茶杯仰头问月（近景）→ 相机缓慢拉远、猪恢复四足 → 揭示猪妈妈与小猪一起望月（中景）。台词三句中文，Whisper 转写基本正确（「圆」被听成近音「远」）。用时 3699s（生成 3502s）。

## 经验

- **14.4s 是 H3 上限**：345 帧 = 14.375s；362 帧 = 15.083s 会被 pipeline 拒绝（`max_duration=15`）。
- **分辨率 vs 参考图数量**：960×544 + 2 参考图（S≈70k）峰值显存 48.0/49.1GB；+3 参考图（S≈78k）实测峰值降到 38GB 却更慢（138s/步），说明 offload 在起作用，**长序列主要瓶颈是时间而非显存**。
- **prompt 里指定 `<Picture N>` 为首帧锚点**可锁定开场构图（ref2va 不接收 `first_frame`，与 reference_image 互斥）。
- **多角色易漏**：佩奇版乔治始终未出现（参考图里他较小/偏右）。写实猪版改用「一家三口合影 + 明确站位」后三只都出现了。
- 长视频（345 帧）单条 **约 50–62 分钟**。

---

# 附二：LTX-2.5 spike（RTX 6000 Ada）

结论：**LTX-2.5 在本机完全可用，且快两个数量级——全片可以主要用它。**

## Ada 适配（4 处，已跑通）

| 官方模板 | 本机改法 | 原因 |
|---|---|---|
| `...-comfy-int8-convrot` transformer | `ltx-2.5-22b-distilled-transformer-bf16.safetensors` + `weight_dtype=fp8_e4m3fn` | `ltx_kernels` 未装；nvfp4 需 Blackwell |
| `gemma4-12b-...-convrot` TE | `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | 同上 |
| `ltx-2.5-video-vae-bf16` | **`ltx-2.5-video-vae-conv-bf16`** | 非 conv 的 DiffVAE 需要未装的 `natten` |
| `TextGenerateLTX2Prompt` | 跳过 | 缺 `gemma4_e2b_int8_convrot` |

节点图：`UNETLoader → CLIPLoader(ltxv) → CLIPTextEncode×2 → LTXVConditioning → LTXVPreprocess → EmptyLTXVLatentVideo → LTXVImgToVideoInplace → LTXVEmptyLatentAudio → LTXVConcatAVLatent → LTXVDualCFGGuider + ManualSigmas + KSamplerSelect(euler_ancestral) → SamplerCustomAdvanced → LTXVSeparateAVLatent → VAEDecodeTiled / LTXVAudioVAEDecode → CreateVideo → SaveVideo`。
脚本：`h3_ref2va/comfy_ltx_i2v.py`。

## 实测（960×544, 24fps, 8 sigmas, fp8）

| 帧数 | 时长 | 耗时 |
|---|---|---|
| 97 | 4.0s | **50s** |
| 193 | 8.0s | **60s** |
| 241 | 10.0s | 69s |
| 345 | 14.4s | 105s |
| **481** | **20.0s** | **162s** |

- 显存峰值 ~42–44GB（48GB 卡放得下，无 OOM）
- **支持中文对白**：转写「月亮月亮，你今天为什么这么远呀?」（同 H3 的近音误听），口型随语音动
- 画质：良好（比 H3 略软），20s 内一致性 stable

## 对 5 分钟短剧的意义

| 方案 | 估计总时长 |
|---|---|
| 纯 H3 | ~20 小时 |
| **纯 LTX-2.5**（15 × 20s） | **~40 分钟生成** + ~30 分钟关键帧 ≈ **1.5 小时** |
| 混合（H3 只做 2–4 个 hero 镜） | ~4–5 小时 |

→ 建议 **LTX-2.5 为主，H3 只留给最关键的少数镜头**。

---

# 附三：5 分钟短剧《嘟嘟的一天》（纯 LTX-2.5）

**成片：`video/dudu_full.mp4`**（960×544，**300.7s = 5 分钟**，h264+aac，66.5 MiB）

## 配置

| 项 | 值 |
|---|---|
| 剧本 | 《嘟嘟的一天》9 场戏，动物拟人日常 |
| 角色 | 嘟嘟（小猪）、妈妈猪、老师大鹅、咩咩（小羊）、汪汪（小狗）、嘎嘎（小鸭），自然写实外观 |
| 后端 | **纯 LTX-2.5**（i2v），15 镜 × 20.04s（481 帧）= 300s |
| 台词 | 中文对白 + 中文旁白，由 LTX 原生生成语音与口型 |
| 字幕 | 中英双语（ASS 烧入，Noto Sans CJK） |
| BGM | 后期 numpy 合成的八音盒摇篮曲，单轨混入（0.20 音量 + 结尾淡出） |
| 标题卡 | 首 4.5s「嘟嘟的一天 / Dodo's Day」 |

## 产物链

| 阶段 | 脚本 | 产物 |
|---|---|---|
| 角色参考图 ×6 | `comfy_t2i.py`（多参考） | `refs/char/*.png` |
| 关键帧 ×15 | `gen_keyframes.py` | `refs/story/*.png` |
| 逐镜动画 ×15 | `gen_shots.py`（LTX i2v） | `video/shots/*.mp4` |
| 拼接/BGM/字幕 | `post_production.py` | `video/dudu_full.mp4` |
| 抽检/转写 | `inspect_av.py` | — |

## 实测

- LTX-2.5 单镜 481 帧（20s）960×544：**~160–180s**（含 ComfyUI 每次重载模型）
- 15 镜总计生成约 **45 分钟**（对比纯 H3 同长度需 ~20 小时）
- 中文对白转写基本正确（如 02 镜「嘟嘟太阳晒屁股啦…」、05 镜「今天我们学习分享…」）
- 15 镜关键帧一致性良好，角色跨镜可辨

## 环境变更（重要）

制作期间 `/home/acm/paul_arc/` 被整体清空，**h3_nv venv 与 H3 参考脚本已丢失**。影响：
- 项目 `/home/acm/paul_nv/cat` 与 ComfyUI(`env/comfy`)、`~/work/models` 均完好
- **LTX-2.5 路线不受影响**（跑在 ComfyUI）
- 若要再用 H3，需要重建 venv；转写已改用 `env/comfy`（已装 soundfile）
