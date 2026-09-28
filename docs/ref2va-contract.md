# MiniMax-H3 `ref2va` 打包契约（从 diffusers 官方实现提取）

来源：`diffusers/modular_pipelines/minimax_h3/`（本机 venv 内 diffusers 0.40.0）
- `references.py` — 参考类型
- `before_encoder.py` — 归一化（Resize/Setup）
- `encoders.py` — 文本呈现 + 参考编码
- `before_denoise.py` — 布局 + 条件 latent + 时间步
- `denoise.py` — 去噪循环
- `schedulers/scheduling_minimax_h3.py` — 调度器

目标：在 `h3_t2va.py`（原生 int8）上复刻 `ref2va`，与官方实现逐位对齐。

---

## 1. 常量

| 名称 | 值 |
|---|---|
| `fps` | 24 |
| `text_tag` / `video_tag` / `audio_tag` | **1 / 0 / 2** |
| `text_encoder_layer` | 50（取 Qwen3-VL 第 50 层 hidden state，非最后一层） |
| `audio_channels` | 2（channel-major） |
| `audio_latent_channels` | 32 |
| `audio_sampling_rate` | 32000 |
| `audio_latents_per_second` | 40 |
| `vae_frames_per_chunk` / `vae_latents_per_chunk` | 17 / 5 |
| `vae_spatial_compression_ratio` | 16 |
| `patch_size` | (1, 2, 2) |
| `canvas_multiple` | 32 (= 16 × 2) |
| `canvas_short_edge` / `canvas_max_pixels` | 768 / 768×1344 |
| `reference_image_short_edge` | **2048** |
| `keyframe_encode_seed` | 42 |
| `keyframe_noise_aug` | **0.999** |
| `pixel_mean` / `pixel_std` | ImageNet (0.485,0.456,0.406) / (0.229,0.224,0.225) |
| `min_duration` / `max_duration` | 5.0 / 15.0 |
| video scheduler shift / audio scheduler shift | 12.0 / 3.0 |
| 上限 | 图 ≤9，视频 ≤3，音频 ≤3，合计 ≤12；音频不能单独存在 |

---

## 2. 参考归一化（`MiniMaxH3Ref2VASetupStep`）

- **画布**：ref2va 的参考**不约束**生成几何；默认 16:9 → `resolve_canvas_size(16,9,32,768,768*1344)`。
- **帧数**：对齐到 `17*n+5`，时长 ∈ [5,15]s。
- **图片参考**：缩放到**短边 2048**（可上采样、无面积上限），对齐到 32 的倍数：
  `scale = 2048 / min(w,h)`；`target_h = max(32, round(h*scale/32)*32)`，宽同理；PIL LANCZOS。
- **视频参考**：重采样到 24fps（丢/复制整帧），截断到生成帧数，放到**它自己宽高比**解析出的画布（与目标同一规则）。
- **音频参考**：截断到生成时长，重采样到 32kHz，单声道复制成双声道（`(2, N)` float32）。

---

## 3. 文本呈现（`MiniMaxH3Ref2VATextEncoderStep._build_presentation`）

按**参考顺序**为每个参考加标签，标签按模态各自编号：

- 图片参考：`"<Picture i>: "` + 视觉块
- 音频参考：`"<Audio j>: "`（无视觉块）
- 视频参考：若带声 → 先 `"<Audio j>: "`；再 `"<Video k>: "` + 每个合并帧对一个 `"<{t:.1f} seconds>"` + 视觉块
- 最后 **prompt 原文**（无 chat template、无特殊 token）

视觉块 token 形式：
```
<|vision_start|>  +  [pad_token] * num_vision_tokens  +  <|vision_end|>
```
- 图片用 `<|image_pad|>`；视频用 `<|video_pad|>`。
- **这些行的 `token_tag = video_tag (0)`**，其余文本行 `= text_tag (1)`。

`mm_token_type_ids`（Qwen 内部，0=text/1=image/2=video）由 `processor.create_mm_token_type_ids` 生成，驱动 Qwen3-VL 的**3D mrope**，不离开 conditioner。

> ⚠️ 关键：图片参考的视觉块需要 **Qwen3-VL 视觉塔**（27 层 ViT，hidden 1152，patch 16，merge 2）+ **deepstack** 注入。原生 `h3_t2va.py` 只有纯文本流式编码器，**不含视觉**。本方案改用 HF `Qwen3VLForConditionalGeneration` 计算 `prompt_embeds`。

---

## 4. 参考编码（`MiniMaxH3Ref2VAReferenceEncoderStep`）

**图片/视频 → 视频 VAE**（`encode_vae_condition`）：
1. 像素 `uint8` → `float/255`，按 `pixel_mean/std` 归一化；
2. `vae.encode(pixels)` → posterior；
3. **采样** `posterior.sample(generator=manual_seed(42))`（注意：是采样，不是 mean）；
4. 转 `float16` 再回 `float`（量化到 ~11 bit）；
5. 移到 CPU，按 `latents_mean/std` 归一化。
- 图片：`pixels` 形状 `(1,3,1,H,W)`（单帧，走空间编码器）。
- 视频：帧数向下对齐到 `17n+5` 再编码（走时间分块）。

**音频 → 音频 VAE**：`audio_vae.encode(audio[:,None])` → `posterior.mode()`（**不采样**）→ `transpose(1,2)` → 归一化 → reshape 成 `(num_audio_latents*2, 32)`。

---

## 5. 打包序列（`build_ref2va_packed_sequence`）

顺序：
```
[ text | 参考块（按请求顺序） | 目标音频 | 目标视频 ]
```
每个参考贡献：
- image → 视频行
- audio → 音频行
- video → 音频行（若带声）**紧接**视频行

**RoPE 时钟**（`rotary_time` 从 `num_text_tokens` 起，共享音视频）：
- text 行：`position_ids[:,0] = arange(num_text_tokens)`，h/w = 0
- image 参考：视频行 t = `rotary_time`（整帧只占 **1.0**，不是 5/3），h/w 用该参考自己的 `_frame_position_grid`；`rotary_time += 1.0`
- audio 参考：`_fill_audio_positions`；`rotary_time += num_audio_latents`
- video 参考：音频行用**该视频自己的** width grid；视频行 t = `_temporal_position_grid(num_latent_frames, rotary_time)`（帧跨 `5/3*(1,4,4,4,4)`），h/w 用其自己的 grid；`rotary_time += max(reference_audio_latents, video_span)`，其中 `video_span = Σ 5/3*pattern[i%5]`（**顺序求和**，非 pairwise）
- 目标：音频行起点 = cursor，视频行紧随；两者共用参考块留下的 `rotary_time`；音频用**目标** width grid，视频用目标 frame grid。

**空间 grid**（`_spatial_position_grid`）：`ratio = dim/sqrt(area)`，`left=(1-ratio)/2`，`np.linspace(left, left+ratio, dim//patch, endpoint=False) * 32`（float64，必须用 numpy 语义）。

**行索引**：`video_indices` = 各参考视频行 + 目标视频行；`audio_indices` = 各参考音频行 + 目标音频行；`text_indices = arange(num_text_tokens)`。
返回 `num_reference_video_rows` / `num_reference_audio_rows`（= 条件行数）。

---

## 6. 条件 latent 噪声化（`MiniMaxH3PrepareConditionLatentsStep`）

- **先**抽条件噪声（每个条件一次），**再**抽生成噪声——顺序影响 generator 复现。
- 每个条件 latent 各自：`noise = randn(shape, generator)`；`noised = scale_noise(cond, 0.999, noise)`；`patchify`。
- `scale_noise(x0, t, noise) = t*x0 + (1-t)*noise`（t=0.999 ≈ 干净）。
- `condition_rows = cat(packed)`，作为**前导视频行**拼到生成视频行前：`latents = cat([condition_rows, target_video_rows])`。
- 音频参考行**不噪声化**（`t=1.0`），直接拼到目标音频行前：`audio_latents = cat([ref_audio_rows..., target_audio_rows])`。

---

## 7. 行时间步（`build_row_timesteps`）

每步：
- 默认全部行 = `video_timestep`
- 条件视频行 = `max(video_timestep, keyframe_noise_aug=0.999)`
- 音频行 = `audio_timestep`
- 条件音频行 = `1.0`
- 归约为 `(unique_timesteps, timestep_indices)`；AdaLN 行 = `timestep_indices * 3 + token_tags`

**调度器**（flow-match，t = 1 - sigma）：
- `set_timesteps`：`base=linspace(1,0,N)`；`sigma = shift*base/(1+(shift-1)*base)`；`unique_consecutive`；`timesteps = 1 - sigmas[:-1]`
- `scale_noise(x0,t,noise) = t*x0 + (1-t)*noise`
- `step`：`sigma_from_t = 1 - t`；`x0 = x_t + sigma_from_t * v`（data-ward velocity，**加号**）；`r = sigma_next/sigma`；`x_next = r*x_t + (1-r)*x0`
- **只更新生成行**：`latents[num_condition_video_rows:]`、`audio_latents[num_condition_audio_rows:]`；条件行永不被写 → 天然保持。

---

## 8. 与现有 `h3_t2va.py` 的差异清单

| 项 | t2va（现有） | ref2va（新增） |
|---|---|---|
| 序列布局 | `[text \| target audio \| target video]` | `[text \| ref blocks \| target audio \| target video]` |
| 文本编码 | 纯文本流式（layers 0-49，1D rope） | **需视觉塔 + 3D mrope + deepstack** |
| 条件行 | 无 | 图/视频条件行（noised@0.999）+ 音频参考行（t=1） |
| 时间步 | text+video 一个 t，audio 一个 t | 4 类行（video/cond-video/audio/cond-audio） |
| RoPE | 目标网格 | 每个参考块自己的网格 + 时钟推进 |
| VAE | 仅 decode | 需 **encode**（图片单帧、视频分块） |
| 噪声抽取 | video→audio | **condition→video→audio** |

---

## 9. 对拍 oracle

用 diffusers 的**纯函数**生成 golden（无需大模型）：
- `MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(...)`
- `MiniMaxH3SetTimestepsStep.build_row_timesteps(...)`
- `MiniMaxH3Scheduler.set_timesteps` / `scale_noise` / `step`
- `_spatial_position_grid` / `_temporal_position_grid` / `_frame_position_grid` / `_fill_audio_positions`

原生实现必须与这些 golden 逐位（float64）对齐。
