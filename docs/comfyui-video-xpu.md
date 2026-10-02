# 在本机生成视频（ComfyUI/.venv，xpu 环境）

> **⚠️ 本文描述的是另一套环境，与本仓库主流程不同，请勿混用。**
>
> 本仓库主流程（`setup.sh` 建的 `env/comfy`）是 **CUDA 版**：torch 2.11.0+cu128，
> 跑在 RTX 6000 Ada 上，可用满 48 GB 显存。**日常出片请以根目录
> [README.md](../README.md) 为准。**
>
> 本文对应的是另一处独立检出 `/home/acm/paul/cat`：ComfyUI 在 `ComfyUI/`，
> venv 在 `ComfyUI/.venv`，torch 是 **2.14.0+xpu，没有 CUDA**，
> 因此 ComfyUI 落到 Intel Arc 核显（`xpu:0`，31 GB 共享）。
>
> 由此导致本文的结论**只对 xpu 环境成立**：
>
> - 「ComfyUI 看不到 RTX 6000 Ada」「31 GB 装不下」——在 CUDA 环境里都不成立。
> - **checkpoint 建议正好相反**：xpu 环境只能上 bf16；而 nvfp4 / `int8-convrot`
>   在 CUDA 环境才是可用且推荐的（见根 README 的 Ada 限制一节：
>   Ada 不支持 nvfp4，故 CUDA 流程走 bf16 + fp8 weight_dtype + `-conv` VAE）。
> - 本文提到的 `extra_model_paths.yaml`、模型软链路径，也只针对 `/home/acm/paul/cat`
>   那套目录结构，与本仓库的 `env/comfy` 无关。

用 ComfyUI 跑文生视频 / 图生视频。该目录下是 ComfyUI 的本地检出，
配好 uv 虚拟环境，FLUX.2 klein 9B 权重已在本地。

环境布局、venv 路径、测试命令见该目录的 `AGENTS.md`，本文只讲视频生成。

---

## 先读这一节：显存是硬约束

ComfyUI 在这台机器上**看不到** RTX 6000 Ada。

- 装的是 `+xpu` 版 torch，`torch.cuda.is_available()` 为 `False`，
  `comfy.model_management.get_torch_device()` 返回 **`xpu:0`** —— Intel Arc 核显，
  **31 GB** 共享内存。启动日志实测：`Device: xpu:0 Intel(R) Graphics [0xe223]`。
- RTX 6000 Ada（49 GB）目前有 ~44 GB 被 Qwen3.8 的 vLLM 服务占着（监听 8000 端口，
  从 `~/paul/qwen3.8` 启动）。除非明确要求，不要动它。

LTX-2.5 默认模板要的权重已经超过 31 GB（还没算任何激活值显存）：

| 组件 | 文件 | 体积 |
|---|---|---|
| DiT（int8 convrot） | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | 20.0 GB |
| 文本编码器（int8 convrot） | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | 14.3 GB |

光权重就 34 GB，ComfyUI 会开始往系统内存里换（本机 183 GB 内存，约 85 GB 空闲），
速度会很慢。可行做法，按推荐顺序：

1. **换 nvfp4 DiT**：`ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` 为
   17.4 GB。配 31 GB 依然偏紧，但换页少得多。注意 nvfp4 需要 Blackwell 级显卡才快，
   在 Intel 上不会快。
2. **换 Conv VAE**：用 `ltx-2.5-video-vae-conv-bf16.safetensors` 代替
   `ltx-2.5-video-vae-bf16.safetensors`。解码更省更快，画质略降。
3. **从小尺寸起步**：512×288、3 秒、16 fps → `length` 为 49（而不是 121）。
   帧数对 latent 体积和 VAE 峰值显存的影响远大于分辨率。
4. **真正腾出显卡**：只停 vLLM 不够——当前 venv 没有 CUDA 版 torch，
   ComfyUI 照样退回 `xpu:0`。要上 CUDA 得另建一个 venv。

先做 1 + 3，再去判断是不是真坏了。

---

## 本地模型

`~/work/models/ltx-2.5/`（共 188 GB）是按 ComfyUI 目录切分的权重包，
目前**还没有**注册给 ComfyUI。

```
~/work/models/ltx-2.5/
├── diffusion_models/      ltx-2.5-22b-{distilled,dev}-transformer-{bf16,comfy-int8-convrot,nvfp4}.safetensors
├── text_encoders/         gemma4-12b-with-proj-ltx-2.5-{bf16,comfy-int8-convrot}.safetensors
├── vae/                   ltx-2.5-{video-vae,video-vae-conv,audio-vae}-bf16.safetensors
├── loras/                 ltx-2.5-22b-distilled-lora-450-bf16.safetensors
├── latent_upscale_models/ ltx-2.5-latent-{spatial,temporal}-upscaler-x2-bf16-1.0.safetensors
└── model_patches/         ltx-2.5-duration-head-bf16.safetensors
```

另有一份 `~/work/models/ltx-2.5/README.md`，是上游的模型卡，逐个文件说明了用途——
换 checkpoint 前值得先读。

### 会踩到的缺失文件

- `gemma4_e2b_it_int8_convrot.safetensors` —— **prompt enhance**（提示词增强）路径
  用的第二个文本编码器。**本机没有**（已全盘搜索确认）。所以「Enable Prompt Enhance」
  必须保持关闭。主用的 12B 编码器是有的。
- `~/work/models/h3-comfy/` 缺 `minimax_h3_fl2va_pruned_int8_convrot.safetensors`，
  而 `Image to Video (MiniMax H3)` 模板要的就是它。手上有
  `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` 和一个 turbo LoRA，
  所以那个模板需要先改模型下拉框才能加载。

### 注册模型路径

写 `ComfyUI/extra_model_paths.yaml`（已被 gitignore）：

```yaml
ltx25:
    base_path: /home/acm/work/models/ltx-2.5/
    diffusion_models: diffusion_models/
    text_encoders: text_encoders/
    vae: vae/
    loras: loras/
    latent_upscale_models: latent_upscale_models/
    model_patches: model_patches/
```

目录名必须和 `folder_paths.py` 里的 key **完全一致**，写错会静默不生效。
把单个文件软链到 `ComfyUI/models/` 也可以，但 yaml 更不容易出错。
改完都要重启服务。

---

## 跑起来

```bash
cd /home/acm/paul/cat/ComfyUI
.venv/bin/python main.py
```

打开 <http://127.0.0.1:8188>，然后 **Workflows → Templates**，选一个：

- `Text to Video (LTX-2.5)` —— 文生视频
- `Image to Video (LTX-2.5)` —— 图生视频
- `First & Last Frame to Video (LTX-2.5)` —— 首尾帧插值
- `Image to Video (MiniMax H3)` —— 需按上面改模型

这些模板会加载成一个已经连好线的 subgraph 节点。对外只暴露少数几个控件，
其余都在内部。

### 真正要动的几个控件

以 `Text to Video (LTX-2.5)` 为例，默认值：

| 控件 | 默认 | 说明 |
|---|---|---|
| Prompt | *(空)* | |
| Enable Prompt Enhance | `false` | **必须保持关闭** —— 它需要的模型本机没有 |
| Duration（秒） | `5` | |
| Width | `1280` | |
| Height | `720` | |
| Frame Rate | `24` | |
| Seed | 随机 | |

显存不够时，先降 `Duration`、`Width`、`Height`、`Frame Rate`。
理解下面的换算之后，就别再随便改 `Duration` 和 `Frame Rate` 了。

---

## LTX-2.5 流程是怎么串的

以下都是从模板的 subgraph JSON 里读出来的。之所以强调这一点：好几个数值是**算出来的**
而不是填进去的，改错节点会静默让 latent 对不上。

### 帧数和分辨率是推导出来的

```
length = duration * fps + 1     # 5 * 24 + 1 = 121
基础宽 = width / 2             # 1280 -> 640
基础高 = height / 2            # 720  -> 360
```

由三个 `ComfyMathExpression` 节点完成。第一阶段在半分辨率下采样，
之后 `LTXVLatentUpsampler` 再放大一倍。如果 Width 设成奇数，
或者改了 fps 却没同步改这些表达式节点，latent 就和最终尺寸对不上了。

### 两段采样，固定调度

不是「KSampler + 一个 steps 字段」。distilled DiT 用 `ManualSigmas` 节点，
sigma 调度是写死的：

- **第一阶段**（8 步）：`1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`
- **第二阶段**（x2 上采样之后，4 步）：`0.85, 0.7250, 0.4219, 0.0`

两段都用 `euler_ancestral`。 guider 是 `LTXVDualCFGGuider`，它有分开的
`video_cfg` 和 `audio_cfg` 两个输入（节点默认 3.0 / 7.0）；
distilled 模型在模板里两个都设成 `1.0`。要改就改 sigma 字符串，不是改步数字段。

### 音频是和视频一起生成的

LTX-2.5 不是无声视频模型。采样前 `LTXVConcatAVLatent` 把视频和音频 latent 合并，
采样后 `LTXVSeparateAVLatent` 再拆开，分别喂给 `VAEDecodeTiled`（视频，
512 分块、64 像素重叠）和 `LTXVAudioVAEDecode`（音频，含声码器）。
这也是音频 VAE 总会被加载的原因。视频和音频是**同一次**扩散过程，
不能只砍掉音频分支而留着音频 latent。

### 完整链路

```
UNETLoader ─┬─> LTXVDualCFGGuider -> SamplerCustomAdvanced（8 步，640x360）
            │        ^                    ^
   CLIPLoader┘       │              EmptyLTXVLatentVideo（length=121）
   （prompt enhance 经 ComfySwitchNode，默认关）
                     │
                     v
            LTXVLatentUpsampler（x2，来自 LatentUpscaleModelLoader）
                     │
                     v
            SamplerCustomAdvanced（4 步，1280x720）
                     │
            LTXVSeparateAVLatent ─┬─> VAEDecodeTiled ────┐
                                   └─> LTXVAudioVAEDecode┤
                                                            v
                                              CreateVideo（fps、位深、色彩空间）
                                                            v
                                                     SaveVideo
```

---

## 输出

`SaveVideo` 写到 `ComfyUI/output/video/`。默认文件名前缀是 `video/ComfyUI`，
落盘文件名形如 `ComfyUI_00001_.mp4`。

容器：`mp4`、`mkv`、`webm`（还有 `auto`：H.264 选 mp4，AV1 选 webm）。
编码：`auto`、`h264`、`av1`。`CreateVideo` 另有 `bit_depth`
（`auto` / 8 / 10）和 `color_space`（`sRGB` / `HDR` / `HDR PQ`）——
`auto` 位深在 sRGB 下取 8 位、HDR 下取 10 位。

用 `PreviewAny` 节点可以在 UI 里直接预览，不必落盘。

---

## 用 API 驱动

`POST http://127.0.0.1:8188/prompt`，payload 结构见
`ComfyUI/script_examples/basic_api_example.py`。

这些模板是 **subgraph**，手写节点图不值得。在 UI 里载入模板、填好提示词，
然后 **Workflow → Export (API)**，把导出的 JSON POST 过去即可。
之后只需改 `text`、seed 和那几个 int 控件。

---

## 选哪个 checkpoint

| 需求 | 选择 |
|---|---|
| 默认、步数最少 | distilled DiT，`euler_ancestral`，上面写死的 sigma |
| 显存压力小些 | `-nvfp4` DiT（17.4 GB） |
| 解码更快更省 | `ltx-2.5-video-vae-conv-bf16.safetensors` |
| 画质更好、显存更紧 | `-bf16` DiT（39.1 GB）——会大量换页 |
| 要训练 / 要更高创造性 | `dev` DiT，需要更多步数和真实的 CFG |

`comfy-int8-convrot` 变体只能在 ComfyUI 里用，**不要**喂给 `ltx-pipelines`
或原生 PyTorch。`loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors`
是给 `dev` transformer 工作流用的，不是 distilled 默认流程。

---

## 需要说明的地方

- 以上内容**没有端到端跑过**。每个文件名、体积、默认值、sigma 字符串和公式，
  都是从模板 JSON、`comfy_extras/nodes_lt.py` 与 `nodes_video.py` 的节点定义、
  以及磁盘上的实际文件里读出来的。第一次真跑大概率会撞上显存问题——
  这正是第一节存在的原因。
- Intel 核显上跑 LTX-2.5，预期是**每条片子几分钟**，不是几秒。distilled 只有 8 步，
  但 DiT 有 22B 参数。
- `output/`、`temp/`、`user/comfyui.db` 都是 gitignore 的本地状态，不是源码。