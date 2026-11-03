# LitInfer

> 从零实现的 CPU 大模型推理引擎 · C++17 / Python / AVX2 / OpenMP

LitInfer 是一个**不依赖 PyTorch 的轻量级 LLM 推理引擎**。它用 C++ 手写 Tensor、Transformer 算子与 Qwen2 前向计算，通过稳定的 C API 暴露给 Python，并在其上实现了算子向量化、INT8 量化、分页 KV Cache、PagedAttention、连续批处理和 OpenAI 兼容的流式服务。

项目目标不是复刻一个工业级框架，而是把大模型推理链路拆开、自己实现一遍：**数据怎么存、矩阵乘怎么快、KV Cache 怎么省、多请求怎么调度**。

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 核心特性

| 模块 | 实现内容 |
|---|---|
| **推理引擎** | Qwen2 端到端推理（prefill + decode + KV Cache），C++ 后端 / C API / Python ctypes 三层架构 |
| **自定义算子** | `linear`、`rms_norm`、`rope`、`self_attention`、`swiglu`、`embedding`、`argmax`、`rearrange`，支持 F32 / F16 / BF16 |
| **性能优化** | Profiler 热点定位；AVX2 / FMA / F16C 向量化；OpenMP 多线程；RoPE 预计算；online softmax |
| **量化推理** | INT8 weight-only 对称 per-channel 量化，在线反量化融合 GEMV，加载后销毁 FP16 权重 |
| **分页 KV Cache** | 16-token block 池 + 每序列 block table，按需分配/释放，解决连续预分配的内存碎片 |
| **PagedAttention** | 单序列 / 多序列批量 attention，支持 GQA、乱序 block table、online softmax |
| **连续批处理** | C++ 引擎按 iteration 重新组 batch，Python 后台调度线程 + Future，请求完成即释放槽位 |
| **推理服务** | FastAPI + SSE，OpenAI 兼容 `/v1/models`、`/v1/chat/completions`，支持逐 token 流式输出 |
| **正确性验证** | 与 NumPy / PyTorch 逐算子对拍；与 HuggingFace 逐 token 对齐 |

---

## 系统架构

```text
┌──────────────────────────────────────────────────────────┐
│ Python 前端  python/llaisys/                             │
│   读取 config.json / safetensors，ctypes 调用 C API       │
├──────────────────────────────────────────────────────────┤
│ C API 边界  include/llaisys/*.h                          │
│   稳定 ABI，屏蔽 C++ name mangling                        │
├──────────────────────────────────────────────────────────┤
│ 模型层  src/llaisys/models/qwen2.cpp                     │
│   Qwen2 forward、分页 KV、INT8 权重、连续批引擎            │
├──────────────────────────────────────────────────────────┤
│ 算子层  src/ops/                                         │
│   linear / attention / rope / swiglu / quantize / paged   │
│   AVX2 + OpenMP 内核，CPUID 运行时能力分派                 │
├──────────────────────────────────────────────────────────┤
│ 运行时  src/core/  src/device/  src/tensor/              │
│   Tensor(storage/shape/stride/offset)、内存与设备抽象      │
└──────────────────────────────────────────────────────────┘

服务层  serving/
  server.py         单请求 / SSE 流式
  engine_server.py  连续批处理 HTTP 服务
  engine_runtime.py 后台调度线程 + Future
```

一次 decode 的数据流：

```text
token ids → embedding → [RMSNorm → QKV → RoPE → Attention(KV Cache)
→ O → 残差 → RMSNorm → SwiGLU → 残差] × N
→ final RMSNorm → lm_head → logits → 采样 → next token
```

---

## 性能评测

测试环境：**Intel i5-1035G1（4C/8T，15W）/ 8 GB RAM / Windows 11 / MSVC + xmake**，模型为 **DeepSeek-R1-Distill-Qwen-1.5B（Qwen2 架构，FP16）**。每组 3 条 prompt、`max_steps=8`、`repeats=3`，表中为 3 条 prompt 的 median 范围。

| 配置 | 线程数 | TTFT (s) | Decode (token/s) |
|---|---:|---:|---:|
| FP16 | 1 | 3.86 – 6.85 | 1.78 – 2.71 |
| FP16 | 8 | 1.92 – 3.15 | 3.31 – 3.52 |
| INT8 weight-only | 1 | 2.88 – 4.87 | 3.30 – 3.45 |
| INT8 weight-only | 8 | 1.66 – 2.59 | 4.43 – 4.67 |

**观察到的结论：**

- OpenMP 1→8 线程：decode 提速 **约 1.2–2.0×**，prefill（TTFT）提速 **约 2.0–2.2×**。
- INT8 单线程 decode 相对 FP16 提速 **约 1.25–1.94×**；8 线程仅 **约 1.3×**，未达到权重字节减半对应的 2×。
- INT8 greedy 输出与 FP16 的 token 一致率 **81.2%（39/48）**，其中 **2/3 条 prompt 逐 token 100% 一致**。
- 连续批处理 4 路并发 **8/8 请求成功、0 错误、无崩溃**；当前聚合吞吐约 **1.0–1.16×**，瓶颈在 linear GEMM 未做 M 复用（每个 batch 行重复流式读取权重），不是 PagedAttention。

> 口径说明：以上为单机 CPU 实测，**不可**与桌面多通道内存 CPU 的 20–30 token/s 直接比较。连续批吞吐数据受 8 GB 内存与双服务进程换页影响，仅作为**正确性证据**，不作为加速结论。原始评测数据位于 [`benchmarks/`](benchmarks/)，其中 [`torch_align.jsonl`](benchmarks/torch_align.jsonl) 记录了与 PyTorch greedy 输出逐 token 一致（`matched: true`）的证据。

### 为什么 decode 是瓶颈

decode 阶段每生成一个 token 都要流式读取全部线性层权重，是典型的 **memory-bound**：

- FP16 每 token 约流式读取 **2.62 GB** 权重，INT8 约 **1.31 GB**；
- 本机可用 DRAM 带宽由 prefill 反推约 **14.5 GB/s**，因此 FP16 8 线程的物理下限约 **5.5 token/s**；
- 当前 AVX 内核内层只有 1 个 FMA 累加器，存在 4-cycle 依赖链，单核计算吞吐被锁死——这是单线程与 prefill 的主要优化空间。

---

## 快速开始

### 1. 环境依赖

- C++17 编译器（Windows: MSVC；Linux/macOS: GCC/Clang）
- [xmake](https://xmake.io/)
- Python 3.9+，8 GB 以上内存

### 2. 编译与安装

```bash
# 编译 C++ 共享库，并复制到 Python 包目录
xmake
xmake install

# 安装运行依赖（numpy / fastapi / uvicorn / transformers）
pip install -r requirements.txt

# 让 Python 直接使用仓库内的包
# Linux / macOS
export PYTHONPATH=python
# Windows PowerShell
$env:PYTHONPATH = "python"
```

### 3. 准备模型

从 HuggingFace 下载任意 Qwen2 架构模型，例如：

```text
DeepSeek-R1-Distill-Qwen-1.5B/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── model-*.safetensors
```

### 4. 验证

```bash
# 算子正确性冒烟测试（NumPy 参考）
python scripts/smoke_test.py

# 端到端 benchmark
python scripts/benchmark_cpu.py \
  --model ../DeepSeek-R1-Distill-Qwen-1.5B \
  --device cpu --max_steps 8 --repeats 3 --threads 8
```

### 5. 启动 OpenAI 兼容服务

```bash
# 流式服务（单请求 / SSE）
python -m serving.server --model ../DeepSeek-R1-Distill-Qwen-1.5B --port 8000

# 连续批处理服务（多请求并发）
python -m serving.engine_server --model-path ../DeepSeek-R1-Distill-Qwen-1.5B --port 8001
```

调用示例：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"litinfer-qwen2","messages":[{"role":"user","content":"你好"}],"stream":true,"max_tokens":16}'
```

### 6. Python API

```python
from transformers import AutoTokenizer
from llaisys.models.qwen2 import Qwen2

model_path = "../DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = Qwen2(model_path)              # 或 Qwen2(model_path, quantize="int8")
input_ids = tokenizer.encode("你好")
output_ids = model.generate(input_ids, max_new_tokens=16)
print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

---

## 目录结构

```text
include/llaisys/          C API 头文件
src/core/                 Tensor 存储、内存分配
src/device/               设备抽象（CPU 已实现，CUDA 仅骨架）
src/tensor/               Tensor 实现（shape / stride / offset / view）
src/ops/                  算子实现
  ├── linear/cpu/         AVX2 GEMV / GEMM 内核
  ├── quantize/           INT8 量化与在线反量化
  └── paged_attention/    单序列 / 批量分页 attention
src/llaisys/models/       Qwen2 forward、KV Cache、连续批引擎
src/llaisys/kv_cache/     分页 KV 块池与 block table
src/utils/                Profiler
python/llaisys/           Python 前端（ctypes / sampling / 模型加载）
serving/                  OpenAI 兼容 HTTP 服务
scripts/                  冒烟测试、benchmark、压测与对比脚本
benchmarks/               原始评测 JSON
test/                     算子、Tensor、推理与 KV Cache 测试
```

---

## 关键实现

**1. Profiler 驱动优化**

通过环境变量 `LLAISYS_PROFILE=1` 开启按算子计时，定位到 prefill 阶段 `linear` 占比约 **94.85%**，随后优先重写矩阵乘内核，而不是凭直觉优化。

**2. AVX2 / FMA / F16C + OpenMP**

- decode 走 GEMV，prefill 走 GEMM，分别按输出行 / 输入行并行；
- `F16C` 一次转换 8 个 FP16，`FMA` 融合乘加；
- CPUID 运行时检测，缺少 AVX2 的机器自动回退通用内核；
- 权重按需读取，避免生成完整转置副本。

**3. INT8 weight-only 量化**

对每个线性层权重按行做对称量化：

```text
scale[n] = max(|W[n, :]|) / 127
q[n, k]  = clip(round(W[n, k] / scale[n]), -127, 127)
```

计算时在 GEMV/GEMM 累加后乘以 `scale`，**不物化反量化后的 FP16 权重矩阵**。tied embedding 场景下，`lm_head` 量化出独立副本，词嵌入保持 FP16 以支持查表。

**4. 分页 KV Cache + PagedAttention**

- KV 以 **16 token** 为一个物理 block 存放在共享池中，空闲块用 free list 管理；
- 每个请求维护 `logical block → physical block` 的 block table，支持非连续块寻址；
- attention 使用 **online softmax** 一遍遍历完成，支持 GQA 与多序列批量；
- 修复了按 `max_position_embeddings`（13 万）预分配 KV 导致数 GB OOM 的问题。

**5. 连续批处理**

C++ 引擎把所有未完成序列的 query 沿 token 维拼接，共享一次前向；Python `EngineRuntime` 在后台线程循环调用 `engine_step`，提交请求返回 Future。修复了 `engine_add_request` 与后台调度线程之间的 `vector` 数据竞态，以及 `max_new_tokens` off-by-one。

---

## 已知限制与 Roadmap

当前版本优先保证**正确性与可复现性**，性能优化仍有明确空间：

- [ ] **CUDA / GPU 后端**：仅保留设备抽象骨架，尚未实现；
- [ ] **多累加器 FMA 内核**：当前单累加器依赖链限制单核吞吐，预期 1.5–2×；
- [ ] **GEMM M/N 分块**：让一次权重加载服务多行输入，提升 prefill 与连续批吞吐；
- [ ] **PagedAttention 向量化**：长上下文（≥1K）下仍为标量实现，是下一阶段首要热点；
- [ ] **量化质量评估**：当前使用 greedy token 一致率，后续补充 PPL 增量指标；
- [ ] **静态服务并发安全**：`server.py` 使用共享单序列 KV，仅适合串行/流式；并发请使用 `engine_server.py`。

---

## 测试

```bash
# 算子 / Tensor / 推理测试
python test/test_tensor.py
python test/ops/linear.py
python test/test_infer.py --model ../DeepSeek-R1-Distill-Qwen-1.5B --test

# INT8 与 FP16 输出一致性
python scripts/compare_int8_fp16.py --model ../DeepSeek-R1-Distill-Qwen-1.5B --max_steps 16

# 连续批调度逻辑（Mock，无需权重）
python scripts/test_engine_runtime.py
```

> `--test` 会关闭 top-p / top-k / temperature 采样，强制双方使用 greedy，便于逐 token 对齐。

---

## 致谢与许可

本项目基于 [InfiniTensor/llaisys](https://github.com/InfiniTensor/llaisys) 课程框架扩展而来，上游版权归 InfiniTensor 所有，遵循 [MIT License](LICENSE)。LitInfer 在其基础上完成了算子优化、INT8 量化、分页 KV Cache、PagedAttention、连续批处理与服务化等扩展。

推理主路径不依赖 PyTorch；PyTorch / HuggingFace 仅用于结果对齐与参考实现。
