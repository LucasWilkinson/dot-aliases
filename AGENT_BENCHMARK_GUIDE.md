# vLLM Benchmark and Evaluation Guide for AI Agents

**Last Updated**: 2025-10-18  
**Repository**: `/home/lwilkinson/code/vllm`

This guide provides instructions for AI agents on how to run vLLM benchmarks and evaluations efficiently.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Configuration Essentials](#configuration-essentials)
3. [Benchmarking](#benchmarking)
4. [GSM8K Evaluations](#gsm8k-evaluations)
5. [GPQA-Diamond Evaluations](#gpqa-diamond-evaluations)
6. [Profiling](#profiling)
7. [Agent Output Guidelines](#agent-output-guidelines)
8. [Advanced Configuration](#advanced-configuration)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Decision Tree

| User Asks | Use Command |
|-----------|-------------|
| "Benchmark current branch" | `benchmark --model <MODEL> -tp N` |
| "Compare main vs my-branch" | `benchmark-compare --model <MODEL> --main-ref main --pr-ref my-branch` |
| "GSM8K eval on model" | `gsm8k-eval --model <MODEL>` |
| "GPQA-Diamond eval on model" | `gpqa-diamond-eval --model <MODEL>` |
| "Profile model" | `profile --model <MODEL> -tp N --random-input-len <IN> --random-output-len <OUT> --num-prompts <N>` |

### Common Commands

```bash
# Benchmark current branch (standalone)
benchmark --model deepseek-ai/DeepSeek-V3.2-Exp -tp 8 --rates "1,5,10"

# With rebuild
benchmark --model deepseek-ai/DeepSeek-V3.2-Exp --build

# Compare branches
benchmark-compare --model deepseek-ai/DeepSeek-V3.2-Exp \
  --main-ref main --pr-ref my-feature \
  --build-main 1 --build-pr 1 -tp 8

# GSM8K evaluation
gsm8k-eval --model deepseek-ai/DeepSeek-V2-Lite --limit 100

# GPQA-Diamond evaluation
gpqa-diamond-eval --model deepseek-ai/DeepSeek-V2-Lite

# Profile model (capture torch profiler traces)
profile --model deepseek-ai/DeepSeek-V2-Lite -tp 2 --random-input-len 1000 --random-output-len 100 --num-prompts 50

# With GPU reservation (if chg available)
chg run -g 8 -- benchmark --model deepseek-ai/DeepSeek-V3 -tp 8
```

### Request Pattern Recognition

| User Says | Interpret As |
|-----------|--------------|
| "benchmark in 1k out 128 num prompts 512" | `--random-input-len 1000 --random-output-len 128 --num-prompts 512 --dataset-name random --request-rate inf` |
| "bench 2048 in 256 out rate 5" | `--random-input-len 2048 --random-output-len 256 --request-rate 5 --dataset-name random` |
| "mtp 3" or "with mtp 3" | `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` |
| "profile X in Y out N prompts" | Use `vllm bench serve --profile` with `VLLM_TORCH_PROFILER_DIR` set |

---

## Configuration Essentials

### Model Name Mappings

| User Says | HuggingFace ID |
|-----------|----------------|
| `DeepSeek-V3.2-Exp` or `DeepSeek-V3.2` | `deepseek-ai/DeepSeek-V3.2-Exp` |
| `DeepSeek-V3` | `deepseek-ai/DeepSeek-V3` |
| `DeepSeek-R1` | `deepseek-ai/DeepSeek-R1` |
| `DeepSeek-V2-Lite` | `deepseek-ai/DeepSeek-V2-Lite` |
| `llama3 8b` or `Llama-3-8B` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `llama3 70b` | `meta-llama/Meta-Llama-3-70B-Instruct` |
| `Qwen2.5` | `Qwen/Qwen2.5-<size>-Instruct` |
| `Qwen3 Next` | `Qwen/Qwen3-Next-80B-A3B-{Instruct,Thinking}{,-FP8}` (default: Instruct) |

### Parallelism Defaults

**If user doesn't specify N for `-tp N`, `-dp N`, or `-dcp N`:**

| Model Family | Default TP/DP/DCP |
|--------------|-------------------|
| DeepSeek-V3 (all variants including V3.2) | `8` |
| DeepSeek-R1 | `8` |
| DeepSeek-V2-Lite | `2` |
| Other models | `2` |

**Usage:**
- Tensor Parallel: `-tp N` (split model layers across GPUs)
- Data Parallel: `-dp N --enable-expert-parallel` (for MoE models)
- Distributed Checkpoint: `-dcp N` (distributed checkpoint parallelism)
- Pipeline Parallel: `-pp N` (split model stages across GPUs)

### Port Management

**Default port order**: Try `3333` → `8001` → `8002` → `8003` → `8080` → `5000` (avoid 8000, often busy)

Check if port in use: `lsof -i :3333` | Kill process: `kill $(lsof -t -i :3333)`

---

## Benchmarking

### Standalone Benchmark (Current Branch)

```bash
# Basic
benchmark --model deepseek-ai/DeepSeek-V3.2-Exp -tp 8

# With rebuild + custom config
benchmark --model deepseek-ai/DeepSeek-V2-Lite --build --max-model-len 8192 --rates "1,5,10,25"

# Test variants (CUDA graphs, scheduling, etc.)
benchmark --model deepseek-ai/DeepSeek-V3.2-Exp -tp 8 \
    --variants 'base::;full::--compilation-config '"'"'{"cudagraph_mode":"FULL"}'"'"';async::--enable-async-scheduling;eager::--enforce-eager'

# MoE with DBO (requires DP+EP and DeepEP backend)
benchmark --model deepseek-ai/DeepSeek-V2-Lite \
    --server-args "-dp 2 --enable-expert-parallel --enable-dbo --all2all-backend deepep_low_latency"

# With FP8 KV-cache (saves ~50% memory)
benchmark --model deepseek-ai/DeepSeek-V3.2-Exp -tp 8 \
    --server-args "--kv-cache-dtype fp8"

# MTP speculative decoding
benchmark --model <MODEL_WITH_MTP> -tp 8 \
    --variants 'base::;mtp1::--speculative-config '"'"'{"method":"mtp","num_speculative_tokens":1}'"'"';mtp3::--speculative-config '"'"'{"method":"mtp","num_speculative_tokens":3}'"'"''
```

**Key Arguments:**
- `--model` - Model to benchmark (required)
- `-tp N` - Tensor parallel size
- `--max-model-len` - Maximum sequence length
- `--build` - Rebuild vLLM before benchmarking
- `--rates` - Request rates CSV (default: "1,5,10")
- `--run-seconds` - Duration per rate (default: 120)
- `--server-args` - Additional server arguments
- `--branch` - Branch to benchmark (default: current)
- `--variants` - Test multiple configurations (e.g., `'base::;fullcg::-O.cudagraph_mode=FULL'`)
- `--resume` - Skip completed runs
- `--dataset` - Dataset name (default: "random")
- `--random-in`, `--random-out` - Random input/output lengths
- `--label-suffix` - Suffix for result filenames

### Benchmark Compare (Branch Comparison)

Compare performance between two git branches:

**Tip**: To test a GitHub PR, check it out first: `gh pr checkout <pr_number>`

```bash
# Basic comparison
benchmark-compare --model deepseek-ai/DeepSeek-V3.2-Exp \
    --main-ref main --pr-ref my-feature --build-main 1 --build-pr 1 -tp 8

# With variants + shared args
benchmark-compare --model <MODEL> --server-args-base "--max-model-len 8192" \
    --variants 'base::;full::-O.cudagraph_mode=FULL' --main-ref main --pr-ref my-branch

# Resume (skip completed) or test only PR
benchmark-compare --model <MODEL> --resume
benchmark-compare --model <MODEL> --which pr --pr-ref my-branch
```

**Key Arguments:**

| Argument | Description | Example |
|----------|-------------|---------|
| `--model` | Model to benchmark | `deepseek-ai/DeepSeek-V3.2-Exp` |
| `--main-ref` | Git ref for baseline | `main` |
| `--pr-ref` | Git ref for comparison | `my-feature` |
| `--build-main 1` | Rebuild main branch | Required on first run |
| `--build-pr 1` | Rebuild PR branch | Required on first run |
| `--server-args-base` | Args for both branches | `"--max-model-len 8192"` |
| `-tp N` | Tensor parallel size | `-tp 8` |
| `--rates` | Request rates CSV | `"1,5,10,25"` |
| `--run-seconds` | Duration per rate | `120` |
| `--resume` | Skip completed runs | Add for resuming |
| `--ui-mode` | UI mode | `simple` (default for agents) |

### Output Structure

```
results/
├── bench_main_<MODEL>_random/
│   ├── bench_model-<MODEL>_rate-1.0_v-base_*.json
│   ├── bench_model-<MODEL>_rate-5.0_v-base_*.json
│   └── ...
├── bench_pr_<MODEL>_random/
│   └── ...
└── logs/
    ├── server.log
    ├── bench.log
    └── script.log
```

### Low-Level: vllm bench CLI

Requires running server separately. Use `benchmark` script instead for automation.

```bash
vllm bench serve --model <MODEL> --port 3333 --random-input-len 1000 --random-output-len 100 --num-prompts 500
```

---

## GSM8K Evaluations

GSM8K (Grade School Math 8K) evaluates mathematical reasoning on 1319 grade-school math problems.

### Basic Usage

```bash
# Standard evaluation (full dataset)
gsm8k-eval --model deepseek-ai/DeepSeek-V2-Lite

# Quick test (limited questions)
gsm8k-eval --model deepseek-ai/DeepSeek-V2-Lite --limit 100

# With custom config
gsm8k-eval --model deepseek-ai/DeepSeek-V3 \
    --server-args "-tp 8 --max-model-len 8192" \
    --port 8001 \
    --ui-mode simple
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `deepseek-ai/DeepSeek-R1` | Model to evaluate |
| `--port` | `3333` | Server port |
| `--limit` | None (all 1319) | Number of test cases |
| `--num-concurrent` | `256` | Concurrent requests |
| `--server-args` | `""` | Extra server arguments |
| `--ui-mode` | `auto` | UI mode (simple/tui/auto) |

### Output

```
gsm8k-results/
├── <model_name>/
│   ├── results_*.json           # Evaluation results
│   └── samples_gsm8k_*.jsonl    # Per-sample results
└── logs/
    ├── server.log
    ├── eval.log
    └── script.log
```

### Metrics

- **exact_match (strict-match)**: Exact answer matching with strict regex
- **exact_match (flexible-extract)**: Flexible numeric extraction
- Standard error for each metric

---

## GPQA-Diamond Evaluations

GPQA-Diamond is the highest-quality subset of GPQA (Graduate-Level Google-Proof Q&A), containing expert-level questions across physics, biology, and chemistry.

### Basic Usage

```bash
# Standard evaluation (full dataset)
gpqa-diamond-eval --model deepseek-ai/DeepSeek-V2-Lite

# With custom config
gpqa-diamond-eval --model deepseek-ai/DeepSeek-V3 \
    --server-args "-tp 8 --max-model-len 8192" \
    --port 8001 \
    --ui-mode simple
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `deepseek-ai/DeepSeek-R1` | Model to evaluate |
| `--port` | `3333` | Server port |
| `--num-concurrent` | `256` | Concurrent requests |
| `--server-args` | `""` | Extra server arguments |
| `--ui-mode` | `auto` | UI mode (simple/tui/auto) |

### Output

```
gpqa-diamond-results/
├── <model_name>/
│   ├── results_*.json              # Evaluation results
│   └── samples_gpqa_diamond_*.jsonl # Per-sample results
└── logs/
    ├── server.log
    ├── eval.log
    └── script.log
```

### Metrics

- **acc** (accuracy): Percentage of correct answers
- **acc_norm**: Normalized accuracy
- Standard error for each metric

---

## Agent Output Guidelines

### Required Elements

When reporting results, **always include**:
1. **Commands Used** - Extract server and benchmark/eval commands from logs
2. **Results Table** - Metrics in clean tabular format
3. **Configuration** - Model, TP/DP, branch/commit, port
4. **File Locations** - Results directory and log files

### Unified Output Template

```
================================================================================
📊 [RESULTS TYPE: BENCHMARK / COMPARISON / EVALUATION]
================================================================================
Model: <model_name> | Branch: <branch> | Port: 3333
Config: -tp <N> --max-model-len <X>
[For comparisons: MAIN: <commit> | PR: <commit>]
================================================================================

[Benchmark Results]
| Rate | Throughput | TTFT   | TPOT  | E2E Latency |
|------|------------|--------|-------|-------------|
| 1.0  | 0.86 r/s   | 199 ms | 51 ms | 5234 ms     |

[Comparison Results]
| Metric     | MAIN     | PR       | Δ %     |
|------------|----------|----------|---------|
| Throughput | 4.23 r/s | 4.28 r/s | +1.18%  |

[GSM8K Results]
| Metric                    | Score  | Stderr  |
|---------------------------|--------|---------|
| Exact Match (strict)      | 38.67% | ± 1.34% |

[GPQA-Diamond Results]
| Metric     | Score  | Stderr  |
|------------|--------|---------|
| Accuracy   | 45.2%  | ± 2.1%  |

🔧 Commands for Reproduction:
  Server:   vllm serve <model> --host localhost --port 3333 -tp 8
  Client:   vllm bench serve --model <model> --port 3333 --random-input-len 1000 ...
            (or: lm_eval --model local-completions --tasks gsm8k,gpqa_diamond ...)

📁 Results: /path/to/results/ [or /path/to/bench_main_*/ and /path/to/bench_pr_*/]
📋 Logs: /path/to/logs/ (server.log, bench.log, eval.log, script.log)
================================================================================
```

**Guidelines**: Extract commands from logs, use tables for metrics, include file paths, show commits for comparisons, keep format consistent

---

## Profiling

Capture torch profiler traces to analyze vLLM performance and identify bottlenecks.

### Basic Usage

```bash
# Profile model with default settings
profile --model deepseek-ai/DeepSeek-V2-Lite -tp 2

# Custom profiling parameters
profile --model deepseek-ai/DeepSeek-V3 -tp 8 \
    --random-input-len 1000 \
    --random-output-len 100 \
    --num-prompts 50 \
    --profile-dir ./vllm-profiles

# Quick test (small workload)
profile --model deepseek-ai/DeepSeek-V2-Lite -tp 2 \
    --random-input-len 1000 \
    --random-output-len 4 \
    --num-prompts 16
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | Model to profile |
| `-tp` | `1` | Tensor parallel size |
| `--random-input-len` | `1000` | Input sequence length |
| `--random-output-len` | `100` | Output sequence length |
| `--num-prompts` | `50` | Number of prompts to run |
| `--profile-dir` | `./vllm-profiles` | Output directory for traces |
| `--server-args` | `""` | Additional server arguments |
| `--port` | `3333` | Server port |

### Output

```
vllm-profiles/
├── *-rank-0.*.pt.trace.json.gz    # Trace for TP rank 0
├── *-rank-1.*.pt.trace.json.gz    # Trace for TP rank 1 (if TP>1)
├── *-rank-N.*.pt.trace.json.gz    # Trace for TP rank N
├── *.async_llm.*.pt.trace.json.gz # Async LLM engine trace
└── logs/
    ├── server.log                  # Server logs
    ├── bench.log                   # Benchmark logs
    └── script.log                  # Script logs
```

### Analyzing Traces

Use Chrome Trace Viewer or Perfetto to visualize traces:

```bash
# View in Chrome
# 1. Open chrome://tracing in Chrome/Chromium
# 2. Click "Load" and select a .pt.trace.json.gz file

# Or use Perfetto (online)
# Open https://ui.perfetto.dev and load trace file
```

### How It Works

1. **Server Setup**: Starts vLLM server with `VLLM_TORCH_PROFILER_DIR` set
2. **Profiling**: Runs `vllm bench serve --profile` to capture traces
3. **Cleanup**: Stops server and reports trace file locations and sizes
4. **Verification**: Checks that trace files are non-empty (>0 bytes)

### Environment Variables

The profiling script automatically sets:
- `VLLM_TORCH_PROFILER_DIR`: Output directory for traces
- `VLLM_ALLREDUCE_USE_SYMM_MEM=0`: Avoids symmetric memory initialization errors

### Manual Profiling (Low-Level)

For more control, manually start server and run benchmark:

```bash
# Setup
export VLLM_TORCH_PROFILER_DIR=./profiles
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
mkdir -p $VLLM_TORCH_PROFILER_DIR

# Terminal 1: Start server with profiling
vllm serve <MODEL> -tp 8 --port 3333

# Terminal 2: Run benchmark with profiling (wait for server to initialize)
vllm bench serve \
    --model <MODEL> \
    --backend openai \
    --base-url http://localhost:3333 \
    --endpoint /v1/completions \
    --random-input-len 1000 \
    --random-output-len 100 \
    --num-prompts 50 \
    --profile

# Terminal 1: Stop server (Ctrl+C) after benchmark completes
```

### Verification

Check that rank-specific traces exist and are non-empty:

```bash
ls -lh ./vllm-profiles/*-rank-*.pt.trace.json.gz
```

Expected output: Multiple files with sizes >0 bytes (typically 5-20 MB each)

---

## Advanced Configuration

### CUDA Graphs & Eager Mode

| Mode | Description | Memory | Throughput | Use Case |
|------|-------------|--------|------------|----------|
| **PARTIAL (default)** | CUDA graphs for decode only | Medium | Good | General use |
| **FULL** | CUDA graphs for prefill + decode | High | Best | Max throughput |
| **FULL_AND_PIECEWISE** | Full for decode batches, piecewise for mixed batches | High | Best | Max throughput w/ flexibility |
| **PIECEWISE** | Multiple smaller CUDA graphs | Medium | Better | More flexible than FULL |
| **Eager Mode** | No CUDA graphs | Low | Baseline | Debug, profiling |

**Enable CUDA Graph Modes:**
```bash
# Full CUDA graphs
vllm serve <MODEL> --compilation-config '{"cudagraph_mode":"FULL"}'

# Full for decode batches, piecewise for mixed prefill-decode batches (default for v1)
vllm serve <MODEL> --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'

# Piecewise
vllm serve <MODEL> --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
```

**Note**: The `--compilation-config` flag takes JSON. When using in variants, use triple-quoted escaping:
```bash
--variants 'full::--compilation-config '"'"'{"cudagraph_mode":"FULL"}'"'"''
```

**Enable Eager Mode:**
```bash
vllm serve <MODEL> --enforce-eager
```

**Comparison Examples:**
```bash
# CUDA graphs & scheduling
benchmark-compare --model <MODEL> \
    --variants 'base::;full::--compilation-config '"'"'{"cudagraph_mode":"FULL"}'"'"';async::--enable-async-scheduling;eager::--enforce-eager'

# MoE: DBO & all2all backends (DeepSeek-V2-Lite)
benchmark-compare --model deepseek-ai/DeepSeek-V2-Lite \
    --variants 'base::-dp 2 --enable-expert-parallel;dbo::-dp 2 --enable-expert-parallel --enable-dbo --all2all-backend deepep_low_latency;pplx::-dp 2 --enable-expert-parallel --all2all-backend pplx'

# Compare KV-cache quantization (memory vs performance)
benchmark-compare --model <MODEL> -tp 8 \
    --variants 'auto::;fp8::--kv-cache-dtype fp8'
```

### Common Server Arguments

**Memory & Model:**
- `--max-model-len <N>` - Max sequence length
- `--gpu-memory-utilization <0.0-1.0>` - GPU memory fraction (default: 0.9)
- `--max-num-seqs <N>` - Max concurrent sequences
- `--max-num-batched-tokens <N>` - Max batched tokens
- `--kv-cache-dtype <TYPE>` - KV-cache quantization (default: auto)
  - `fp8` - Use FP8 quantization for KV-cache (saves ~50% memory)
  - `fp8_e5m2` - FP8 with e5m2 format
  - `fp8_e4m3` - FP8 with e4m3 format
  - `auto` - Automatic selection based on model
  - Example: `vllm serve <MODEL> --kv-cache-dtype fp8`

**Parallelism:**
- `-tp N` - Tensor parallel size (split layers across GPUs)
- `-dp N --enable-expert-parallel` - Data parallel (MoE models)
- `-dcp N` - Distributed checkpoint parallel size
- `-pp N` - Pipeline parallel size (split stages across GPUs)

**Performance:**
- `--disable-log-stats` - Disable stats logging
- `--no-enable-prefix-caching` - Disable prefix caching
- `--enable-chunked-prefill` - Enable chunked prefill
- `--trust-remote-code` - Trust remote model code

**Scheduling:**
- `--enable-async-scheduling` - Enable async request scheduling
  - Improves throughput by overlapping scheduling with execution
  - Recommended for high-throughput scenarios
  - Example: `vllm serve <MODEL> --enable-async-scheduling`
- `--enable-dbo` - Enable Dual Batch Overlap (DP+EP only)
  - Overlaps MoE all-to-all communication with computation
  - **Only supported with data parallelism + expert parallelism**
  - Requires: `-dp N` (N>1) + `--enable-expert-parallel` + DeepEP backend
  - Example: `vllm serve <MODEL> -dp 2 --enable-expert-parallel --enable-dbo --all2all-backend deepep_low_latency`
- `--all2all-backend <BACKEND>` - Set all2all backend for MoE communication
  - Choose backend optimized for your workload
  - `deepep_low_latency` - Best for decode/low-latency workloads (supports DBO)
  - `deepep_high_throughput` - Best for prefill/high-throughput workloads (supports DBO)
  - `pplx` - PerplexityAI kernels (async)
  - `allgather_reducescatter` - Default, works in most cases
  - Example: `vllm serve <MODEL> --all2all-backend deepep_low_latency`

**Speculative Decoding:**
- `--speculative-config '{"method":"mtp","num_speculative_tokens":N}'` - MTP speculative decoding
  - When user says "mtp 3": use `num_speculative_tokens=3`

### Environment Variables

**Most Common:**

| Variable | Usage | Example |
|----------|-------|---------|
| `VLLM_ATTENTION_BACKEND` | Set attention backend | `FLASH_ATTN_MLA`, `FLASHINFER`, `TRITON_ATTN` |
| `VLLM_ALL2ALL_BACKEND` | Set all2all backend for MoE | `deepep_low_latency`, `pplx`, `allgather_reducescatter` |
| `VLLM_TORCH_PROFILER_DIR` | Profiler output directory | `/tmp/profiles` (use with `--profile`) |
| `VLLM_ALLREDUCE_USE_SYMM_MEM=0` | Disable symm mem | Required for profiling |
| `CUDA_VISIBLE_DEVICES` | Limit GPU visibility | `0,1,2,3` |
| `VLLM_LOGGING_LEVEL` | Debug logging | `DEBUG` |

**Attention backends:** `FLASH_ATTN`, `FLASH_ATTN_MLA`, `FLASHINFER`, `FLASHINFER_MLA`, `TRITON_ATTN`, `TRITON_MLA`, `XFORMERS`, `TORCH_SDPA`

**All2All backends:** See `--all2all-backend` in Common Server Arguments for details. DBO requires DeepEP backends.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Server won't start | Check GPU memory (`nvidia-smi`), kill zombies (`pkill -9 -f "vllm"`), try alternate port |
| Port already in use | Check with `lsof -i :3333`, kill process, or use `--port 8001` |
| Out of memory | Reduce `--max-model-len`, increase `-tp`, use `--enforce-eager`, disable prefix caching |
| Benchmark fails | Check `results/logs/server.log`, verify server health (`curl http://localhost:3333/health`) |
| Results not found | Check output directory structure, look in model-specific subdirectories |
| Build fails | Check for CUDA errors, kill zombie processes, ensure GPUs available |
| Slow server startup | Large models take 5+ minutes, check logs for progress, wait patiently |

**Cleanup Commands:**
```bash
# Kill all vLLM processes
pkill -9 -f "api_server" 2>/dev/null || true
pkill -9 -f "benchmarks" 2>/dev/null || true
pkill -9 -f "VLLM::" 2>/dev/null || true

# Check GPU status
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Check processes using GPUs
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

---

**End of Guide**
