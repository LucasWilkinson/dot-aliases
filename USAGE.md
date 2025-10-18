# vLLM Test Infrastructure - Quick Usage Guide

## Installation

After running the installer, the scripts are automatically available as aliases. Just install the Python dependencies:

```bash
vllm-test-infra-install
```

This will install `textual` and `requests` packages.

## Using the Aliases

The aliases work from **any directory** and point to the installed scripts.

### GSM8K Evaluation

```bash
# Basic usage (uses defaults)
gsm8k-eval --model deepseek-ai/DeepSeek-R1 --limit 100

# With custom server args
gsm8k-eval \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --limit 500 \
  --server-args "-tp 2" \
  --port 3333

# Simple mode (no TUI, just stdout)
gsm8k-eval \
  --model deepseek-ai/DeepSeek-R1 \
  --ui-mode simple \
  --limit 100
```

### Benchmark Comparison

```bash
# Compare main vs PR branch
benchmark-compare \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --main-ref main \
  --pr-ref my-feature-branch \
  --rates 1,5,10 \
  --build-pr 1

# Test multiple variants
benchmark-compare \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --variants 'base::;fullcg::-O {"full_cuda_graph":true}' \
  --rates 1,10,100 \
  -tp 2

# Resume interrupted benchmark
benchmark-compare \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --resume \
  --which pr

# With custom output directory
benchmark-compare \
  --model deepseek-ai/DeepSeek-V3 \
  --out-base ./my-benchmarks \
  --rates 1,5,10,25 \
  -tp 8
```

## Common Options

### GSM8K Eval Options

- `--model MODEL` - Model to evaluate (default: deepseek-ai/DeepSeek-R1)
- `--port PORT` - Server port (default: 3333)
- `--limit N` - Limit number of test cases
- `--num-concurrent N` - Concurrent requests (default: 256)
- `--server-args "ARGS"` - Additional vllm serve arguments
- `--ui-mode {tui,simple,auto}` - UI mode (default: auto)
- `--out-base DIR` - Output directory (default: ./gsm8k-results)

### Benchmark Compare Options

- `--model MODEL` - Model to benchmark
- `--rates RATES` - Comma-separated request rates (default: 1,5,10,25,100)
- `--variants SPEC` - Variant specification (see below)
- `--which {both,main,pr}` - Which branches to run (default: both)
- `--main-ref REF` - Git ref for main branch (default: main)
- `--pr-ref REF` - Git ref for PR branch
- `--build-main 1` - Rebuild main branch after checkout
- `--build-pr 1` - Rebuild PR branch after checkout
- `--resume` - Skip existing results
- `-tp N` - Tensor parallel size (default: 1)
- `--ui-mode {tui,simple,auto}` - UI mode (default: auto)

### Variant Specification Format

```bash
# Simple variants (label::args)
--variants 'base::;fullcg::-O {"full_cuda_graph":true}'

# Variants with environment variables (label::env:K=V,K2=V2::args)
--variants 'piece::env:VLLM_USE_PIECEWISE=1::--compilation-config {"cudagraph_mode":"PIECEWISE"}'

# Separate variants for main and PR
--variants-main 'std::'
--variants-pr 'experimental::--some-flag'
```

## UI Mode

Scripts use **simple mode** (clean stdout with organized log files):
- All output is timestamped and logged clearly to stdout
- Subprocess output automatically saved to log files
- Works everywhere: interactive terminals, scripts, CI/CD
- No threading complexity or signal handler issues

Log files are organized in `<out-base>/logs/`:
- `server.log` - vLLM server output
- `eval.log` / `bench.log` - Test tool output  
- `script.log` - Main script execution log

**Note**: The `--ui-mode` flag still exists but TUI mode is currently disabled (would require significant architectural changes). Simple mode provides all necessary functionality.

## Examples by Use Case

### Quick GSM8K Test
```bash
# Test on first 100 questions
gsm8k-eval --model meta-llama/Meta-Llama-3-8B-Instruct --limit 100
```

### Full GSM8K Evaluation
```bash
# Run all 1319 questions
gsm8k-eval --model deepseek-ai/DeepSeek-R1
```

### Compare Performance Before/After Changes
```bash
# Make changes to vLLM, then:
benchmark-compare \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --main-ref main \
  --pr-ref HEAD \
  --rates 1,5,10,25 \
  --build-pr 1 \
  --repo-dir ~/code/vllm
```

### Test Different Configurations
```bash
benchmark-compare \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --variants 'nocache::--no-enable-prefix-caching;cache::' \
  --rates 1,10,100
```

### Run in Background / CI
```bash
# Use simple mode for non-interactive environments
gsm8k-eval \
  --model deepseek-ai/DeepSeek-R1 \
  --ui-mode simple \
  --limit 500 > gsm8k.log 2>&1 &
```

## Logs and Results

### GSM8K Eval
- Results: `./gsm8k-results/results_<model>.json`
- Logs: `./gsm8k-results/logs/`
  - `server.log` - vLLM server output
  - `eval.log` - lm_eval output
  - `script.log` - Script execution log

### Benchmark Compare
- Results: `./results/bench_main_<model>_<dataset>/` and `./results/bench_pr_<model>_<dataset>/`
- Logs: `./results/logs/`
  - `server.log` - vLLM server output
  - `bench.log` - Benchmark client output
  - `script.log` - Script execution log

## Troubleshooting

### Scripts not found
```bash
# Reload aliases
source ~/.bashrc  # or ~/.zshrc
```

### Dependencies not installed
```bash
vllm-test-infra-install
```

### Port already in use
```bash
# Use different port
gsm8k-eval --port 8001
benchmark-compare --port 8001
```

### GPU allocation failed
- Scripts automatically use `chg` to reserve GPUs if available
- GPU count is auto-detected from `-tp` args
- Check `chg` is installed: `which chg`

### Server fails to start
- Check logs in `<out-base>/logs/server.log`
- Scripts fast-fail on OOM, assertions, CUDA errors
- Try with `--ui-mode simple` to see full output

## Getting Help

```bash
# Show all options
gsm8k-eval --help
benchmark-compare --help

# See full documentation
cat ~/.config/aliases/dot-aliases/python/README.md
```

## Alias Reference

- `gsm8k-eval` → `~/.config/aliases/dot-aliases/scripts/gsm8k_eval.py`
- `benchmark-compare` → `~/.config/aliases/dot-aliases/scripts/benchmark_compare.py`
- `vllm-test-infra-install` → Install Python dependencies

