# vLLM Test Infrastructure

A reusable Python infrastructure for running vLLM benchmarks and evaluations with robust server management, process monitoring, logging, and user interfaces.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Or using uv:

```bash
uv pip install -r requirements.txt
```

## Components

### Core Infrastructure (`vllm_test_infra/`)

- **`server.py`**: VLLMServer class for managing vLLM server lifecycle
  - Auto-detects GPU requirements from `-tp`/`-dp` args
  - Uses `chg` to reserve GPUs when available
  - Monitors logs for fast-fail on errors (OOM, assertions, CUDA errors)
  - Graceful shutdown with INT → TERM → KILL progression

- **`process.py`**: ProcessManager for running and monitoring subprocesses
  - Background process execution with logging
  - Signal handling (Ctrl-C propagation)
  - Automatic cleanup on exit
  - Zombie process detection and cleanup

- **`logging.py`**: LogManager for organizing log files
  - Structured logging with timestamps
  - Log file tailing and searching
  - Output redirection support

- **`ui.py`**: UIManager for user interfaces
  - **Simple mode** (default): Clean stdout logging with timestamps
  - **TUI mode** (opt-in): Textual-based multi-pane interface
  - Proper threading: Textual in main thread, work in background
  - Thread-safe queue communication
  - Automatic fallback to simple mode on TUI failure

- **`git.py`**: GitManager for git operations
  - Checkout branches/commits
  - Pull latest changes (with smart branch detection)
  - Build vLLM extensions

- **`config.py`**: Config for environment and path management
  - Virtual environment activation
  - Environment variable management
  - Path normalization

- **`utils.py`**: Common utilities
  - Timestamp formatting
  - GPU memory checking
  - Zombie process cleanup
  - Variant spec parsing

## Scripts

### `gsm8k_eval.py`

Run GSM8K evaluation using vLLM server and lm_eval.

```bash
./scripts/gsm8k_eval.py --model deepseek-ai/DeepSeek-R1 --limit 100
```

Options:
- `--model MODEL`: Model to evaluate
- `--port PORT`: Server port (default: 3333)
- `--limit N`: Limit number of test cases
- `--num-concurrent N`: Concurrent requests (default: 256)
- `--server-args "ARGS"`: Additional vllm serve arguments
- `--ui-mode {tui,simple,auto}`: UI mode

### `benchmark_compare.py`

Compare vLLM performance across branches with multiple variants and rates.

```bash
./scripts/benchmark_compare.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --rates 1,5,10,25 \
  --variants 'base::;fullcg::-O {"full_cuda_graph":true}'
```

Options:
- `--model MODEL`: Model to benchmark
- `--rates RATES`: Comma-separated request rates
- `--variants SPEC`: Variant specification (see below)
- `--which {both,main,pr}`: Which branches to run
- `--build-main/--build-pr`: Rebuild after checkout
- `--resume`: Skip existing results
- `--ui-mode {tui,simple,auto}`: UI mode

#### Variant Specification

Variants allow testing different configurations:

```bash
# Simple variants
--variants 'base::;fullcg::-O {"full_cuda_graph":true}'

# Variants with environment variables
--variants 'piece::env:VLLM_USE_PIECEWISE=1::--compilation-config {"cudagraph_mode":"PIECEWISE"}'

# Separate variants for main and PR
--variants-main 'std::'
--variants-pr 'piece::env:CUDA_VISIBLE_DEVICES=0::--compilation-config {"cudagraph_mode":"PIECEWISE"}'
```

Format: `label::args` or `label::env:K=V,K2=V2::args`

## Features

### Robust Server Management

- **Fast-fail error detection**: Monitors logs for OOM, assertions, CUDA errors
- **GPU reservation**: Uses `chg` to reserve GPUs based on parallelism settings
- **Graceful shutdown**: Proper process cleanup with signal escalation

### Process Monitoring

- **Signal handling**: Proper Ctrl-C propagation to all subprocesses
- **Zombie cleanup**: Always checks for and kills stray processes on exit
- **Timeouts**: Optional timeouts on all long-running operations

### User Interfaces

**TUI Mode** (with Textual):
- Multi-pane layout showing live logs
- Scrollable with mouse support
- Server, eval, and script logs in separate panes

**Simple Mode** (for AI agents):
- Only main output to stdout
- All subprocess logs to files
- No interactive elements

### Logging

- Organized log directory structure
- Automatic timestamps
- Log tailing and error pattern detection
- Easy post-execution analysis

## Usage Examples

### Run GSM8K evaluation

```bash
# Basic evaluation
./scripts/gsm8k_eval.py --model deepseek-ai/DeepSeek-R1

# With limits and custom args
./scripts/gsm8k_eval.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --limit 500 \
  --server-args "-tp 2" \
  --ui-mode simple
```

### Compare benchmark performance

```bash
# Compare main vs PR branch
./scripts/benchmark_compare.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --main-ref main \
  --pr-ref my-feature-branch \
  --rates 1,5,10 \
  --build-pr 1

# Test multiple variants
./scripts/benchmark_compare.py \
  --model deepseek-ai/DeepSeek-V3 \
  --variants 'base::;fullcg::-O {"full_cuda_graph":true};piece::--compilation-config {"cudagraph_mode":"PIECEWISE"}' \
  --rates 1,10,100 \
  -tp 8

# Resume interrupted benchmark
./scripts/benchmark_compare.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --resume \
  --which pr
```

## Extending the Infrastructure

To create a new test script:

1. Import the infrastructure components:
```python
from python.vllm_test_infra import (
    Config, LogManager, ProcessManager, 
    VLLMServer, UIManager
)
```

2. Setup managers:
```python
config = Config(venv_path=".venv")
log_manager = LogManager("./logs")
log_manager.setup()
process_manager = ProcessManager(log_manager)
ui_manager = UIManager(log_manager)
```

3. Start server:
```python
server = VLLMServer(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    log_manager=log_manager,
    process_manager=process_manager
)

with server:
    server.start(args="-tp 2")
    server.wait_for_ready()
    
    # Run your tests here
```

4. Run commands:
```python
result = process_manager.run(
    name="my_test",
    command=["vllm", "bench", "..."],
    log_file="test"
)
```

## Architecture

```
python/vllm_test_infra/
├── __init__.py         # Package exports
├── server.py           # Server lifecycle management
├── process.py          # Subprocess execution & monitoring
├── logging.py          # Log file organization
├── ui.py              # User interface (TUI/simple)
├── git.py             # Git operations
├── config.py          # Configuration management
└── utils.py           # Common utilities

scripts/
├── gsm8k_eval.py      # GSM8K evaluation script
└── benchmark_compare.py  # Benchmark comparison script
```

## UI Mode

Scripts use **simple mode** by default for reliability:
- Clean, timestamped stdout logging
- All subprocess output saved to organized log files
- Works in interactive terminals, scripts, CI/CD, and with AI agents
- No threading complexity or signal handler issues
- Proven reliable in all environments

### TUI Mode (Optional, Experimental)

Enable TUI mode with `--ui-mode tui` for a live multi-pane interface:
- Real-time log display in separate panes (server, eval/bench, script)
- Powered by Textual with proper worker thread architecture
- Work runs in background thread, TUI in main thread
- Queue-based communication for thread safety
- Clean shutdown on completion or Ctrl-C

**Architecture:**
```
Main Thread:    Textual TUI (display updates)
Worker Thread:  Server + evaluation work
Communication:  Thread-safe queues
```

**When to use:**
- Interactive development and debugging
- Want live log updates in single window
- Have Textual installed (`pip install textual`)

**When to use simple mode:**
- Running in scripts or CI/CD
- AI agent execution
- No TTY available
- Prefer proven reliability

**Note:** Simple mode is always the safe fallback. If TUI fails to initialize, it automatically falls back to simple mode.

## Requirements

- Python 3.8+
- requests >= 2.31.0
- vLLM (in venv)
- **Optional:** textual >= 0.47.0 (for TUI mode)
- **Optional:** chg (for GPU reservation)
- **Optional:** lm_eval (for GSM8K evaluations)

