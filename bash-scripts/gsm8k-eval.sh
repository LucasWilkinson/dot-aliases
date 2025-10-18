#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# GSM8K Evaluation Script with vLLM server and lm_eval
# ============================================================

# Defaults
VENV="${VENV:-.venv}"
VENV_PY="$VENV/bin/python"
VENV_VLLM="$VENV/bin/vllm"

HOST="${HOST:-localhost}"
PORT="${PORT:-3333}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1}"
TERSE_MODEL_NAME="${TERSE_MODEL_NAME:-$(basename $MODEL)}"

# Eval config
LIMIT="${LIMIT:-}"
NUM_CONCURRENT="${NUM_CONCURRENT:-256}"
BATCH_SIZE="${BATCH_SIZE:-auto}"

# Server args (space-separated string)
SERVER_ARGS="${SERVER_ARGS:-}"

# Output
OUT_BASE="${OUT_BASE:-./gsm8k-results}"
RESULTS_FILE="${RESULTS_FILE:-$OUT_BASE/results_${TERSE_MODEL_NAME}.json}"
LOG_DIR="${LOG_DIR:-$OUT_BASE/logs}"
SERVER_LOG="${SERVER_LOG:-$LOG_DIR/server.log}"
EVAL_LOG="${EVAL_LOG:-$LOG_DIR/eval.log}"
SCRIPT_LOG="${SCRIPT_LOG:-$LOG_DIR/script.log}"

# tmux
TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-gsm8k-eval-$$}"
RUN_INSIDE_TMUX="${RUN_INSIDE_TMUX:-0}"
WAIT_TOKEN="gsm8k_eval_done_$$"
WAIT_TOKEN_ESCAPED=$(printf '%q' "$WAIT_TOKEN")

# GPU launcher prefix (if using chg)
GPU_PREFIX="${GPU_PREFIX:-}"

# ============================================================
# Usage
# ============================================================
usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Run GSM8K evaluation using vLLM server and lm_eval.

Options:
  --model MODEL              Model to evaluate (default: deepseek-ai/DeepSeek-R1)
  --port PORT                Server port (default: 3333)
  --limit N                  Limit number of test cases (default: no limit)
  --num-concurrent N         Number of concurrent requests (default: 256)
  --batch-size N             Batch size for lm_eval (default: auto)
  --server-args "ARGS"       Additional vllm serve arguments (quoted string)
  --out-base PATH            Output directory (default: ./gsm8k-results)
  --gpu-prefix "CMD"         GPU launcher prefix (e.g., "chg run -g 2 --")
  -h, --help                 Show this help

Example:
  $0 --model deepseek-ai/DeepSeek-R1 --limit 100 --server-args "--tensor-parallel-size 2"
  $0 --model meta-llama/Meta-Llama-3-8B-Instruct --gpu-prefix "chg run -g 1 --"
EOF
  exit 0
}

# ============================================================
# Argument parsing
# ============================================================
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --num-concurrent) NUM_CONCURRENT="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --server-args) SERVER_ARGS="$2"; shift 2 ;;
    --out-base) OUT_BASE="$2"; shift 2 ;;
    --gpu-prefix) GPU_PREFIX="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# Recompute derived paths
TERSE_MODEL_NAME="$(basename $MODEL)"
RESULTS_FILE="$OUT_BASE/results_${TERSE_MODEL_NAME}.json"
LOG_DIR="$OUT_BASE/logs"
SERVER_LOG="$LOG_DIR/server.log"
EVAL_LOG="$LOG_DIR/eval.log"
SCRIPT_LOG="$LOG_DIR/script.log"

# ============================================================
# Setup tmux wrapper (if not already inside)
# ============================================================
if [[ $RUN_INSIDE_TMUX -eq 0 ]]; then
  ORIG_ARGS=()
  for arg in "$@"; do
    ORIG_ARGS+=("$(printf '%q' "$arg")")
  done
  ORIG_ARGS_ESCAPED="${ORIG_ARGS[*]}"
  SCRIPT_PATH="$(readlink -f "$0")"
  VENV_ESCAPED=$(printf '%q' "$VENV")
  SERVER_LOG_ESCAPED=$(printf '%q' "$SERVER_LOG")
  EVAL_LOG_ESCAPED=$(printf '%q' "$EVAL_LOG")
  
  echo "Starting tmux session: $TMUX_SESSION_NAME"
  
  # Create directories
  mkdir -p "$LOG_DIR"
  mkdir -p "$(dirname "$RESULTS_FILE")"
  
  # Initialize logs
  : > "$SERVER_LOG"
  : > "$EVAL_LOG"
  : > "$SCRIPT_LOG"
  
  # Create tmux session with 3 panes:
  # Top-left: server log
  # Top-right: eval log
  # Bottom: script execution (full width)
  tmux new-session -d -s "$TMUX_SESSION_NAME" "tail -n +1 -F '$SERVER_LOG'"
  
  # Ctrl-C: fan out to all panes
  tmux bind-key -n C-c "run-shell \"for p in \$(tmux list-panes -t '$TMUX_SESSION_NAME' -F '#{pane_id}'); do tmux send-keys -t \\\$p C-c; done; sleep 0.2; tmux wait-for -S $WAIT_TOKEN_ESCAPED\""
  
  # Enable mouse and scrollback
  tmux set-option -t "$TMUX_SESSION_NAME" mouse on
  tmux set-option -t "$TMUX_SESSION_NAME" history-limit 200000
  
  # Top-right: eval log
  tmux split-window -h -t "$TMUX_SESSION_NAME":0.0 "tail -n +1 -F '$EVAL_LOG'"
  
  # Bottom: script execution
  tmux select-pane -L -t "$TMUX_SESSION_NAME"
  tmux split-window -v -t "$TMUX_SESSION_NAME" \
    "RUN_INSIDE_TMUX=1 VENV=$VENV_ESCAPED bash '$SCRIPT_PATH' $ORIG_ARGS_ESCAPED; tmux wait-for -S $WAIT_TOKEN_ESCAPED"
  
  # Make bottom pane larger
  tmux resize-pane -t "$TMUX_SESSION_NAME":0.2 -y 15
  
  # Focus bottom pane
  tmux select-pane -t "$TMUX_SESSION_NAME":0.2
  
  # Auto-close session when script completes
  tmux set-hook -t "$TMUX_SESSION_NAME" session-closed "run-shell 'tmux wait-for -S $WAIT_TOKEN_ESCAPED'"
  
  # Attach to the session
  tmux attach-session -t "$TMUX_SESSION_NAME"
  
  # Wait for completion signal
  tmux wait-for "$WAIT_TOKEN"
  
  echo "Evaluation complete. Results saved to: $RESULTS_FILE"
  exit 0
fi

# ============================================================
# Main execution (runs inside tmux bottom pane)
# ============================================================
exec > >(tee -a "$SCRIPT_LOG") 2>&1

echo "=========================================="
echo "GSM8K Evaluation"
echo "=========================================="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Limit: ${LIMIT:-none}"
echo "Concurrent: $NUM_CONCURRENT"
echo "Server args: ${SERVER_ARGS:-none}"
echo "Results: $RESULTS_FILE"
echo "=========================================="
echo ""

# Activate venv
if [[ -f "$VENV_PY" ]]; then
  echo "Activating venv: $VENV"
  source "$VENV/bin/activate"
else
  echo "=========================================="
  echo "❌ ERROR: venv not found at $VENV"
  echo "=========================================="
  echo "Python binary expected at: $VENV_PY"
  echo "Current directory: $PWD"
  echo ""
  echo "Please ensure:"
  echo "  1. You're running from a directory with .venv/"
  echo "  2. Or set VENV environment variable to correct path"
  echo "=========================================="
  sleep 3  # Give time to see the error before tmux closes
  exit 1
fi

# Check for lm_eval
if ! command -v lm_eval &> /dev/null; then
  echo "=========================================="
  echo "❌ ERROR: lm_eval not found"
  echo "=========================================="
  echo "Install with: pip install lm-eval"
  echo "=========================================="
  sleep 3  # Give time to see the error before tmux closes
  exit 1
fi

# ============================================================
# Cleanup function
# ============================================================
cleanup() {
  echo ""
  echo "Cleaning up..."
  if [[ -n "${SERVER_PID:-}" ]]; then
    echo "Killing vLLM server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
  fi
  
  # Additional cleanup
  pkill -9 -f "vllm.*serve.*$PORT" 2>/dev/null || true
  sleep 2
}

trap cleanup EXIT INT TERM

# ============================================================
# Start vLLM server
# ============================================================
echo "Starting vLLM server..."
echo "Command: ${GPU_PREFIX} vllm serve $MODEL --port $PORT $SERVER_ARGS"
echo ""

SERVER_CMD="$VENV_VLLM serve $MODEL --port $PORT $SERVER_ARGS"

if [[ -n "$GPU_PREFIX" ]]; then
  # Parse GPU_PREFIX into array
  read -ra GPU_PREFIX_ARRAY <<< "$GPU_PREFIX"
  "${GPU_PREFIX_ARRAY[@]}" $SERVER_CMD >> "$SERVER_LOG" 2>&1 &
else
  $SERVER_CMD >> "$SERVER_LOG" 2>&1 &
fi

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to be ready..."
MAX_WAIT=300  # 5 minutes
WAITED=0
while [[ $WAITED -lt $MAX_WAIT ]]; do
  if curl -s "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
    echo "✅ Server is ready!"
    break
  fi
  
  # Check if server process died
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ ERROR: Server process died!"
    echo "Check server log: $SERVER_LOG"
    exit 1
  fi
  
  sleep 5
  WAITED=$((WAITED + 5))
  echo "  ... still waiting ($WAITED/${MAX_WAIT}s)"
done

if [[ $WAITED -ge $MAX_WAIT ]]; then
  echo "❌ ERROR: Server failed to start within ${MAX_WAIT}s"
  echo "Check server log: $SERVER_LOG"
  exit 1
fi

# ============================================================
# Run GSM8K evaluation
# ============================================================
echo ""
echo "=========================================="
echo "Running GSM8K evaluation..."
echo "=========================================="
echo ""

# Build lm_eval command
EVAL_CMD="lm_eval --model local-completions"
EVAL_CMD="$EVAL_CMD --model_args model=$MODEL,base_url=http://${HOST}:${PORT}/v1/completions,num_concurrent=$NUM_CONCURRENT,batch_size=$BATCH_SIZE"
EVAL_CMD="$EVAL_CMD --tasks gsm8k"
EVAL_CMD="$EVAL_CMD --output_path $(dirname "$RESULTS_FILE")"
EVAL_CMD="$EVAL_CMD --log_samples"

if [[ -n "$LIMIT" ]]; then
  EVAL_CMD="$EVAL_CMD --limit $LIMIT"
fi

echo "Command: $EVAL_CMD"
echo ""

# Run evaluation (output goes to both eval log and stdout)
eval "$EVAL_CMD" 2>&1 | tee -a "$EVAL_LOG"

EVAL_STATUS=$?

if [[ $EVAL_STATUS -eq 0 ]]; then
  echo ""
  echo "=========================================="
  echo "✅ Evaluation complete!"
  echo "=========================================="
  echo "Results: $RESULTS_FILE"
  echo "Server log: $SERVER_LOG"
  echo "Eval log: $EVAL_LOG"
else
  echo ""
  echo "=========================================="
  echo "❌ Evaluation failed with status: $EVAL_STATUS"
  echo "=========================================="
  exit $EVAL_STATUS
fi

# Keep the pane open for a moment so user can see results
sleep 5

