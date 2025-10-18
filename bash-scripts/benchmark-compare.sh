#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Defaults (override via CLI; see --help)
# ============================================================
VENV="${VENV:-.venv}"
VENV_PY="$VENV/bin/python"
VENV_VLLM="$VENV/bin/vllm"

# GPU launcher prefix (array). Example default uses chg to pin a single GPU:
GPU_PREFIX_DEFAULT=(chg run -g 1 --)
GPU_PREFIX=("${GPU_PREFIX[@]:-${GPU_PREFIX_DEFAULT[@]}}")

HOST="${HOST:-localhost}"
PORT="${PORT:-3333}"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
TERSE_MODEL_NAME="${TERSE_MODEL_NAME:-llama3_8b}"

# Dataset config
DATASET="${DATASET:-random}"
RANDOM_IN="${RANDOM_IN:-1000}"
RANDOM_OUT="${RANDOM_OUT:-100}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0}"

# Traffic
RATES_CSV="${RATES_CSV:-1,5,10,25,100}"
RUN_SECONDS_PER_RATE="${RUN_SECONDS_PER_RATE:-120}"

# Parallelism
TP="${TP:-1}"

# Branch / refs
PR_BRANCH="${PR_BRANCH:-full_cudagraph_FA2_FlashInfer}"
MAIN_REF="${MAIN_REF:-}"     # If empty, defaults to 'main' at runtime
PR_REF="${PR_REF:-}"         # If empty, defaults to PR_BRANCH

# Build toggles
BUILD_MAIN="${BUILD_MAIN:-0}"
BUILD_PR="${BUILD_PR:-0}"

# Which sides to run
WHICH="${WHICH:-both}"       # both|main|pr

# Git repo directory and pull behavior
REPO_DIR="${REPO_DIR:-.}"
PULL_LATEST_MAIN="${PULL_LATEST_MAIN:-0}"
PULL_LATEST_PR="${PULL_LATEST_PR:-0}"

# Resumability
RESUME="${RESUME:-0}"
RERUN_MAIN="${RERUN_MAIN:-0}"
RERUN_PR="${RERUN_PR:-0}"
OVERWRITE="${OVERWRITE:-0}"

# Output base (everything derives from here)
OUT_BASE="${OUT_BASE:-./results}"
RESULTS_DIR_MAIN="${RESULTS_DIR_MAIN:-$OUT_BASE/bench_main_${TERSE_MODEL_NAME}_${DATASET}}"
RESULTS_DIR_PR="${RESULTS_DIR_PR:-$OUT_BASE/bench_pr_${TERSE_MODEL_NAME}_${DATASET}}"
PLOTS_DIR="${PLOTS_DIR:-$OUT_BASE/plots_${DATASET}_compact_max10}"
LOG_DIR="${LOG_DIR:-$OUT_BASE/logs}"

# Logging files
SERVER_LOG="${SERVER_LOG:-$LOG_DIR/server.log}"   # server stdout/stderr
BENCH_LOG="${BENCH_LOG:-$LOG_DIR/bench.log}"      # vllm bench serve + client headers
SCRIPT_LOG="${SCRIPT_LOG:-$LOG_DIR/script.log}"   # whole script output
SUMMARY_TSV="${SUMMARY_TSV:-$LOG_DIR/summary.tsv}"

# tmux
TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-vllm-bench-$$}"
# Removed bottom pane height setting

# Variants:
#   Format entries (semicolon-separated):
#     1) label::args
#     2) label::env:K=V,K2=V2::args
#   Examples:
#     `--variants-pr   'piece::env:CUDA_VISIBLE_DEVICES=0::--compilation-config {"cudagraph_mode":"PIECEWISE"};full::--compilation-config {"cudagraph_mode":"FULL"}'`
#     --variants-main 'std::;fullcg::-O {"full_cuda_graph":true}'
#     --variants      'std::;fullcg::-O {"full_cuda_graph":true}'  # applies to both MAIN and PR unless overridden
VARIANTS_MAIN_SPEC="${VARIANTS_MAIN_SPEC:-}"
VARIANTS_PR_SPEC="${VARIANTS_PR_SPEC:-}"
VARIANTS_BOTH_SPEC="${VARIANTS_BOTH_SPEC:-}"
VARIANTS_MAIN_SET=0
VARIANTS_PR_SET=0

# Baseline server-args:
#  - BASE: applied to *all* variants and both branches
#  - MAIN/PR: used only when no variants are supplied for that side
SERVER_ARGS_BASE_STR=${SERVER_ARGS_BASE_STR:-""}
SERVER_ARGS_MAIN_STR=${SERVER_ARGS_MAIN_STR:-""}
SERVER_ARGS_PR_STR=${SERVER_ARGS_PR_STR:-""}

# Always-on server args (appended for ALL runs, per your request)
ALWAYS_SERVER_ARGS=(--no-enable-prefix-caching --disable-log-stats --trust-remote-code)

# Plotting
DO_PLOT="${DO_PLOT:-0}"

# Optional filename suffix (e.g., host tag)
LABEL_SUFFIX="${LABEL_SUFFIX:-}"

# Summary options
SUMMARY_ORDER="${SUMMARY_ORDER:-rate}"       # rate|branch
SUMMARY_PER_VARIANT="${SUMMARY_PER_VARIANT:-0}"  # 0|1

# ============================================================
# CLI parsing
# ============================================================
print_help() {
  cat <<'EOF'
Usage: run_benchmark.sh [options]

General:
  --venv PATH                         Virtualenv path (default: .venv)
  --host HOST                         Host (default: localhost)
  --port PORT                         Port (default: 3333)
  --gpu-prefix "cmd ... --"           GPU prefix array (default: chg run -g 1 --)

Model / Data:
  --model HF_OR_PATH                  Model (HF id or local)
  --terse-name NAME                   Short name for filenames (e.g., llama3_8b)
  --dataset NAME                      Dataset (default: random)
  --random-in N                       Random input len (default: 1000)
  --random-out N                      Random output len (default: 100)
  --random-range-ratio X              (default: 0)

Parallelism:
  -tp N                               Tensor-parallel size (default: 1)

Rates / Durations:
  --rates "1,5,10"                    Request rates CSV (default: 1,5,10,25,100)
  --run-seconds N                     Seconds per rate (default: 120)

Git / Build:
  --main-ref REF_OR_HASH              Checkout this ref for MAIN (default: main)
  --pr-ref   REF_OR_HASH              Checkout this ref for PR   (default: value of --pr-branch)
  --pr-branch NAME                    PR branch fallback (default: full_cudagraph_FA2_FlashInfer)
  --build-main 0|1                    Run setup.py build on MAIN (default: 0)
  --build-pr   0|1                    Run setup.py build on PR   (default: 0)
  --repo-dir DIR                      Git repository directory (default: current directory)
  --pull-latest                       Pull latest changes for MAIN and PR
  --pull-latest-main                  Pull latest changes for MAIN
  --pull-latest-pr                    Pull latest changes for PR
  --resume                            Resume and only run missing (branch,variant,rate) combos
  --re-run-pr                         With --resume, re-run all PR variants regardless of existing results
  --re-run-main                       With --resume, re-run all MAIN variants regardless of existing results
  --which both|main|pr                Which side(s) to run (default: both)

Output / Paths:
  --out-base DIR                      Base output directory
  --results-main DIR                  Results dir for MAIN
  --results-pr DIR                    Results dir for PR
  --plots DIR                         Plots dir
  --log-dir DIR                       Log dir
  --label-suffix STR                  Suffix for result filenames

Server Args:
  --server-args-base "..."            Args added to EVERY server run (all variants, both sides)
  --server-args-main "..."            Extra args for MAIN server (used if no variants)
  --server-args-pr   "..."            Extra args for PR server   (used if no variants)

Sweep Variants (preferred):
  --variants       'label::args;label2::args2'  (sets both MAIN and PR)
  --variants-main  'label::args;label2::args2'  (overrides MAIN only)
  --variants-pr    'label::args;label2::args2'  (overrides PR only)
    - Each entry is one of:
        label::args
        label::env:K=V,K2=V2::args      # per-variant env vars (server process only)
    - Examples:
        --variants-pr 'piece::env:CUDA_VISIBLE_DEVICES=0::--compilation-config {"cudagraph_mode":"PIECEWISE"};full::--compilation-config {"cudagraph_mode":"FULL"}'
        --variants-main 'std::;fullcg::-O {"full_cuda_graph":true}'
        --variants 'std::;fullcg::-O {"full_cuda_graph":true}'

Plotting:
  --plot 0|1                          Plot at end (default: 0)
  
  Summary / Tables:
    --summary-order rate|branch       Order rows by request-rate first (default: rate) or by branch first
    --summary-per-variant             If present, print one table per variant

tmux:
  (auto layout; no fixed bottom pane height)

Other:
  --help                              Show this help

Notes:
- Always includes: --no-enable-prefix-caching --disable-log-stats --trust-remote-code
- No "mode" concept; encode cudagraph behavior in variant args (e.g., -O {"full_cuda_graph":true} or --compilation-config {"cudagraph_mode":"FULL"})
- Summary table (TSV) at logs/summary.tsv with columns:
    branch, variant, rate, req/s, median TTFT (ms), median TPOT (ms)
EOF
}

# Helper to ensure flags that require a value actually get one
ensure_has_value() {
  local opt="$1" val="${2-}"
  if [[ -z "$val" ]]; then
    echo "Error: $opt requires a value" >&2
    print_help
    exit 2
  fi
}

# Preserve original CLI args so we can forward them into tmux
ORIG_ARGS=("$@")
# Shell-escaped originals for safe embedding into a tmux command string
ORIG_ARGS_ESCAPED=$(printf '%q ' "${ORIG_ARGS[@]:-}")
VENV_ESCAPED=$(printf '%q' "$VENV")

GPU_PREFIX_CLI=0
VENV_CLI=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -tp) ensure_has_value "$1" "${2-}"; TP="$2"; shift 2;;
    --tp) ensure_has_value "$1" "${2-}"; TP="$2"; shift 2;;
    --venv) ensure_has_value "$1" "${2-}"; VENV="$2"; VENV_CLI=1; shift 2;;
    --host) ensure_has_value "$1" "${2-}"; HOST="$2"; shift 2;;
    --port) ensure_has_value "$1" "${2-}"; PORT="$2"; shift 2;;
    --gpu-prefix) ensure_has_value "$1" "${2-}"; read -r -a GPU_PREFIX <<<"$2"; GPU_PREFIX_CLI=1; shift 2;;

    --model) ensure_has_value "$1" "${2-}"; MODEL="$2"; shift 2;;
    --terse-name) ensure_has_value "$1" "${2-}"; TERSE_MODEL_NAME="$2"; shift 2;;
    --dataset) ensure_has_value "$1" "${2-}"; DATASET="$2"; shift 2;;
    --random-in) ensure_has_value "$1" "${2-}"; RANDOM_IN="$2"; shift 2;;
    --random-out) ensure_has_value "$1" "${2-}"; RANDOM_OUT="$2"; shift 2;;
    --random-range-ratio) ensure_has_value "$1" "${2-}"; RANDOM_RANGE_RATIO="$2"; shift 2;;

    --rates) ensure_has_value "$1" "${2-}"; RATES_CSV="$2"; shift 2;;
    --run-seconds) ensure_has_value "$1" "${2-}"; RUN_SECONDS_PER_RATE="$2"; shift 2;;

    --pr-branch) ensure_has_value "$1" "${2-}"; PR_BRANCH="$2"; shift 2;;
    --main-ref) ensure_has_value "$1" "${2-}"; MAIN_REF="$2"; shift 2;;
    --pr-ref) ensure_has_value "$1" "${2-}"; PR_REF="$2"; shift 2;;
    --build-main) ensure_has_value "$1" "${2-}"; BUILD_MAIN="$2"; shift 2;;
    --build-pr) ensure_has_value "$1" "${2-}"; BUILD_PR="$2"; shift 2;;
    --repo-dir) ensure_has_value "$1" "${2-}"; REPO_DIR="$2"; shift 2;;
    --pull-latest) PULL_LATEST_MAIN="1"; PULL_LATEST_PR="1"; shift 1;;
    --pull-latest-main) PULL_LATEST_MAIN="1"; shift 1;;
    --pull-latest-pr) PULL_LATEST_PR="1"; shift 1;;
    --resume) RESUME="1"; shift 1;;
    --re-run-pr) RERUN_PR="1"; shift 1;;
    --re-run-main) RERUN_MAIN="1"; shift 1;;
    --which) ensure_has_value "$1" "${2-}"; WHICH="$2"; shift 2;;

    --out-base) ensure_has_value "$1" "${2-}"; OUT_BASE="$2"; shift 2;;
    --results-main) ensure_has_value "$1" "${2-}"; RESULTS_DIR_MAIN="$2"; shift 2;;
    --results-pr) ensure_has_value "$1" "${2-}"; RESULTS_DIR_PR="$2"; shift 2;;
    --plots) ensure_has_value "$1" "${2-}"; PLOTS_DIR="$2"; shift 2;;
    --log-dir) ensure_has_value "$1" "${2-}"; LOG_DIR="$2"; shift 2;;
    --label-suffix) ensure_has_value "$1" "${2-}"; LABEL_SUFFIX="$2"; shift 2;;

    --server-args-base) ensure_has_value "$1" "${2-}"; SERVER_ARGS_BASE_STR="$2"; shift 2;;
    --server-args-main) ensure_has_value "$1" "${2-}"; SERVER_ARGS_MAIN_STR="$2"; shift 2;;
    --server-args-pr)   ensure_has_value "$1" "${2-}"; SERVER_ARGS_PR_STR="$2"; shift 2;;

    --variants)      ensure_has_value "$1" "${2-}"; VARIANTS_BOTH_SPEC="$2"; shift 2;;
    --variants-main) ensure_has_value "$1" "${2-}"; VARIANTS_MAIN_SPEC="$2"; VARIANTS_MAIN_SET=1; shift 2;;
    --variants-pr)   ensure_has_value "$1" "${2-}"; VARIANTS_PR_SPEC="$2"; VARIANTS_PR_SET=1; shift 2;;

    --plot) ensure_has_value "$1" "${2-}"; DO_PLOT="$2"; shift 2;;
    --summary-order) ensure_has_value "$1" "${2-}"; SUMMARY_ORDER="$2"; shift 2;;
    --summary-per-variant)
      # Flag form (sets to 1) or optional explicit value (0/1)
      next_val="${2-}"
      if [[ -n "$next_val" && "$next_val" != --* && "$next_val" != -* ]]; then
        SUMMARY_PER_VARIANT="$next_val"; shift 2
      else
        SUMMARY_PER_VARIANT="1"; shift 1
      fi;;

    --help) print_help; exit 0;;
    --*) echo "Error: unknown option: $1" >&2; print_help; exit 2;;
    *) echo "Error: unexpected positional argument: $1" >&2; print_help; exit 2;;
  esac
done

# Default VENV to REPO_DIR/.venv if not explicitly set on CLI and left as default
if [[ "$VENV_CLI" -eq 0 && "$VENV" == ".venv" ]]; then
  VENV="$REPO_DIR/.venv"
fi

# Re-derive paths post-CLI
VENV_PY="$VENV/bin/python"
VENV_VLLM="$VENV/bin/vllm"
VENV_ESCAPED=$(printf '%q' "$VENV")

# Ensure results and log directories exist before anything else
mkdir -p "$OUT_BASE" "$LOG_DIR" "$RESULTS_DIR_MAIN" "$RESULTS_DIR_PR" "$PLOTS_DIR"

# Normalize output paths to absolute to avoid issues after cd into REPO_DIR
OUT_BASE=$(readlink -f "$OUT_BASE")
RESULTS_DIR_MAIN=$(readlink -f "$RESULTS_DIR_MAIN")
RESULTS_DIR_PR=$(readlink -f "$RESULTS_DIR_PR")
PLOTS_DIR=$(readlink -f "$PLOTS_DIR")
LOG_DIR=$(readlink -f "$LOG_DIR")

# Live summary file (used for tmux bottom-right pane)
SUMMARY_CURRENT_FILE="${SUMMARY_CURRENT_FILE:-$LOG_DIR/summary_current.txt}"
SUMMARY_CURRENT_FILE=$(readlink -f "$SUMMARY_CURRENT_FILE")
SUMMARY_CURRENT_FILE_ESCAPED=$(printf '%q' "$SUMMARY_CURRENT_FILE")

# Safe dataset token for filenames
DS_SAFE=$(echo "$DATASET" | sed -E 's/[^A-Za-z0-9._-]+/-/g')

# Apply unified --variants fallback to each side unless side-specific flags were provided
if [[ -n "$VARIANTS_BOTH_SPEC" ]]; then
  if [[ "${VARIANTS_MAIN_SET}" -eq 0 ]]; then
    VARIANTS_MAIN_SPEC="$VARIANTS_BOTH_SPEC"
  fi
  if [[ "${VARIANTS_PR_SET}" -eq 0 ]]; then
    VARIANTS_PR_SPEC="$VARIANTS_BOTH_SPEC"
  fi
fi

# If TP is set, update server-args-base and GPU prefix unless user provided a custom GPU prefix.
if [[ -n "${TP:-}" && "$TP" =~ ^[0-9]+$ && "$TP" -ge 1 ]]; then
  # Inject -tp into base server args if not already present
  if [[ " $SERVER_ARGS_BASE_STR " != *" -tp "* ]]; then
    SERVER_ARGS_BASE_STR+=" ${SERVER_ARGS_BASE_STR:+ }-tp $TP"
  fi
  # Align GPU prefix "chg run -g N --" if not overridden
  if [[ "$GPU_PREFIX_CLI" -eq 0 ]]; then
    GPU_PREFIX=(chg run -g "$TP" --)
  fi
fi

mkdir -p "$LOG_DIR" "$RESULTS_DIR_MAIN" "$RESULTS_DIR_PR" "$PLOTS_DIR"
: > "$SERVER_LOG"; : > "$BENCH_LOG"; : > "$SCRIPT_LOG"; : > "$SUMMARY_CURRENT_FILE"

IFS=',' read -r -a RATES <<<"$RATES_CSV"

# Activate venv
# shellcheck disable=SC1090
source "$VENV/bin/activate"

# ============================================================
# Helpers
# ============================================================
ts() { date +"%Y-%m-%d %H:%M:%S"; }
note() { printf "[%s] %s\n" "$(ts)" "$*"; }

split_to_array() {
  local s="$1"; shift
  local __outvar="$1"
  # shellcheck disable=SC2206
  local arr=( $s )
  eval "$__outvar"='("${arr[@]}")'
}

log_hdr_client() {
  local branch="$1" vlabel="$2" rate="$3" prompts="$4"
  local line
  line=$(printf "[%s] ===== Client run branch=%s var=%s rate=%s prompts=%s =====" \
               "$(ts)" "$branch" "$vlabel" "$rate" "$prompts")
  echo "$line"
  echo "$line" >> "$BENCH_LOG"
}

# Parse variant spec:
#   "label::args"
#   "label::env:K=V,K2=V2::args"
# outputs three parallel arrays by name: labels, args, envs
parse_variants() {
  local spec="$1" out_labels="$2" out_args="$3" out_envs="$4"
  local IFS_SAVE="$IFS"
  IFS=';' read -r -a entries <<<"$spec"
  IFS="$IFS_SAVE"

  local labels=() args=() envs=()
  for e in "${entries[@]}"; do
    [[ -z "$e" ]] && continue
    local cnt=$(awk -F'::' '{print NF-1}' <<<"$e")
  if [[ "$cnt" -eq 1 ]]; then
      # Either label::args OR label::env:K=V(,K2=V2)
      local label="${e%%::*}"
      local rest="${e#*::}"
      if [[ "$rest" == env:* ]]; then
        # env-only, no explicit args
        local ev="${rest#env:}"
        labels+=("$label"); args+=(""); envs+=("$ev")
      else
        labels+=("$label"); args+=("$rest"); envs+=("")
      fi
    else
      # label::env:K=V,K2=V2::args
      local first="${e%%::*}" rest="${e#*::}"
      local maybe_env="${rest%%::*}" argstr="${rest#*::}"
      if [[ "$maybe_env" == env:* ]]; then
        local ev="${maybe_env#env:}"
        labels+=("$first"); args+=("$argstr"); envs+=("$ev")    
      else
        # fallback treat as label::args
        labels+=("$first"); args+=("$rest"); envs+=("")
      fi
    fi
  done
  eval "$out_labels"='("${labels[@]}")'
  eval "$out_args"='("${args[@]}")'
  eval "$out_envs"='("${envs[@]}")'
}

# ============================================================
# Server control
# ============================================================
compose_server_cmd() {
  # $1=args_str  -> prints NUL-delimited argv vector
  local args_str="$1"
  split_to_array "$args_str" CUSTOM_ARGS
  split_to_array "$SERVER_ARGS_BASE_STR" BASE_ARGS

  printf '%s\0' "${GPU_PREFIX[@]}" "$VENV_PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host "$HOST" --port "$PORT" \
    "${CUSTOM_ARGS[@]}" \
    "${BASE_ARGS[@]}" \
    "${ALWAYS_SERVER_ARGS[@]}"
}

# Produce a printable server command string without GPU_PREFIX
compose_server_cmd_string_no_gpu() {
  local args_str="$1"
  split_to_array "$args_str" CUSTOM_ARGS
  split_to_array "$SERVER_ARGS_BASE_STR" BASE_ARGS
  local -a parts=(
    "$VENV_PY" -m vllm.entrypoints.openai.api_server
    --model "$MODEL" --host "$HOST" --port "$PORT"
  )
  parts+=("${CUSTOM_ARGS[@]}")
  parts+=("${BASE_ARGS[@]}")
  parts+=("${ALWAYS_SERVER_ARGS[@]}")
  printf '%q ' "${parts[@]}"
}

# Produce a printable server command string using vllm CLI (no GPU_PREFIX)
compose_server_cmd_vllm_string_no_gpu() {
  local args_str="$1"
  split_to_array "$args_str" CUSTOM_ARGS
  split_to_array "$SERVER_ARGS_BASE_STR" BASE_ARGS
  local -a parts=(
    "$VENV_VLLM" serve "$MODEL"
    --host "$HOST" --port "$PORT"
  )
  parts+=("${CUSTOM_ARGS[@]}")
  parts+=("${BASE_ARGS[@]}")
  parts+=("${ALWAYS_SERVER_ARGS[@]}")
  printf '%q ' "${parts[@]}"
}

# Like compose_server_cmd_vllm_string_no_gpu but with an env CSV prefix (K=V,K2=V2)
compose_server_cmd_vllm_string_no_gpu_with_env() {
  local args_str="$1" env_csv="$2"
  split_to_array "$args_str" CUSTOM_ARGS
  split_to_array "$SERVER_ARGS_BASE_STR" BASE_ARGS
  local -a env_prefix=()
  if [[ -n "$env_csv" ]]; then
    IFS=',' read -r -a kvs <<<"$env_csv"
    for kv in "${kvs[@]}"; do
      [[ -z "$kv" ]] && continue
      env_prefix+=("$kv")
    done
  fi
  local -a parts=(
    "$VENV_VLLM" serve "$MODEL"
    --host "$HOST" --port "$PORT"
  )
  parts+=("${CUSTOM_ARGS[@]}")
  parts+=("${BASE_ARGS[@]}")
  parts+=("${ALWAYS_SERVER_ARGS[@]}")
  if [[ ${#env_prefix[@]} -gt 0 ]]; then
    printf '%q ' env "${env_prefix[@]}" "${parts[@]}"
  else
    printf '%q ' "${parts[@]}"
  fi
}

# ============================================================
# Printing helpers for summary blocks
# ============================================================
print_server_commands_block() {
  # $1: mode ("tee" to also append to $LOG_DIR/summary.txt, anything else for stdout only)
  local mode="${1:-}"
  local do_tee=0
  [[ "$mode" == "tee" ]] && do_tee=1

  echo
  echo "Server commands:"

  # MAIN
  if [[ -z "$VARIANTS_MAIN_SPEC" ]]; then
    local main_cmd
    main_cmd=$(compose_server_cmd_vllm_string_no_gpu "$SERVER_ARGS_MAIN_STR")
    if (( do_tee )); then
      printf "%-6s\t%s\n" "MAIN:" "$main_cmd" | tee -a "$LOG_DIR/summary.txt"
    else
      printf "%-6s\t%s\n" "MAIN:" "$main_cmd"
    fi
  else
    parse_variants "$VARIANTS_MAIN_SPEC" _L _A _E
    for idx in "${!_L[@]}"; do
      local main_cmd
      main_cmd=$(compose_server_cmd_vllm_string_no_gpu_with_env "${_A[$idx]}" "${_E[$idx]}")
      if (( do_tee )); then
        printf "%-6s\t%s\n" "MAIN/${_L[$idx]}:" "$main_cmd" | tee -a "$LOG_DIR/summary.txt"
      else
        printf "%-6s\t%s\n" "MAIN/${_L[$idx]}:" "$main_cmd"
      fi
    done
  fi

  # PR
  if [[ -z "$VARIANTS_PR_SPEC" ]]; then
    local pr_cmd
    pr_cmd=$(compose_server_cmd_vllm_string_no_gpu "$SERVER_ARGS_PR_STR")
    if (( do_tee )); then
      printf "%-6s\t%s\n" "PR:" "$pr_cmd" | tee -a "$LOG_DIR/summary.txt"
    else
      printf "%-6s\t%s\n" "PR:" "$pr_cmd"
    fi
  else
    parse_variants "$VARIANTS_PR_SPEC" _L _A _E
    for idx in "${!_L[@]}"; do
      local pr_cmd
      pr_cmd=$(compose_server_cmd_vllm_string_no_gpu_with_env "${_A[$idx]}" "${_E[$idx]}")
      if (( do_tee )); then
        printf "%-6s\t%s\n" "PR/${_L[$idx]}:" "$pr_cmd" | tee -a "$LOG_DIR/summary.txt"
      else
        printf "%-6s\t%s\n" "PR/${_L[$idx]}:" "$pr_cmd"
      fi
    done
  fi
}

print_client_command_template() {
  # $1: mode ("tee" to also append to $LOG_DIR/summary.txt)
  local mode="${1:-}"
  local do_tee=0
  [[ "$mode" == "tee" ]] && do_tee=1

  echo
  echo "Client command (template):"
  local bench_cmd=(
    vllm bench serve
    --model "$MODEL" --host "$HOST" --port "$PORT"
    --dataset-name "$DATASET"
    --random-input-len "$RANDOM_IN" --random-output-len "$RANDOM_OUT"
    --random-range-ratio "$RANDOM_RANGE_RATIO"
    --num-prompts "<req_rate * $RUN_SECONDS_PER_RATE>" --request-rate "<req_rate>"
    --save-result --result-dir "<results_dir>"
    --result-filename "<filename>"
    --seed 42
    --ignore-eos --trust-remote-code
  )
  if (( do_tee )); then
    printf "%-6s\t%s\n" "CLIENT:" "${bench_cmd[*]}" | tee -a "$LOG_DIR/summary.txt"
  else
    printf "%-6s\t%s\n" "CLIENT:" "${bench_cmd[*]}"
  fi
}

SERVER_PID=""
stop_server() {
  # Try graceful shutdown: send INT to the process group so launchers like chg can clean up
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT -$SERVER_PID 2>/dev/null || true
    # Wait briefly for graceful exit
    for _ in {1..10}; do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
    # Fallback to TERM, then KILL
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -TERM -$SERVER_PID 2>/dev/null || true
      sleep 1
    fi
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL -$SERVER_PID 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  # Also best-effort kill any stray matching servers on the same port
  pkill -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" 2>/dev/null || true
}

wait_for_server_ready() {
  local url="http://${HOST}:${PORT}/v1/models"
  local tries=600
  note "Waiting for server at ${url} ..."
  for ((i=1;i<=tries;i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      note "Server is ready."
      return 0
    fi
    sleep 1
  done
  note "WARN: Server did not become ready after ${tries}s"
  return 1
}

CLEANED_UP=0
cleanup() {
  ((CLEANED_UP)) && return
  CLEANED_UP=1
  note "Cleanup: stopping server"
  stop_server
  if [[ -z "${TMUX:-}" && -n "${TMUX_SESSION_NAME:-}" ]]; then
    tmux kill-session -t "$TMUX_SESSION_NAME" >/dev/null 2>&1 || true
  fi
}
trap 'exit 130' INT
trap 'cleanup' EXIT

start_server_with_args() {
  local argstr="$1" label="$2" branch="$3" env_csv="$4"
  note "Starting ${branch} server var=${label}"

  # build env prefix string: KEY=VAL KEY2=VAL2
  local env_prefix=()
  if [[ -n "$env_csv" ]]; then
    IFS=',' read -r -a kvs <<<"$env_csv"
    for kv in "${kvs[@]}"; do
      [[ -z "$kv" ]] && continue
      env_prefix+=("$kv")
    done
  fi

  stop_server
  # shellcheck disable=SC2046
  SERVER_PID=$( 
    set -f
    # Build argv from compose_server_cmd (NUL-delimited)
    local -a cmd_argv=()
    while IFS= read -r -d '' tok; do
      cmd_argv+=("$tok")
    done < <(compose_server_cmd "$argstr")

    # Insert env KEY=VAL pairs after the GPU prefix (if any), using `env` so execve doesn't treat KEY=VAL as a program
    local -a full_cmd=()
    if [[ ${#env_prefix[@]} -gt 0 ]]; then
      local gpulen=${#GPU_PREFIX[@]}
      if [[ $gpulen -gt 0 ]]; then
        local -a head=("${cmd_argv[@]:0:gpulen}")
        local -a tail=("${cmd_argv[@]:gpulen}")
        full_cmd=("${head[@]}" env "${env_prefix[@]}" "${tail[@]}")
      else
        full_cmd=(env "${env_prefix[@]}" "${cmd_argv[@]}")
      fi
    else
      full_cmd=("${cmd_argv[@]}")
    fi
    if [[ "${#full_cmd[@]}" -eq 0 ]]; then
      echo "ERROR: empty server command" >&2
      exit 1
    fi
    setsid "${full_cmd[@]}" >>"$SERVER_LOG" 2>&1 &
    echo $!
  )
  note "${branch} server PID: $SERVER_PID"
  wait_for_server_ready
}

# ============================================================
# Client loop
# ============================================================
run_client_loop() {
  local results_dir="$1" branch_label="$2" variant_label="$3"
  for R in "${RATES[@]}"; do
    local N=$((R*RUN_SECONDS_PER_RATE))
    log_hdr_client "$branch_label" "$variant_label" "$R" "$N"
    local result_file="${results_dir}/bench_model-${TERSE_MODEL_NAME}_rate-${R}_v-${variant_label}_np-${N}_in-${RANDOM_IN}_out-${RANDOM_OUT}_ds-${DS_SAFE}${LABEL_SUFFIX:+_}${LABEL_SUFFIX}.json"

    # Overwrite/delete existing result if requested
    if [[ "$OVERWRITE" == "1" && -s "$result_file" ]]; then
      note "Overwrite: removing existing result $result_file"
      rm -f "$result_file" || true
    fi
    # Reuse existing results unless overwrite or re-run flags are set
    if [[ -s "$result_file" ]]; then
      if [[ ( "$branch_label" == "PR" && "$RERUN_PR" == "1" ) || ( "$branch_label" == "MAIN" && "$RERUN_MAIN" == "1" ) ]]; then
        note "Re-run requested for $branch_label/$variant_label rate=$R despite existing result"
      else
        note "Reusing existing result for $branch_label/$variant_label rate=$R ($result_file)"
        write_summary_table > "$SUMMARY_CURRENT_FILE" || true
        continue
      fi
    fi
    "$VENV_VLLM" bench serve \
      --model "$MODEL" \
      --host "$HOST" --port "$PORT" \
      --dataset-name "$DATASET" \
      --random-input-len "$RANDOM_IN" \
      --random-output-len "$RANDOM_OUT" \
      --random-range-ratio "$RANDOM_RANGE_RATIO" \
      --num-prompts "$N" --request-rate "$R" \
      --save-result --result-dir "$results_dir" \
      --result-filename "bench_model-${TERSE_MODEL_NAME}_rate-${R}_v-${variant_label}_np-${N}_in-${RANDOM_IN}_out-${RANDOM_OUT}_ds-${DS_SAFE}${LABEL_SUFFIX:+_}${LABEL_SUFFIX}.json" \
      --seed 42 \
      --ignore-eos --trust-remote-code \
      >>"$BENCH_LOG" 2>&1
    note "Completed client run branch=$branch_label var=$variant_label rate=$R prompts=$N"
    # Update live summary after each iteration
    write_summary_table > "$SUMMARY_CURRENT_FILE" || true
  done
}

run_variants() {
  # $1=results_dir  $2=branch_label  $3=labels[@]  $4=args_list[@]  $5=envs_list[@]
  local results_dir="$1" branch_label="$2"; shift 2
  local -n labels_ref="$1"; local -n args_ref="$2"; local -n envs_ref="$3"

  for i in "${!labels_ref[@]}"; do
    local vlabel="${labels_ref[$i]}"
    local vargs="${args_ref[$i]}"
    local venvs="${envs_ref[$i]}"
    local cmd_print
    cmd_print=$(compose_server_cmd_vllm_string_no_gpu_with_env "$vargs" "$venvs")
    note "Server command ($branch_label/$vlabel): $cmd_print"
    # If all rate JSONs exist and not overwriting or forced re-run, skip server
    local all_exist=1
    for R in "${RATES[@]}"; do
      local N=$((R*RUN_SECONDS_PER_RATE))
      local rf="${results_dir}/bench_model-${TERSE_MODEL_NAME}_rate-${R}_v-${vlabel}_np-${N}_in-${RANDOM_IN}_out-${RANDOM_OUT}_ds-${DS_SAFE}${LABEL_SUFFIX:+_}${LABEL_SUFFIX}.json"
      if [[ ! -s "$rf" ]]; then all_exist=0; break; fi
    done
    if [[ $all_exist -eq 1 && "$OVERWRITE" != "1" ]]; then
      if [[ ( "$branch_label" == "MAIN" && "$RERUN_MAIN" != "1" ) || ( "$branch_label" == "PR" && "$RERUN_PR" != "1" ) ]]; then
        note "Skipping $branch_label/$vlabel (all requested rates present)"
        continue
      fi
    fi
    start_server_with_args "$vargs" "$vlabel" "$branch_label" "$venvs"
    run_client_loop "$results_dir" "$branch_label" "$vlabel"
    stop_server
  done
}

# ============================================================
# Summary (from JSON result files)
# ============================================================
write_summary_table() {
  # Build allowed lists based on current run config
  local allowed_rates_csv="$RATES_CSV"
  local include_main=1 include_pr=1
  if [[ "$WHICH" == "pr" ]]; then include_main=0; fi
  if [[ "$WHICH" == "main" ]]; then include_pr=0; fi

  # Compute allowed variant labels for MAIN and PR
  local ALLOWED_MAIN_VARIANTS_CSV="" ALLOWED_PR_VARIANTS_CSV=""
  if [[ -n "$VARIANTS_MAIN_SPEC" ]]; then
    local _ML=() _MA=() _ME=()
    parse_variants "$VARIANTS_MAIN_SPEC" _ML _MA _ME
    ALLOWED_MAIN_VARIANTS_CSV=$(IFS=,; echo "${_ML[*]}")
  else
    ALLOWED_MAIN_VARIANTS_CSV="base"
  fi
  if [[ -n "$VARIANTS_PR_SPEC" ]]; then
    local _PL=() _PA=() _PE=()
    parse_variants "$VARIANTS_PR_SPEC" _PL _PA _PE
    ALLOWED_PR_VARIANTS_CSV=$(IFS=,; echo "${_PL[*]}")
  else
    ALLOWED_PR_VARIANTS_CSV="base"
  fi

  "$VENV_PY" - \
    "$RESULTS_DIR_MAIN" "$RESULTS_DIR_PR" \
    "$SUMMARY_TSV" "$SUMMARY_ORDER" "$SUMMARY_PER_VARIANT" \
    "$allowed_rates_csv" "$include_main" "$include_pr" \
    "$ALLOWED_MAIN_VARIANTS_CSV" "$ALLOWED_PR_VARIANTS_CSV" \
    "$RUN_SECONDS_PER_RATE" "$RANDOM_IN" "$RANDOM_OUT" "$DS_SAFE" <<'PY'
import json, sys, pathlib, re
import os

res_main, res_pr, out_tsv = sys.argv[1], sys.argv[2], sys.argv[3]
order_by = sys.argv[4] if len(sys.argv) > 4 else 'rate'
per_variant = sys.argv[5] if len(sys.argv) > 5 else '0'
allowed_rates_csv = sys.argv[6] if len(sys.argv) > 6 else ''
include_main = (sys.argv[7] == '1') if len(sys.argv) > 7 else True
include_pr = (sys.argv[8] == '1') if len(sys.argv) > 8 else True
allowed_main_variants_csv = sys.argv[9] if len(sys.argv) > 9 else ''
allowed_pr_variants_csv = sys.argv[10] if len(sys.argv) > 10 else ''
run_seconds = float(sys.argv[11]) if len(sys.argv) > 11 else None
expected_in = int(sys.argv[12]) if len(sys.argv) > 12 else None
expected_out = int(sys.argv[13]) if len(sys.argv) > 13 else None
expected_ds = sys.argv[14] if len(sys.argv) > 14 else None

def extract_filename_meta(filename):
    # Extract rate, variant, optional np/in/out/ds
    m = re.search(r"bench_model-.*?_rate-([\d.]+)_v-([^_]+)", filename)
    if not m:
        return None
    rate = float(m.group(1)); variant = m.group(2)
    np = None; in_tok = None; out_tok = None; ds = None
    m2 = re.search(r"_np-([0-9]+)", filename); np = int(m2.group(1)) if m2 else None
    m3 = re.search(r"_in-([0-9]+)", filename); in_tok = int(m3.group(1)) if m3 else None
    m4 = re.search(r"_out-([0-9]+)", filename); out_tok = int(m4.group(1)) if m4 else None
    # Keep dataset token optional; filtering will use filename suffix check
    ds = None
    return rate, variant, np, in_tok, out_tok, ds

def collect_rows(dir_path, branch):
    rows = []
    if not dir_path or not os.path.isdir(dir_path):
        return rows
    for name in os.listdir(dir_path):
        if not name.endswith('.json'):
            continue
        meta = extract_filename_meta(name)
        if meta is None:
            continue
        rate, variant, np_from_name, in_from_name, out_from_name, ds_from_name = meta
        full = os.path.join(dir_path, name)
        try:
            with open(full, 'r') as f:
                data = json.load(f)
        except Exception:
            continue
        req_s = data.get('request_throughput')
        ttft_med = data.get('median_ttft_ms')
        tpot_med = data.get('median_tpot_ms')
        ttft_std = data.get('std_ttft_ms')
        tpot_std = data.get('std_tpot_ms')
        # p99 fields (assumed available)
        ttft_p99 = data.get('p99_ttft_ms')
        tpot_p99 = data.get('p99_tpot_ms')
        num_prompts = data.get('num_prompts', 0)
        if req_s is None or ttft_med is None or tpot_med is None:
            continue
        rows.append({'branch': branch, 'variant': variant, 'rate': float(rate),
                     'req_s': float(req_s), 'ttft_med': float(ttft_med), 'tpot_med': float(tpot_med),
                     'ttft_std': ttft_std, 'tpot_std': tpot_std,
                     'ttft_p99': ttft_p99, 'tpot_p99': tpot_p99,
                     'num_prompts': int(num_prompts),
                     'np_name': np_from_name, 'in_name': in_from_name, 'out_name': out_from_name, 'ds_name': ds_from_name,
                     'file_name': name})
    return rows

rows = []
if include_main: rows += collect_rows(res_main, 'MAIN')
if include_pr: rows += collect_rows(res_pr, 'PR')

# Filter to allowed rates/variants only
allowed_rates = set()
if allowed_rates_csv:
    for x in allowed_rates_csv.split(','):
        x = x.strip()
        if not x: continue
        try:
            allowed_rates.add(float(x))
        except ValueError:
            pass
allowed_main_variants = set(v for v in allowed_main_variants_csv.split(',') if v.strip()) if allowed_main_variants_csv else None
allowed_pr_variants = set(v for v in allowed_pr_variants_csv.split(',') if v.strip()) if allowed_pr_variants_csv else None

def keep_row(r):
    if allowed_rates:
        rr = float(r['rate'])
        if min((abs(rr - a) for a in allowed_rates), default=0.0) > 1e-9:
            return False
    if r['branch'] == 'MAIN' and allowed_main_variants is not None and r['variant'] not in allowed_main_variants: return False
    if r['branch'] == 'PR' and allowed_pr_variants is not None and r['variant'] not in allowed_pr_variants: return False
    # num_prompts filter: must match current run's expected N = rate * run_seconds
    if run_seconds is not None:
        exp_np = int(float(r['rate']) * run_seconds)
        # Prefer JSON num_prompts, fallback to filename np
        np_val = int(r.get('num_prompts') or 0)
        if np_val == 0 and r.get('np_name') is not None:
            np_val = int(r['np_name'])
        if np_val != exp_np:
            return False
    # in/out/ds filters when present in filename
    if expected_in is not None and r.get('in_name') is not None and int(r['in_name']) != expected_in:
        return False
    if expected_out is not None and r.get('out_name') is not None and int(r['out_name']) != expected_out:
        return False
    if expected_ds:
        # Check dataset token from filename suffix to avoid regex mismatches
        fname = r.get('file_name') or ''
        if not fname.endswith(f"_ds-{expected_ds}.json"):
            return False
    return True

rows = [r for r in rows if keep_row(r)]

if not rows:
    # Debug: print diagnostics to understand why rows are filtered out
    print("No rows matched strict filters. Diagnostics:")
    all_rows = []
    if include_main:
        all_rows += collect_rows(res_main, 'MAIN')
    if include_pr:
        all_rows += collect_rows(res_pr, 'PR')
    print(f"Total JSON rows found: {len(all_rows)}")
    print(f"Allowed rates: {sorted(list(allowed_rates)) if allowed_rates else 'ANY'}")
    print(f"Allowed MAIN variants: {sorted(list(allowed_main_variants)) if allowed_main_variants is not None else 'ANY'}")
    print(f"Allowed PR variants: {sorted(list(allowed_pr_variants)) if allowed_pr_variants is not None else 'ANY'}")
    print(f"Expected run_seconds: {run_seconds}, expected_in: {expected_in}, expected_out: {expected_out}, expected_ds: {expected_ds}")
    for r in all_rows:
        reasons = []
        rr = float(r['rate'])
        if allowed_rates:
            if min((abs(rr - a) for a in allowed_rates), default=0.0) > 1e-9:
                reasons.append('rate')
        if r['branch'] == 'MAIN' and allowed_main_variants is not None and r['variant'] not in allowed_main_variants:
            reasons.append('variant_main')
        if r['branch'] == 'PR' and allowed_pr_variants is not None and r['variant'] not in allowed_pr_variants:
            reasons.append('variant_pr')
        if run_seconds is not None:
            exp_np = int(float(r['rate']) * run_seconds)
            np_val = int(r.get('num_prompts') or 0)
            if np_val == 0 and r.get('np_name') is not None:
                np_val = int(r['np_name'])
            if np_val != exp_np:
                reasons.append(f"num_prompts({np_val}!={exp_np})")
        if expected_in is not None and r.get('in_name') is not None and int(r['in_name']) != expected_in:
            reasons.append(f"in({r.get('in_name')}!={expected_in})")
        if expected_out is not None and r.get('out_name') is not None and int(r['out_name']) != expected_out:
            reasons.append(f"out({r.get('out_name')}!={expected_out})")
        if expected_ds and r.get('ds_name') is not None and r['ds_name'] != expected_ds:
            reasons.append(f"ds({r.get('ds_name')}!={expected_ds})")
        print(f"Row {r['branch']} v={r['variant']} rate={r['rate']} np={r.get('num_prompts')} fn_np={r.get('np_name')} in={r.get('in_name')} out={r.get('out_name')} ds={r.get('ds_name')} -> filtered_by={','.join(reasons) if reasons else 'OK'}")

if order_by == 'rate':
    rows.sort(key=lambda r: (r['rate'], r['branch'], r['variant']))
else:
    rows.sort(key=lambda r: (r['branch'], r['variant'], r['rate']))

pathlib.Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
with open(out_tsv, 'w') as out:
    uniq_variants = sorted(set(r['variant'] for r in rows))
    include_variant = len(uniq_variants) > 1
    if include_variant:
        out.write("branch\tvariant\trate\tnum_prompts\treq_s\tmedian_ttft_ms\tstd_ttft_ms\tp99_ttft_ms\tmedian_tpot_ms\tstd_tpot_ms\tp99_tpot_ms\n")
        for r in rows:
            out.write(f"{r['branch']}\t{r['variant']}\t{r['rate']:.2f}\t{int(r.get('num_prompts',0))}\t{r['req_s']:.2f}\t{r['ttft_med']:.2f}\t{(r['ttft_std'] if r['ttft_std'] is not None else '')}\t{(r['ttft_p99'] if r['ttft_p99'] is not None else '')}\t{r['tpot_med']:.2f}\t{(r['tpot_std'] if r['tpot_std'] is not None else '')}\t{(r['tpot_p99'] if r['tpot_p99'] is not None else '')}\n")
    else:
        out.write("branch\trate\tnum_prompts\treq_s\tmedian_ttft_ms\tstd_ttft_ms\tp99_ttft_ms\tmedian_tpot_ms\tstd_tpot_ms\tp99_tpot_ms\n")
        for r in rows:
            out.write(f"{r['branch']}\t{r['rate']:.2f}\t{int(r.get('num_prompts',0))}\t{r['req_s']:.2f}\t{r['ttft_med']:.2f}\t{(r['ttft_std'] if r['ttft_std'] is not None else '')}\t{(r['ttft_p99'] if r['ttft_p99'] is not None else '')}\t{r['tpot_med']:.2f}\t{(r['tpot_std'] if r['tpot_std'] is not None else '')}\t{(r['tpot_p99'] if r['tpot_p99'] is not None else '')}\n")

print("""Print a human-readable table only""")
uniq_variants = sorted(set(r['variant'] for r in rows))
include_variant = len(uniq_variants) > 1

def print_table(the_rows, show_variant_column):
    if show_variant_column:
        colw = [8, 16, 8, 12, 8, 18, 12, 12, 18, 12, 12]
        headers = ["branch","variant","rate","num_prompts","req/s","median TTFT (ms)","std TTFT","p99 TTFT","median TPOT (ms)","std TPOT","p99 TPOT"]
        fmt = lambda cols: " ".join(str(c).ljust(w) for c,w in zip(cols,colw))
        print(fmt(headers))
        print(fmt(["-"*len(h) for h in headers]))
        for r in the_rows:
            print(fmt([r['branch'], r['variant'], f"{r['rate']:.2f}", int(r.get('num_prompts',0)), f"{r['req_s']:.2f}", f"{r['ttft_med']:.2f}", (f"{r['ttft_std']:.2f}" if r.get('ttft_std') is not None else ''), (f"{r['ttft_p99']:.2f}" if r.get('ttft_p99') is not None else ''), f"{r['tpot_med']:.2f}", (f"{r['tpot_std']:.2f}" if r.get('tpot_std') is not None else ''), (f"{r['tpot_p99']:.2f}" if r.get('tpot_p99') is not None else '')]))
    else:
        colw = [8, 8, 12, 8, 18, 12, 12, 18, 12, 12]
        headers = ["branch","rate","num_prompts","req/s","median TTFT (ms)","std TTFT","p99 TTFT","median TPOT (ms)","std TPOT","p99 TPOT"]
        fmt = lambda cols: " ".join(str(c).ljust(w) for c,w in zip(cols,colw))
        print(fmt(headers))
        print(fmt(["-"*len(h) for h in headers]))
        for r in the_rows:
            print(fmt([r['branch'], f"{r['rate']:.2f}", int(r.get('num_prompts',0)), f"{r['req_s']:.2f}", f"{r['ttft_med']:.2f}", (f"{r['ttft_std']:.2f}" if r.get('ttft_std') is not None else ''), (f"{r['ttft_p99']:.2f}" if r.get('ttft_p99') is not None else ''), f"{r['tpot_med']:.2f}", (f"{r['tpot_std']:.2f}" if r.get('tpot_std') is not None else ''), (f"{r['tpot_p99']:.2f}" if r.get('tpot_p99') is not None else '')]))

if per_variant == '1':
    # If no variants found, print an empty table header for readability
    if not uniq_variants:
        print_table(rows, show_variant_column=False)
    else:
        for v in uniq_variants:
            print(f"Variant: {v}")
            variant_rows = [r for r in rows if r['variant'] == v]
            print_table(variant_rows, show_variant_column=False)
            print()
else:
    print_table(rows, show_variant_column=include_variant)
PY
}

# ============================================================
# Workflow
# ============================================================
run_benchmarks() {
  local main_ref="${MAIN_REF:-main}"
  local pr_ref="${PR_REF:-$PR_BRANCH}"
  note "Using venv: $VENV"
  note "Using model: $MODEL"
  note "Using dataset: $DATASET"
  note "Using random input len: $RANDOM_IN"
  note "Using random output len: $RANDOM_OUT"
  note "Using random range ratio: $RANDOM_RANGE_RATIO"
  note "Using run seconds per rate: $RUN_SECONDS_PER_RATE"
  note "Using rates: $RATES_CSV"
  note "Using server args base: $SERVER_ARGS_BASE_STR"
  note "===================== beginning run ====================="
  note "Repo dir: $REPO_DIR"
  (
    cd "$REPO_DIR" || { echo "Error: repo dir '$REPO_DIR' not found" >&2; exit 1; }
    note "Fetching tags (no pull by default)"
    git fetch origin --tags || true

    if [[ "$WHICH" == "both" || "$WHICH" == "main" ]]; then
      note "Checking out MAIN ref '${main_ref}'"
      git checkout "$main_ref"
      # Pull only if requested and ref looks like a branch name (not a 7-40 hex)
      if [[ "$PULL_LATEST_MAIN" == "1" && ! "$main_ref" =~ ^[0-9a-f]{7,40}$ ]]; then
        note "Pulling latest for MAIN"
        git pull --rebase || true
      else
        note "Not pulling latest for MAIN (default)"
      fi
      if [[ "$BUILD_MAIN" == "1" ]]; then
        note "Rebuilding extensions on MAIN"
        "$VENV_PY" setup.py build_ext --inplace
      else
        note "Skipping setup.py build on MAIN"
      fi

      local MAIN_LABELS=() MAIN_ARGS_LIST=() MAIN_ENVS_LIST=()
      if [[ -n "$VARIANTS_MAIN_SPEC" ]]; then
        parse_variants "$VARIANTS_MAIN_SPEC" MAIN_LABELS MAIN_ARGS_LIST MAIN_ENVS_LIST
      else
        MAIN_LABELS=("base"); MAIN_ARGS_LIST=("$SERVER_ARGS_MAIN_STR"); MAIN_ENVS_LIST=("")
      fi
      note "Running MAIN variants: [${MAIN_LABELS[*]}]"
      run_variants "$RESULTS_DIR_MAIN" "MAIN" MAIN_LABELS MAIN_ARGS_LIST MAIN_ENVS_LIST
    fi

    if [[ "$WHICH" == "both" || "$WHICH" == "pr" ]]; then
      note "Checking out PR ref '${pr_ref}'"
      git checkout "$pr_ref"
      if [[ "$PULL_LATEST_PR" == "1" && ! "$pr_ref" =~ ^[0-9a-f]{7,40}$ ]]; then
        note "Pulling latest for PR"
        git pull --rebase || true
      else
        note "Not pulling latest for PR (default)"
      fi
      if [[ "$BUILD_PR" == "1" ]]; then
        note "Rebuilding extensions on PR"
        "$VENV_PY" setup.py build_ext --inplace
      else
        note "Skipping setup.py build on PR"
      fi

      local PR_LABELS=() PR_ARGS_LIST=() PR_ENVS_LIST=()
      if [[ -n "$VARIANTS_PR_SPEC" ]]; then
        parse_variants "$VARIANTS_PR_SPEC" PR_LABELS PR_ARGS_LIST PR_ENVS_LIST
      else
        PR_LABELS=("base"); PR_ARGS_LIST=("$SERVER_ARGS_PR_STR"); PR_ENVS_LIST=("")
      fi
      note "Running PR variants: [${PR_LABELS[*]}]"
      run_variants "$RESULTS_DIR_PR" "PR" PR_LABELS PR_ARGS_LIST PR_ENVS_LIST
    fi
  )

  if [[ "$DO_PLOT" == "1" ]]; then
    note "Plotting results (log-y, max-rate 10)"
    "$VENV_PY" plot_benchmark_results.py \
      --current-results "$RESULTS_DIR_PR" \
      --main-results    "$RESULTS_DIR_MAIN" \
      --current-branch-name PR \
      --output-dir "$PLOTS_DIR" \
      --log-y --max-rate 10
  fi

  note "Building summary table from JSON results"
  # Write the summary table directly from JSON results
  write_summary_table | tee "$LOG_DIR/summary.txt"

  # Print reproducible server commands (no GPU prefix)
  echo
  echo "Server commands:"
  # Print reproducible server commands and client template
  print_server_commands_block tee
  print_client_command_template tee
  # (Server commands are printed below outside tmux as part of final summary)

  note "Done. Results: $RESULTS_DIR_PR, $RESULTS_DIR_MAIN | Plots: $PLOTS_DIR | Summary: $SUMMARY_TSV"
}

# ============================================================
# tmux orchestration
# ============================================================
SCRIPT_PATH="$(readlink -f "$0")"

start_tmux_layout_and_run() {
  local WAIT_TOKEN="${TMUX_SESSION_NAME}-done"
  local WAIT_TOKEN_ESCAPED
  WAIT_TOKEN_ESCAPED=$(printf '%q' "$WAIT_TOKEN")
  if [[ ! -t 1 ]]; then
    note "No TTY detected; running without tmux live view."
    run_benchmarks | tee -a "$SCRIPT_LOG"
    return
  fi
  if ! command -v tmux >/dev/null 2>&1; then
    note "tmux not found; running without live view."
    run_benchmarks | tee -a "$SCRIPT_LOG"
    return
  fi

  # If resuming, pre-populate live summary so the pane shows immediately
  if [[ "$RESUME" == "1" ]]; then
    write_summary_table > "$SUMMARY_CURRENT_FILE" || true
  fi

  tmux new-session -d -s "$TMUX_SESSION_NAME" "tail -n +1 -F '$SERVER_LOG'"
  # Ctrl-C: fan out to all panes in the session, then close via wait token
  tmux bind-key -n C-c "run-shell \"for p in \$(tmux list-panes -t '$TMUX_SESSION_NAME' -F '#{pane_id}'); do tmux send-keys -t \$p C-c; done; sleep 0.2; tmux wait-for -S $WAIT_TOKEN_ESCAPED\""
  # Make panes scrollable: enable mouse and increase history for this session
  tmux set-option -t "$TMUX_SESSION_NAME" mouse on
  tmux set-option -t "$TMUX_SESSION_NAME" history-limit 200000
  # 2x2 layout using relative pane selection for robustness
  # top-right (bench)
  tmux split-window -h -t "$TMUX_SESSION_NAME":0.0 "tail -n +1 -F '$BENCH_LOG'"
  # bottom-left (script)
  tmux select-pane -L -t "$TMUX_SESSION_NAME"
  tmux split-window -v -t "$TMUX_SESSION_NAME" \
    "RUN_INSIDE_TMUX=1 VENV=$VENV_ESCAPED bash '$SCRIPT_PATH' $ORIG_ARGS_ESCAPED; tmux wait-for -S $WAIT_TOKEN_ESCAPED"
  # bottom-right (summary)
  tmux select-pane -R -t "$TMUX_SESSION_NAME"
  tmux split-window -v -t "$TMUX_SESSION_NAME" "bash -lc 'watch -n 2 cat $SUMMARY_CURRENT_FILE_ESCAPED'"
  # focus bottom-right
  tmux select-pane -R -t "$TMUX_SESSION_NAME"; tmux select-pane -D -t "$TMUX_SESSION_NAME"
  # Auto-close the session when the script pane signals completion
  ( tmux wait-for "$WAIT_TOKEN"; tmux kill-session -t "$TMUX_SESSION_NAME" ) &
  # Attach to show live panes while running
  tmux attach -t "$TMUX_SESSION_NAME" || true
}

if [[ "${RUN_INSIDE_TMUX:-}" == "1" ]]; then
  trap 'cleanup; exit 130' INT
  run_benchmarks 2>&1 | tee -a "$SCRIPT_LOG"
  # Signal the parent tmux controller that we're done
  tmux wait-for -S "${TMUX_SESSION_NAME}-done" >/dev/null 2>&1 || true
  # Do not kill the tmux session here; let the top-level orchestrator decide or the user close it
  exit 0
fi

start_tmux_layout_and_run

# Ensure tmux session is closed, then print final summary and paths outside tmux
tmux kill-session -t "$TMUX_SESSION_NAME" >/dev/null 2>&1 || true

echo "Final summary (also saved to $SUMMARY_TSV)"
write_summary_table | tee "$LOG_DIR/summary.txt"

# Print reproducible server commands (no GPU prefix)
print_server_commands_block tee
# Then client command template
print_client_command_template tee
echo
echo "Logs directory: $LOG_DIR"
echo "Results directories: PR=$RESULTS_DIR_PR | MAIN=$RESULTS_DIR_MAIN"
if [[ -d "$PLOTS_DIR" ]]; then
  echo "Plots directory: $PLOTS_DIR"
fi