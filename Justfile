# vLLM development setup recipe
setup-vllm name="vllm" arch="89-real":
    #!/usr/bin/env bash
    set -euo pipefail
    
    echo "Setting up vLLM in {{ name }}..."
    
    # Clone or update repository (resumable)
    if [ -d "{{ name }}" ]; then
      echo "Repository already exists at {{ name }}, fetching updates..."
      cd {{ name }}
      git fetch origin
      git reset --hard origin/main
    else
      echo "Cloning vLLM repository..."
      git clone https://github.com/vllm-project/vllm.git {{ name }}
      cd {{ name }}
    fi
    
    echo "Adding/updating neuralmagic remote..."
    git remote add nm https://github.com/neuralmagic/vllm 2>/dev/null || git remote set-url nm https://github.com/neuralmagic/vllm
    
    # Create or reuse existing venv (resumable)
    if [ ! -d ".venv" ]; then
      echo "Creating virtual environment..."
      uv venv --python=3.12
    else
      echo "Virtual environment already exists, skipping creation..."
    fi
    
    source .venv/bin/activate
    
    echo "Installing build requirements..."
    uv pip install -r requirements/build.txt
    
    echo "Installing vLLM in editable mode..."
    VLLM_DISABLE_SCCACHE=1 CCACHE_NOHASHDIR="true" uv pip install --no-build-isolation -e . -v
    
    echo "Installing EP kernels with TORCH_CUDA_ARCH_LIST={{ arch }}..."
    pushd tools/ep_kernels
    TORCH_CUDA_ARCH_LIST="{{ arch }}" PIP_CMD="uv pip" bash install_python_libraries.sh
    popd
    
    echo "✅ vLLM setup complete in {{ name }}/"

# Run GSM8K evaluation with vLLM server
# Usage: 
#   jl gsm8k-eval                                                     # defaults
#   jl gsm8k-eval deepseek-ai/DeepSeek-R1 "--tp 2"                   # with server args
#   jl gsm8k-eval deepseek-ai/DeepSeek-R1 "--tp 2 --gpu-memory-utilization 0.9" "100"  # with limit
gsm8k-eval model="deepseek-ai/DeepSeek-R1" server_args="" limit="":
    #!/usr/bin/env bash
    set -euo pipefail
    
    # Find script relative to ~/.config/aliases/dot-aliases
    EVAL_SCRIPT="$HOME/.config/aliases/dot-aliases/bash-scripts/gsm8k-eval.sh"
    
    if [[ ! -f "$EVAL_SCRIPT" ]]; then
        echo "ERROR: gsm8k-eval.sh not found at $EVAL_SCRIPT"
        exit 1
    fi
    
    # Use venv from the directory where just was invoked (not where Justfile is)
    INVOKE_DIR="{{ invocation_directory() }}"
    export VENV="$INVOKE_DIR/.venv"
    
    # Verify venv exists before proceeding
    if [[ ! -d "$VENV" ]]; then
        echo "ERROR: venv not found at $VENV"
        echo "Invocation directory: $INVOKE_DIR"
        echo "Please run this command from a directory containing a .venv folder (e.g., ~/code/vllm/)"
        exit 1
    fi
    
    CMD="bash $EVAL_SCRIPT --model '{{ model }}'"
    
    if [[ -n "{{ server_args }}" ]]; then
        CMD="$CMD --server-args '{{ server_args }}'"
    fi
    
    if [[ -n "{{ limit }}" ]]; then
        CMD="$CMD --limit {{ limit }}"
    fi
    
    echo "Running: $CMD"
    eval "$CMD"
