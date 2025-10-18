# ----- Aliases you want everywhere -----
alias ll='ls -lah'
alias la='ls -la'
alias l='ls -la'
alias gs='git status'
alias gp='git pull --rebase --autostash && git push'
alias gd='git diff'
alias gc='git commit'
alias ga='git add'
alias v='nvim'
alias vi='vim'

# Safer file ops
alias cp='cp -i'
alias mv='mv -i'
alias rm='rm -i'

# Quick system info
alias ports='sudo lsof -i -P -n | grep LISTEN'
alias gpu='nvidia-smi 2>/dev/null || echo "No NVIDIA GPU"'
alias gpumem='nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,header'

# vLLM development
alias venv='source /home/lwilkinson/code/vllm/.venv/bin/activate'
alias vllm-code='cd /home/lwilkinson/code/vllm'
alias vllm-build='source ~/.venv/bin/activate && cd ~/code/vllm && export VLLM_FLASH_ATTN_SRC_DIR=~/code/flash-attention && VLLM_DISABLE_SCCACHE=1 python setup.py build_ext --inplace'
alias vpi='VLLM_DISABLE_SCCACHE=1 CCACHE_NOHASHDIR="true" uv pip install --no-build-isolation -e . -v'
alias vbx='python setup.py build_ext --inplace'
alias va='source .venv/bin/activate'
alias vllm-test='pytest -xvs'
alias cleanup-vllm='pkill -9 -f "api_server" 2>/dev/null; pkill -9 -f "benchmarks" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; sleep 2'
alias jl='just -f ~/.config/aliases/dot-aliases/Justfile'

# vLLM Test Infrastructure
alias gsm8k-eval='${HOME}/.config/aliases/dot-aliases/scripts/gsm8k_eval.py'
alias gpqa-diamond-eval='${HOME}/.config/aliases/dot-aliases/scripts/gpqa_diamond_eval.py'
alias benchmark-compare='${HOME}/.config/aliases/dot-aliases/scripts/benchmark_compare.py'
alias benchmark='${HOME}/.config/aliases/dot-aliases/scripts/benchmark_single.py'
alias profile='${HOME}/.config/aliases/dot-aliases/scripts/profile.py'

# Install dependencies with uv pip (preferred) or fallback to pip/pip3
_vllm_test_infra_install() {
  local req_file="${HOME}/.config/aliases/dot-aliases/python/requirements.txt"
  if command -v uv &> /dev/null; then
    echo "Using uv pip..."
    uv pip install -r "$req_file" && echo "✅ vLLM test infrastructure dependencies installed"
  elif command -v pip &> /dev/null; then
    echo "Using pip..."
    pip install -r "$req_file" && echo "✅ vLLM test infrastructure dependencies installed"
  elif command -v pip3 &> /dev/null; then
    echo "Using pip3..."
    pip3 install -r "$req_file" && echo "✅ vLLM test infrastructure dependencies installed"
  else
    echo "❌ Error: No pip/uv found. Please install uv or pip first."
    return 1
  fi
}
alias vllm-test-infra-install='_vllm_test_infra_install'

# CUDA debugging with coredumps
cuda-debug() {
  CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_SHOW_PROGRESS=1 CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory' "$@"
}

# Quick shortcuts
alias py='python'
alias py3='python3'
alias tmp='cd /tmp'
alias home='cd ~'
alias c='clear'

# Development helpers
alias reload='source ~/.bashrc'
alias aliases-edit='$EDITOR ~/.config/aliases/aliases.sh'
alias aliases-reload='source ~/.config/aliases/aliases.sh && echo "Aliases reloaded."'

# Include local/machine-specific aliases if they exist
[ -f "${HOME}/.config/aliases/local.sh" ] && . "${HOME}/.config/aliases/local.sh"
