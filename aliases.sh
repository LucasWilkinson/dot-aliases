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
alias vllm-test='pytest -xvs'
alias cleanup-vllm='pkill -9 -f "api_server" 2>/dev/null; pkill -9 -f "benchmarks" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; sleep 2'

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
