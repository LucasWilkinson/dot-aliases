#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://raw.githubusercontent.com/LucasWilkinson/dot-aliases/main}"
REPO_GIT_URL="${REPO_GIT_URL:-https://github.com/LucasWilkinson/dot-aliases.git}"
TARGET_DIR="${HOME}/.config/aliases"
ALIAS_FILE="${TARGET_DIR}/aliases.sh"
REPO_DIR="${TARGET_DIR}/dot-aliases"

mkdir -p "$TARGET_DIR"

# Clone or update the full dot-aliases repository (for Justfile and other assets)
if [ -d "$REPO_DIR" ]; then
  echo "Updating dot-aliases repository..."
  cd "$REPO_DIR"
  git fetch origin
  git reset --hard origin/main
  cd - > /dev/null
else
  echo "Cloning dot-aliases repository..."
  git clone "$REPO_GIT_URL" "$REPO_DIR"
fi

# Fetch aliases.sh (keep this for backwards compatibility / direct access)
curl -fsSL "$REPO_URL/aliases.sh" -o "$ALIAS_FILE"

# Ensure readable perms
chmod 0644 "$ALIAS_FILE"

# Detect shell and append loader once (idempotent)
append_once() {
  local file="$1" line="$2"
  grep -Fqs "$line" "$file" 2>/dev/null || printf "\n%s\n" "$line" >> "$file"
}

SHELL_NAME="$(basename "${SHELL:-sh}")"

# Install loader for bash
if [ "$SHELL_NAME" = "bash" ] || [ -n "${BASH_VERSION:-}" ]; then
  curl -fsSL "$REPO_URL/bashrc.snippet" -o /tmp/.bash_aliases_loader
  append_once "${HOME}/.bashrc" "$(cat /tmp/.bash_aliases_loader)"
fi

# Install loader for zsh
if [ "$SHELL_NAME" = "zsh" ] || [ -n "${ZSH_VERSION:-}" ]; then
  curl -fsSL "$REPO_URL/zshrc.snippet" -o /tmp/.zsh_aliases_loader
  append_once "${HOME}/.zshrc" "$(cat /tmp/.zsh_aliases_loader)"
fi

# Install just command runner
if ! command -v just &> /dev/null; then
  echo "Installing just..."
  curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to "${HOME}/.local/bin"
  echo "just installed to ~/.local/bin"
else
  echo "just is already installed"
fi

# Make Python scripts executable
chmod +x "${REPO_DIR}/scripts/gsm8k_eval.py" 2>/dev/null || true
chmod +x "${REPO_DIR}/scripts/benchmark_compare.py" 2>/dev/null || true

# Install Python dependencies for vLLM test infrastructure (optional)
echo ""
echo "vLLM Test Infrastructure is available at: ${REPO_DIR}/python/"
echo "To install Python dependencies, run: vllm-test-infra-install"
echo "  (or manually: pip install -r ${REPO_DIR}/python/requirements.txt)"

echo "Aliases installed. Open a new shell or run: source \"$ALIAS_FILE\""
