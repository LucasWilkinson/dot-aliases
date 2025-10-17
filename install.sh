#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://raw.githubusercontent.com/LucasWilkinson/dot-aliases/main}"
TARGET_DIR="${HOME}/.config/aliases"
ALIAS_FILE="${TARGET_DIR}/aliases.sh"

mkdir -p "$TARGET_DIR"

# Fetch aliases.sh
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

echo "Aliases installed. Open a new shell or run: source \"$ALIAS_FILE\""
