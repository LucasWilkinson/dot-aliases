# dot-aliases

A simple, cross-shell repository for keeping your aliases in one place and installing them on any machine in a single command.

## Files

- `aliases.sh` — Your shell aliases and functions (works in bash, zsh, and POSIX-compatible shells)
- `bashrc.snippet` — Loader snippet for Bash (`~/.bashrc`)
- `zshrc.snippet` — Loader snippet for Zsh (`~/.zshrc`)
- `install.sh` — Idempotent installer script

## Quick Install

**One-liner for any machine:**

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/LucasWilkinson/dot-aliases/main/install.sh)"
```

This will:
1. Create `~/.config/aliases/` directory
2. Download `aliases.sh` to `~/.config/aliases/aliases.sh`
3. Add a loader line to your `~/.bashrc` or `~/.zshrc` (idempotent—safe to run multiple times)
4. Load the aliases in your current session

Open a new shell or run:
```bash
source ~/.config/aliases/aliases.sh
```

## Machine-Specific Overrides

Create `~/.config/aliases/local.sh` for host-specific aliases or secrets. It will automatically be sourced at the end of `aliases.sh` if it exists:

```bash
# ~/.config/aliases/local.sh (not tracked in Git)
alias myhost='ssh user@myhost'
export MY_SECRET_TOKEN="..."
```

## Auto-Update

Add this alias to your `aliases.sh` to keep aliases up-to-date:

```bash
alias aliases-update='curl -fsSL https://raw.githubusercontent.com/LucasWilkinson/dot-aliases/main/aliases.sh -o ~/.config/aliases/aliases.sh && source ~/.config/aliases/aliases.sh && echo "Aliases updated."'
```

## Included Aliases

### Navigation & Lists
- `ll` — `ls -lah`
- `la` — `ls -la`
- `l` — `ls -la`

### Git
- `gs` — `git status`
- `gp` — `git pull --rebase --autostash && git push`
- `gd` — `git diff`
- `gc` — `git commit`
- `ga` — `git add`

### Editors
- `v` — `nvim`
- `vi` — `vim`

### System
- `ports` — `sudo lsof -i -P -n | grep LISTEN` (show listening ports)
- `gpu` — `nvidia-smi` or "No NVIDIA GPU"
- `gpumem` — Show GPU memory usage

### vLLM Development
- `venv` — Activate vLLM virtual environment
- `vllm-code` — Jump to vLLM code directory
- `vllm-build` — Rebuild vLLM with CUDA kernels
- `vllm-test` — Run pytest with verbose output
- `cleanup-vllm` — Kill vLLM processes

### Development
- `py` / `py3` — Python shortcuts
- `reload` — Reload `~/.bashrc`
- `aliases-edit` — Edit aliases with `$EDITOR`
- `aliases-reload` — Reload aliases immediately

### Safety
- `cp` — `cp -i` (interactive)
- `mv` — `mv -i` (interactive)
- `rm` — `rm -i` (interactive)

## Customization

Edit `aliases.sh` and commit:

```bash
cd ~/dot-aliases
# Edit aliases.sh
git add aliases.sh
git commit -m "Update aliases"
git push
```

Then on any machine, run:
```bash
aliases-update
```

Or force a fresh pull on your current machine:
```bash
curl -fsSL https://raw.githubusercontent.com/LucasWilkinson/dot-aliases/main/install.sh | bash
```

## Security Notes

- The installer is simple and readable—review it before running
- For extra safety, pin a specific commit:
  ```bash
  bash -c "$(curl -fsSL https://raw.githubusercontent.com/LucasWilkinson/dot-aliases/<COMMIT_SHA>/install.sh)"
  ```
- Use `~/.config/aliases/local.sh` to keep secrets out of Git

## License

MIT
