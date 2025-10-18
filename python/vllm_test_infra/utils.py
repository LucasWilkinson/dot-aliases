"""Common utilities for vLLM test infrastructure."""

import os
import re
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def timestamp() -> str:
    """Return formatted timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def note(msg: str) -> None:
    """Print timestamped message."""
    print(f"[{timestamp()}] {msg}", flush=True)


def cleanup_zombie_processes(user: Optional[str] = None) -> None:
    """Kill stray vLLM and test processes.
    
    Args:
        user: Optional username to filter processes. If None, kills all matching processes.
    """
    patterns = [
        "api_server",
        "benchmarks",
        "VLLM::",
        "pytest",
        "python.*test",
    ]
    
    note("Checking for zombie processes...")
    killed_any = False
    
    for pattern in patterns:
        try:
            # pkill with -f matches full command line
            cmd = ["pkill", "-9", "-f", pattern]
            if user:
                cmd.extend(["-u", user])
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                note(f"  Killed processes matching: {pattern}")
                killed_any = True
        except Exception as e:
            # pkill returns non-zero if no processes found, which is fine
            pass
    
    if killed_any:
        time.sleep(2)  # Give processes time to die
    
    # Check GPU memory status
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            note("GPU Memory Status:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    note(f"  {line}")
    except Exception:
        pass  # nvidia-smi might not be available


def check_gpu_memory() -> List[Dict[str, str]]:
    """Check current GPU memory usage.
    
    Returns:
        List of dicts with 'index', 'used', 'total' keys for each GPU.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        gpus.append({
                            'index': parts[0],
                            'used': parts[1],
                            'total': parts[2]
                        })
            return gpus
    except Exception:
        pass
    return []


def parse_variants(spec: str) -> List[Tuple[str, str, str]]:
    """Parse variant specification string.
    
    Format:
        "label::args;label2::args2"
        "label::env:K=V,K2=V2::args"
    
    Args:
        spec: Semicolon-separated variant specifications.
    
    Returns:
        List of (label, args, env_csv) tuples.
    """
    if not spec:
        return []
    
    variants = []
    entries = spec.split(';')
    
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        
        parts = entry.split('::')
        
        if len(parts) == 1:
            # Just a label, no args or env
            variants.append((parts[0], "", ""))
        elif len(parts) == 2:
            # Either "label::args" or "label::env:..."
            label, rest = parts
            if rest.startswith('env:'):
                # env-only, no explicit args
                env_csv = rest[4:]  # Remove "env:" prefix
                variants.append((label, "", env_csv))
            else:
                # Just args
                variants.append((label, rest, ""))
        elif len(parts) >= 3:
            # "label::env:K=V::args" or "label::something::args"
            label = parts[0]
            middle = parts[1]
            rest = '::'.join(parts[2:])  # Rejoin in case args contain ::
            
            if middle.startswith('env:'):
                env_csv = middle[4:]
                variants.append((label, rest, env_csv))
            else:
                # Treat as label::args (ignore middle for now)
                variants.append((label, middle + '::' + rest, ""))
        
    return variants


def extract_tp_dp_from_args(args: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract tensor-parallel and data-parallel sizes from argument string.
    
    Args:
        args: Space-separated argument string.
    
    Returns:
        Tuple of (tp_size, dp_size), either can be None if not found.
    """
    tp_size = None
    dp_size = None
    
    # Match -tp N or --tensor-parallel-size N
    tp_match = re.search(r'(?:-tp|--tensor-parallel-size)\s+(\d+)', args)
    if tp_match:
        tp_size = int(tp_match.group(1))
    
    # Match -dp N or --data-parallel-size N (if such flag exists)
    dp_match = re.search(r'(?:-dp|--data-parallel-size)\s+(\d+)', args)
    if dp_match:
        dp_size = int(dp_match.group(1))
    
    return tp_size, dp_size


def compute_gpu_count(tp_size: Optional[int] = None, dp_size: Optional[int] = None) -> int:
    """Compute total GPU count needed.
    
    Args:
        tp_size: Tensor parallel size.
        dp_size: Data parallel size.
    
    Returns:
        Total number of GPUs needed (tp * dp, defaulting to 1 if both None).
    """
    tp = tp_size or 1
    dp = dp_size or 1
    return tp * dp


def is_chg_available() -> bool:
    """Check if chg command is available."""
    try:
        result = subprocess.run(['which', 'chg'], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def split_args_string(args: str) -> List[str]:
    """Split argument string into list, handling quoted strings.
    
    Args:
        args: Space-separated argument string.
    
    Returns:
        List of argument strings.
    """
    if not args:
        return []
    
    # Simple split for now - can be enhanced with proper shell parsing if needed
    import shlex
    try:
        return shlex.split(args)
    except ValueError:
        # Fallback to simple split if shlex fails
        return args.split()

