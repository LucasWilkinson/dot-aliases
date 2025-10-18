"""vLLM Test Infrastructure Package.

A reusable infrastructure for running vLLM tests with robust server management,
process monitoring, logging, and user interfaces.
"""

from .config import Config
from .git import GitError, GitManager
from .logging import LogManager
from .process import ProcessManager
from .server import VLLMServer, VLLMServerError
from .ui import UIManager, run_with_ui
from .utils import (
    check_gpu_memory,
    cleanup_zombie_processes,
    compute_gpu_count,
    extract_tp_dp_from_args,
    is_chg_available,
    note,
    parse_variants,
    split_args_string,
    timestamp,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "Config",
    # Git
    "GitManager",
    "GitError",
    # Logging
    "LogManager",
    # Process
    "ProcessManager",
    # Server
    "VLLMServer",
    "VLLMServerError",
    # UI
    "UIManager",
    "run_with_ui",
    # Utils
    "check_gpu_memory",
    "cleanup_zombie_processes",
    "compute_gpu_count",
    "extract_tp_dp_from_args",
    "is_chg_available",
    "note",
    "parse_variants",
    "split_args_string",
    "timestamp",
]

