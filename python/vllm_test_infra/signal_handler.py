"""Global signal handling for graceful shutdown."""

import atexit
import signal
import sys
from typing import Callable, Optional

_cleanup_handlers = []
_signal_received = False


def register_cleanup(handler: Callable[[], None]) -> None:
    """Register a cleanup handler to be called on exit or signal.
    
    Args:
        handler: Callable to invoke during cleanup.
    """
    global _cleanup_handlers
    if handler not in _cleanup_handlers:
        _cleanup_handlers.append(handler)


def _run_cleanup_handlers():
    """Run all registered cleanup handlers."""
    global _cleanup_handlers
    for handler in _cleanup_handlers:
        try:
            handler()
        except Exception as e:
            print(f"Error in cleanup handler: {e}", file=sys.stderr)


def _signal_handler(signum, frame):
    """Handle termination signals."""
    global _signal_received
    
    if _signal_received:
        # Already handling signal, force exit
        sys.exit(128 + signum)
    
    _signal_received = True
    
    signal_name = signal.Signals(signum).name
    print(f"\n🛑 Received {signal_name}, cleaning up...", file=sys.stderr, flush=True)
    
    # Run cleanup handlers
    _run_cleanup_handlers()
    
    # Exit with appropriate code
    sys.exit(128 + signum)


def setup_signal_handlers():
    """Setup global signal handlers for graceful shutdown.
    
    This should be called early in main() of scripts to ensure
    proper cleanup on interruption.
    """
    # Register cleanup to run on normal exit too
    atexit.register(_run_cleanup_handlers)
    
    # Install signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    # On POSIX systems, also handle SIGHUP
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, _signal_handler)

