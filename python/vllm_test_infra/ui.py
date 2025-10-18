"""User interface management for vLLM test infrastructure."""

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from .logging import LogManager
from .utils import note

# Textual imports - optional dependency
try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Vertical, VerticalScroll
    from textual.widgets import Header, Footer, Label, Static, RichLog
    from textual.reactive import reactive
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False


class WorkerThread(threading.Thread):
    """Runs work in background thread, sends updates to TUI via queue."""
    
    def __init__(self, work_func: Callable[[], int], 
                 update_queue: queue.Queue,
                 exception_queue: queue.Queue):
        """Initialize WorkerThread.
        
        Args:
            work_func: Function that does the work, returns exit code.
            update_queue: Queue for sending status updates to TUI.
            exception_queue: Queue for sending exceptions to TUI.
        """
        super().__init__(daemon=True)
        self.work_func = work_func
        self.update_queue = update_queue
        self.exception_queue = exception_queue
        self._exit_code = None
    
    def run(self) -> None:
        """Run the work function and report results."""
        try:
            self._exit_code = self.work_func()
            self.update_queue.put(("complete", self._exit_code))
        except Exception as e:
            self.exception_queue.put(e)
            self.update_queue.put(("error", str(e)))
    
    def get_exit_code(self) -> Optional[int]:
        """Get the exit code from the work function."""
        return self._exit_code


if TEXTUAL_AVAILABLE:
    class LogPane(VerticalScroll):
        """A widget that displays log file contents with auto-refresh."""
        
        def __init__(self, log_path: Path, title: str, **kwargs):
            """Initialize LogPane.
            
            Args:
                log_path: Path to log file to display.
                title: Title for the pane.
            """
            super().__init__(**kwargs)
            self.log_path = log_path
            self.title_text = title
            self.last_position = 0
            self.rich_log = None
        
        def compose(self) -> ComposeResult:
            """Create child widgets."""
            yield Label(f"[bold]{self.title_text}[/bold]")
            yield RichLog(id=f"log-content-{id(self)}", wrap=False, highlight=False, markup=False)
        
        def on_mount(self) -> None:
            """Start auto-refresh when mounted."""
            # Get reference to RichLog widget
            self.rich_log = self.query_one(f"#log-content-{id(self)}", RichLog)
            # Set up interval to update logs
            self.set_interval(0.5, self._update_content)
        
        def _update_content(self) -> None:
            """Read new content from log file."""
            if not self.log_path.exists():
                return
            
            try:
                with open(self.log_path, 'r') as f:
                    f.seek(self.last_position)
                    new_content = f.read()
                    if new_content and self.rich_log:
                        # Write new content to RichLog (it handles auto-scroll)
                        self.rich_log.write(new_content, expand=True)
                        self.last_position = f.tell()
            except Exception:
                pass


    class TestRunnerApp(App):
        """Textual app for displaying test logs with worker thread."""
        
        CSS = """
        Screen {
            layout: grid;
            grid-size: 2 2;
            grid-gutter: 1;
        }
        
        LogPane {
            height: 100%;
            border: solid green;
        }
        
        LogPane:focus-within {
            border: solid yellow;
        }
        
        RichLog {
            height: 1fr;
            border: none;
        }
        
        #server-pane {
            column-span: 1;
            row-span: 1;
        }
        
        #eval-pane {
            column-span: 1;
            row-span: 1;
        }
        
        #script-pane {
            column-span: 2;
            row-span: 1;
        }
        
        Label {
            height: auto;
        }
        """
        
        BINDINGS = [
            ("q", "quit", "Quit"),
            ("ctrl+c", "quit", "Quit"),
            ("tab", "focus_next", "Next pane"),
            ("shift+tab", "focus_previous", "Previous pane"),
        ]
        
        def __init__(self, 
                     log_manager: LogManager,
                     panes: Dict[str, str],
                     work_func: Callable[[], int],
                     **kwargs):
            """Initialize TestRunnerApp.
            
            Args:
                log_manager: LogManager instance.
                panes: Dict of pane_id -> log_file_name mappings.
                work_func: Function that does the work.
            """
            super().__init__(**kwargs)
            self.log_manager = log_manager
            self.panes_config = panes
            self.work_func = work_func
            self.log_panes = {}
            
            # Queues for communication with worker thread
            self.update_queue = queue.Queue()
            self.exception_queue = queue.Queue()
            self.worker_thread = None
            self.exit_code = 0
        
        def compose(self) -> ComposeResult:
            """Create child widgets."""
            yield Header()
            
            for pane_id, log_name in self.panes_config.items():
                log_path = self.log_manager.get_log_path(log_name)
                title = log_name.replace('_', ' ').title()
                pane = LogPane(log_path, title, id=pane_id)
                self.log_panes[pane_id] = pane
                yield pane
            
            yield Footer()
        
        def on_mount(self) -> None:
            """Start worker thread when app is mounted."""
            # Start worker thread
            self.worker_thread = WorkerThread(
                self.work_func,
                self.update_queue,
                self.exception_queue
            )
            self.worker_thread.start()
            
            # Set up interval to check queue
            self.set_interval(0.1, self._check_queue)
        
        def _check_queue(self) -> None:
            """Check queue for messages from worker thread."""
            try:
                # Non-blocking queue check
                while True:
                    msg_type, msg_data = self.update_queue.get_nowait()
                    
                    if msg_type == "complete":
                        self.exit_code = msg_data
                        self.exit()
                    elif msg_type == "error":
                        self.exit_code = 1
                        # Show error in UI before exiting
                        note(f"Error: {msg_data}")
                        self.exit()
            
            except queue.Empty:
                pass
            
            # Check for exceptions
            try:
                exc = self.exception_queue.get_nowait()
                note(f"Exception in worker thread: {exc}")
                self.exit_code = 1
                self.exit()
            except queue.Empty:
                pass
        
        def on_unmount(self) -> None:
            """Cleanup when app closes."""
            # Worker thread is daemon, will be killed automatically
            pass
else:
    # Dummy classes when Textual is not available
    LogPane = None
    TestRunnerApp = None


class UIManager:
    """Manages user interface for test execution."""
    
    def __init__(self, 
                 log_manager: LogManager,
                 mode: str = "auto"):
        """Initialize UIManager.
        
        Args:
            log_manager: LogManager instance for log files.
            mode: UI mode - "tui" (Textual), "simple" (stdout only), or "auto" (detect).
        """
        self.log_manager = log_manager
        self.mode = mode
        self._app = None
        
        # Auto-detect mode if requested
        if self.mode == "auto":
            self.mode = self._detect_mode()
        
        note(f"UI mode: {self.mode}")
    
    def _detect_mode(self) -> str:
        """Detect appropriate UI mode.
        
        Returns:
            "simple" always for now (safe default).
        """
        # Always default to simple mode for reliability
        # User must explicitly request TUI with --ui-mode tui
        return "simple"
    
    def run_with_tui(self, work_func: Callable[[], int], 
                    panes: Optional[Dict[str, str]] = None) -> int:
        """Run work function with TUI interface.
        
        Args:
            work_func: Function that does the work, returns exit code.
            panes: Optional pane configuration.
        
        Returns:
            Exit code from work function.
        """
        if panes is None:
            panes = {
                "server-pane": "server",
                "script-pane": "script",
            }
        
        if not TEXTUAL_AVAILABLE or TestRunnerApp is None:
            note("Textual not available, falling back to simple mode")
            return work_func()
        
        try:
            note("Starting TUI...")
            app = TestRunnerApp(self.log_manager, panes, work_func)
            app.run()
            return app.exit_code
        
        except Exception as e:
            note(f"TUI failed: {e}, falling back to simple mode")
            return work_func()
    
    def run_simple(self, work_func: Callable[[], int]) -> int:
        """Run work function in simple mode (direct execution).
        
        Args:
            work_func: Function that does the work, returns exit code.
        
        Returns:
            Exit code from work function.
        """
        note("Using simple stdout mode")
        return work_func()
    
    def cleanup(self) -> None:
        """Cleanup UI resources."""
        pass


def run_with_ui(ui_manager: UIManager,
                work_func: Callable[[], int],
                panes: Optional[Dict[str, str]] = None) -> int:
    """Run work function with appropriate UI mode.
    
    Args:
        ui_manager: UIManager instance.
        work_func: Function that does the work, returns exit code.
        panes: Optional pane configuration for TUI mode.
    
    Returns:
        Exit code from work function.
    """
    if ui_manager.mode == "tui":
        return ui_manager.run_with_tui(work_func, panes)
    else:
        return ui_manager.run_simple(work_func)
