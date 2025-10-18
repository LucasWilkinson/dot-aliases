#!/usr/bin/env python3
"""Benchmark comparison script for vLLM across branches."""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.vllm_test_infra import BaseBenchmarkRunner, UIManager
from python.vllm_test_infra.ui import run_with_ui


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmark comparison across vLLM branches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model meta-llama/Meta-Llama-3-8B-Instruct --rates 1,5,10
  %(prog)s --variants 'base::;full::-O.cudagraph_mode=FULL'
  %(prog)s --which pr --build-pr 1 --pr-ref my-feature-branch
        """
    )
    
    # General options
    parser.add_argument("--venv", default=".venv", help="Virtualenv path")
    parser.add_argument("--host", default="localhost", help="Host")
    parser.add_argument("--port", type=int, default=3333, help="Port")
    
    # Model / Data
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model (HF id or local path)"
    )
    parser.add_argument(
        "--terse-name",
        help="Short name for filenames (default: derived from model)"
    )
    parser.add_argument("--dataset", default="random", help="Dataset name")
    parser.add_argument("--random-in", type=int, default=1000, help="Random input len")
    parser.add_argument("--random-out", type=int, default=100, help="Random output len")
    parser.add_argument("--random-range-ratio", type=float, default=0, help="Random range ratio")
    
    # Parallelism
    parser.add_argument("-tp", "--tensor-parallel-size", type=int, default=1, help="Tensor-parallel size")
    
    # Rates / Durations
    parser.add_argument("--rates", default="1,5,10,25,100", help="Request rates CSV")
    parser.add_argument("--run-seconds", type=int, default=120, help="Seconds per rate")
    
    # Git / Build
    parser.add_argument("--main-ref", default="main", help="Checkout this ref for MAIN")
    parser.add_argument("--pr-ref", help="Checkout this ref for PR")
    parser.add_argument("--pr-branch", default="full_cudagraph_FA2_FlashInfer", help="PR branch fallback")
    parser.add_argument("--build-main", type=int, default=0, help="Run setup.py build on MAIN")
    parser.add_argument("--build-pr", type=int, default=0, help="Run setup.py build on PR")
    parser.add_argument("--repo-dir", default=".", help="Git repository directory")
    parser.add_argument("--pull-latest", action="store_true", help="Pull latest for MAIN and PR")
    parser.add_argument("--pull-latest-main", action="store_true", help="Pull latest for MAIN")
    parser.add_argument("--pull-latest-pr", action="store_true", help="Pull latest for PR")
    parser.add_argument("--resume", action="store_true", help="Resume and only run missing combos")
    parser.add_argument("--re-run-pr", action="store_true", help="With --resume, re-run all PR variants")
    parser.add_argument("--re-run-main", action="store_true", help="With --resume, re-run all MAIN variants")
    parser.add_argument(
        "--which",
        choices=["both", "main", "pr"],
        default="both",
        help="Which side(s) to run"
    )
    
    # Output / Paths
    parser.add_argument("--out-base", default="./results", help="Base output directory")
    parser.add_argument("--results-main", help="Results dir for MAIN")
    parser.add_argument("--results-pr", help="Results dir for PR")
    parser.add_argument("--log-dir", help="Log directory")
    parser.add_argument("--label-suffix", default="", help="Suffix for result filenames")
    
    # Server Args
    parser.add_argument("--server-args-base", default="", help="Args added to EVERY server run")
    parser.add_argument("--server-args-main", default="", help="Extra args for MAIN server")
    parser.add_argument("--server-args-pr", default="", help="Extra args for PR server")
    
    # Variants
    parser.add_argument("--variants", help="Variant spec for both MAIN and PR")
    parser.add_argument("--variants-main", help="Variant spec for MAIN only")
    parser.add_argument("--variants-pr", help="Variant spec for PR only")
    
    # UI
    parser.add_argument(
        "--ui-mode",
        choices=["tui", "simple", "auto"],
        default="auto",
        help="UI mode"
    )
    
    return parser.parse_args()


class BenchmarkRunner(BaseBenchmarkRunner):
    """Manages benchmark execution across branches and variants."""
    
    def __init__(self, args):
        """Initialize BenchmarkRunner.
        
        Args:
            args: Parsed command line arguments.
        """
        # Call parent constructor
        super().__init__(args)
        
        # Parse variants for main and PR
        self.variants_main = self._parse_variant_spec(
            args.variants_main or args.variants,
            args.server_args_main
        )
        self.variants_pr = self._parse_variant_spec(
            args.variants_pr or args.variants,
            args.server_args_pr
        )
    
    def _setup_directories(self):
        """Setup output directories for compare mode."""
        self.out_base = Path(self.args.out_base).resolve()
        self.results_dir_main = Path(
            self.args.results_main or self.out_base / f"bench_main_{self.terse_model_name}_{self.args.dataset}"
        ).resolve()
        self.results_dir_pr = Path(
            self.args.results_pr or self.out_base / f"bench_pr_{self.terse_model_name}_{self.args.dataset}"
        ).resolve()
        self.log_dir = Path(
            self.args.log_dir or self.out_base / "logs"
        ).resolve()
        
        # Create directories
        self.out_base.mkdir(parents=True, exist_ok=True)
        self.results_dir_main.mkdir(parents=True, exist_ok=True)
        self.results_dir_pr.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    
    def run(self) -> int:
        """Run the benchmark comparison.
        
        Returns:
            Exit code.
        """
        # Log configuration info (will appear in TUI script pane or simple mode stdout)
        self.logger.info("=" * 60)
        self.logger.info("Benchmark Comparison")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {self.args.model}")
        self.logger.info(f"Dataset: {self.args.dataset}")
        self.logger.info(f"Rates: {self.args.rates}")
        self.logger.info(f"Repository: {self.args.repo_dir}")
        self.logger.info(f"Results MAIN: {self.results_dir_main}")
        self.logger.info(f"Results PR: {self.results_dir_pr}")
        self.logger.info(f"Logs: {self.log_dir}")
        self.logger.info("=" * 60)
        
        try:
            # Run MAIN branch
            if self.args.which in ["both", "main"]:
                force_rerun = self.args.re_run_main if hasattr(self.args, 're_run_main') else False
                self.run_branch(
                    "MAIN",
                    self.args.main_ref,
                    self.variants_main,
                    self.results_dir_main,
                    bool(self.args.build_main),
                    self.args.pull_latest or self.args.pull_latest_main,
                    force_rerun
                )
            
            # Run PR branch
            if self.args.which in ["both", "pr"]:
                pr_ref = self.args.pr_ref or self.args.pr_branch
                force_rerun = self.args.re_run_pr if hasattr(self.args, 're_run_pr') else False
                self.run_branch(
                    "PR",
                    pr_ref,
                    self.variants_pr,
                    self.results_dir_pr,
                    bool(self.args.build_pr),
                    self.args.pull_latest or self.args.pull_latest_pr,
                    force_rerun
                )
            
            self.logger.info("=" * 60)
            self.logger.info("Benchmark comparison completed!")
            self.logger.info(f"Results MAIN: {self.results_dir_main}")
            self.logger.info(f"Results PR: {self.results_dir_pr}")
            self.logger.info(f"Logs: {self.log_dir}")
            self.logger.info("=" * 60)
            
            return 0
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
            return 130
        
        except Exception as e:
            self.logger.error(f"Error: {e}", exc_info=True)
            return 1


def main():
    """Main entry point."""
    args = parse_args()
    runner = BenchmarkRunner(args)
    
    # Setup UI manager
    ui_manager = UIManager(runner.log_manager, mode=args.ui_mode)
    
    # Create work function
    def work_func():
        return runner.run()
    
    # Run with appropriate UI mode
    panes = {
        "server-pane": "server",
        "bench-pane": "bench",
        "script-pane": "script",
    }
    
    exit_code = run_with_ui(ui_manager, work_func, panes)
    
    # Print summary after UI closes
    print()
    print("=" * 60)
    if exit_code == 0:
        print("✅ Benchmark comparison completed!")
        print(f"📁 Results MAIN: {runner.results_dir_main}")
        print(f"📁 Results PR: {runner.results_dir_pr}")
    else:
        print("❌ Benchmark comparison failed!")
    
    print(f"📋 Logs: {runner.log_dir}")
    print("=" * 60)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

