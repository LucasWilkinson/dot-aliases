#!/usr/bin/env python3
"""GPQA-Diamond Evaluation Script using vLLM server and lm_eval."""

import argparse
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.vllm_test_infra import (
    Config,
    EvalRunner,
    LogManager,
    ProcessManager,
    UIManager,
    note,
)
from python.vllm_test_infra.ui import run_with_ui


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GPQA-Diamond evaluation using vLLM server and lm_eval"
    )
    
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1",
        help="Model to evaluate (default: deepseek-ai/DeepSeek-R1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3333,
        help="Server port (default: 3333)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of test cases (default: no limit)"
    )
    parser.add_argument(
        "--num-concurrent",
        type=int,
        default=256,
        help="Number of concurrent requests (default: 256)"
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        help="Batch size for lm_eval (default: auto)"
    )
    parser.add_argument(
        "--server-args",
        default="",
        help="Additional vllm serve arguments (quoted string)"
    )
    parser.add_argument(
        "--out-base",
        default="./gpqa-diamond-results",
        help="Output directory (default: ./gpqa-diamond-results)"
    )
    parser.add_argument(
        "--venv",
        default=".venv",
        help="Path to virtual environment (default: .venv)"
    )
    parser.add_argument(
        "--ui-mode",
        choices=["tui", "simple", "auto"],
        default="auto",
        help="UI mode: tui (Textual), simple (stdout only), or auto (detect)"
    )
    
    return parser.parse_args()


def check_lm_eval_available() -> bool:
    """Check if lm_eval is available."""
    import shutil
    return shutil.which("lm_eval") is not None


def run_gpqa_diamond_work(args, log_manager: LogManager, process_manager: ProcessManager) -> int:
    """Run GPQA-Diamond evaluation work.
    
    This function can be called from main thread (simple mode) or worker thread (TUI mode).
    
    Args:
        args: Parsed command line arguments.
        log_manager: LogManager instance.
        process_manager: ProcessManager instance.
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    # Get logger
    logger = log_manager.get_logger("gpqa_diamond_eval", log_to_file="script")
    
    # Determine output paths
    out_base = Path(args.out_base).resolve()
    
    # Create eval runner
    runner = EvalRunner(
        model=args.model,
        task_name="leaderboard_gpqa_diamond",
        host=args.host,
        port=args.port,
        venv_path=args.venv,
        log_manager=log_manager,
        process_manager=process_manager,
        out_base=out_base
    )
    
    # Run evaluation
    return runner.run_evaluation(
        server_args=args.server_args,
        limit=args.limit,
        num_concurrent=args.num_concurrent,
        batch_size=args.batch_size,
        timeout=3600  # 1 hour timeout
    )


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup configuration
    config = Config(venv_path=args.venv)
    config.activate_venv()
    
    # Determine output paths
    out_base = Path(args.out_base).resolve()
    log_dir = out_base / "logs"
    
    # Setup logging
    log_manager = LogManager(str(log_dir))
    log_manager.setup()
    log_manager.init_log_file("server")
    log_manager.init_log_file("eval")
    log_manager.init_log_file("script")
    
    # Get logger for script
    logger = log_manager.get_logger("gpqa_diamond_eval", log_to_file="script")
    
    # Log configuration info (only in simple mode, TUI will show in logs)
    if args.ui_mode != "tui":
        logger.info("=" * 50)
        logger.info("GPQA-Diamond Evaluation")
        logger.info("=" * 50)
        logger.info(f"Model: {args.model}")
        logger.info(f"Output: {out_base}")
        logger.info(f"Server: http://{args.host}:{args.port}")
        if args.limit:
            logger.info(f"Limit: {args.limit}")
        if args.server_args:
            logger.info(f"Server args: {args.server_args}")
        logger.info("")
    
    # Create process manager
    process_manager = ProcessManager(log_manager)
    
    # Check if lm_eval is available
    if not check_lm_eval_available():
        logger.error("lm_eval command not found!")
        logger.error("Please ensure lm-eval is installed in your environment.")
        logger.error("Install with: pip install lm-eval")
        return 1
    
    # Setup UI manager
    ui_manager = UIManager(log_manager, mode=args.ui_mode)
    
    # Create work function
    def work_func():
        return run_gpqa_diamond_work(args, log_manager, process_manager)
    
    # Run with UI manager
    panes = {
        "server-pane": "server",
        "eval-pane": "eval",
        "script-pane": "script",
    }
    
    exit_code = run_with_ui(ui_manager, work_func, panes)
    
    # Display results after UI closes
    if exit_code == 0:
        # Create eval runner to find results
        runner = EvalRunner(
            model=args.model,
            task_name="leaderboard_gpqa_diamond",
            host=args.host,
            port=args.port,
            venv_path=args.venv,
            log_manager=log_manager,
            process_manager=process_manager,
            out_base=out_base
        )
        
        results_file = runner.find_results()
        
        if results_file and results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                
                # Display results summary
                print("\n" + "=" * 60)
                print("GPQA-Diamond Evaluation Results")
                print("=" * 60)
                
                if "results" in results_data and "gpqa_diamond" in results_data["results"]:
                    gpqa_results = results_data["results"]["gpqa_diamond"]
                    
                    # Extract key metrics
                    acc = gpqa_results.get("acc,none", "N/A")
                    acc_norm = gpqa_results.get("acc_norm,none", "N/A")
                    acc_stderr = gpqa_results.get("acc_stderr,none", "N/A")
                    acc_norm_stderr = gpqa_results.get("acc_norm_stderr,none", "N/A")
                    
                    # Format as percentages if available
                    if isinstance(acc, (int, float)):
                        acc = f"{acc * 100:.2f}%"
                    if isinstance(acc_norm, (int, float)):
                        acc_norm = f"{acc_norm * 100:.2f}%"
                    if isinstance(acc_stderr, (int, float)):
                        acc_stderr = f"±{acc_stderr * 100:.2f}%"
                    if isinstance(acc_norm_stderr, (int, float)):
                        acc_norm_stderr = f"±{acc_norm_stderr * 100:.2f}%"
                    
                    print(f"\n📊 Results:")
                    print(f"  Accuracy:            {acc} {acc_stderr}")
                    print(f"  Accuracy (norm):     {acc_norm} {acc_norm_stderr}")
                
                print(f"\n📁 Files:")
                print(f"  Results:     {results_file}")
                print(f"  Server log:  {log_manager.get_log_path('server')}")
                print(f"  Eval log:    {log_manager.get_log_path('eval')}")
                print(f"  Script log:  {log_manager.get_log_path('script')}")
                print("=" * 60)
                print()
                
            except Exception as e:
                print(f"\n⚠️  Could not parse results: {e}")
                print(f"Results file: {results_file}")
        else:
            print("\n⚠️  Results file not found or empty")
            print(f"Expected results in: {out_base}")
            print(f"Check logs for details:")
            print(f"  Eval log:    {log_manager.get_log_path('eval')}")
            print(f"  Script log:  {log_manager.get_log_path('script')}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

