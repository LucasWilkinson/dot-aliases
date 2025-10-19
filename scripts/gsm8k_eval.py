#!/usr/bin/env python3
"""GSM8K Evaluation Script using vLLM server and lm_eval."""

import argparse
import sys
import os
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
    setup_signal_handlers,
)
from python.vllm_test_infra.ui import run_with_ui


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GSM8K evaluation using vLLM server and lm_eval"
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
        default="./gsm8k-results",
        help="Output directory (default: ./gsm8k-results)"
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


def run_gsm8k_work(args, log_manager: LogManager, process_manager: ProcessManager) -> int:
    """Run GSM8K evaluation work.
    
    This function can be called from main thread (simple mode) or worker thread (TUI mode).
    
    Args:
        args: Parsed command line arguments.
        log_manager: LogManager instance.
        process_manager: ProcessManager instance.
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    # Determine output paths
    out_base = Path(args.out_base).resolve()
    
    # Create eval runner
    runner = EvalRunner(
        model=args.model,
        task_name="gsm8k",
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
    # Setup signal handlers FIRST for graceful shutdown
    setup_signal_handlers()
    
    args = parse_args()
    
    # Setup configuration
    config = Config(venv_path=args.venv)
    config.activate_venv()
    
    # Determine output paths
    out_base = Path(args.out_base).resolve()
    log_dir = out_base / "logs"
    terse_model_name = Path(args.model).name
    results_file = out_base / f"results_{terse_model_name}.json"
    
    # Setup logging
    log_manager = LogManager(str(log_dir))
    log_manager.setup()
    log_manager.init_log_file("server")
    log_manager.init_log_file("eval")
    log_manager.init_log_file("script")
    
    # Get logger for script
    logger = log_manager.get_logger("gsm8k_eval", log_to_file="script")
    
    # Log configuration info (only in simple mode, TUI will show in logs)
    if args.ui_mode != "tui":
        logger.info("=" * 50)
        logger.info("GSM8K Evaluation")
        logger.info("=" * 50)
        logger.info(f"Model: {args.model}")
        logger.info(f"Port: {args.port}")
        logger.info(f"Limit: {args.limit if args.limit else 'none'}")
        logger.info(f"Concurrent: {args.num_concurrent}")
        logger.info(f"Server args: {args.server_args if args.server_args else 'none'}")
        logger.info(f"Results: {results_file}")
        logger.info(f"Logs: {log_dir}")
        logger.info("=" * 50)
    
    # Check for lm_eval
    if not check_lm_eval_available():
        print("ERROR: lm_eval not found! Install with: pip install lm-eval", file=sys.stderr)
        sys.exit(1)
    
    # Setup process manager
    process_manager = ProcessManager(log_manager)
    
    # Setup UI manager
    ui_manager = UIManager(log_manager, mode=args.ui_mode)
    
    # Create work function
    def work_func():
        return run_gsm8k_work(args, log_manager, process_manager)
    
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
            task_name="gsm8k",
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
                import json
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                
                # Display results summary
                print("\n" + "=" * 60)
                print("GSM8K Evaluation Results")
                print("=" * 60)
                
                if "results" in results_data:
                    results = results_data["results"]
                    # GSM8K can be under different keys
                    gsm8k = results.get('gsm8k', results.get('gsm8k_cot', {}))
                    
                    if gsm8k:
                        print("\n📊 Results:")
                        
                        # Try different possible metric names
                        for metric_key in ['exact_match,strict-match', 'exact_match,flexible-extract', 'exact_match']:
                            if metric_key in gsm8k:
                                acc = gsm8k[metric_key]
                                stderr_key = metric_key.replace('exact_match', 'exact_match_stderr')
                                stderr = gsm8k.get(stderr_key, 0)
                                
                                metric_name = metric_key.replace(',', ' ').replace('exact_match', 'Accuracy')
                                if isinstance(acc, float):
                                    print(f"  {metric_name:25s}: {acc*100:6.2f}% ± {stderr*100:5.2f}%")
                                else:
                                    print(f"  {metric_name:25s}: {acc}")
                
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
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
