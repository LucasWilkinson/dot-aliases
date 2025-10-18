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
    LogManager,
    ProcessManager,
    VLLMServer,
    UIManager,
    note,
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
    # Get logger
    logger = log_manager.get_logger("gsm8k_eval", log_to_file="script")
    
    # Determine output paths
    out_base = Path(args.out_base).resolve()
    terse_model_name = Path(args.model).name
    results_file = out_base / f"results_{terse_model_name}.json"
    
    try:
        # Create and start vLLM server
        logger.info("Starting vLLM server...")
        server = VLLMServer(
            model=args.model,
            host=args.host,
            port=args.port,
            venv_path=args.venv,
            log_manager=log_manager,
            process_manager=process_manager
        )
        
        with server:
            server.start(args=args.server_args, log_file="server")
            
            # Wait for server to be ready
            if not server.wait_for_ready(timeout=600):
                logger.error("Server failed to start within timeout")
                logger.error(f"Check server log: {log_manager.get_log_path('server')}")
                logger.error("")
                logger.error("=" * 60)
                logger.error("Last 200 lines of server log:")
                logger.error("=" * 60)
                server_log_tail = log_manager.tail_file('server', num_lines=200)
                logger.error(server_log_tail)
                logger.error("=" * 60)
                return 1
            
            # Run GSM8K evaluation
            logger.info("")
            logger.info("=" * 50)
            logger.info("Running GSM8K evaluation...")
            logger.info("=" * 50)
            logger.info("")
            
            # Build lm_eval command
            eval_cmd = [
                "lm_eval",
                "--model", "local-completions",
                "--model_args", (
                    f"model={args.model},"
                    f"base_url=http://{args.host}:{args.port}/v1/completions,"
                    f"num_concurrent={args.num_concurrent}"
                ),
                "--tasks", "gsm8k",
                "--output_path", str(out_base),
                "--log_samples",
                "--batch_size", args.batch_size,
            ]
            
            if args.limit:
                eval_cmd.extend(["--limit", str(args.limit)])
            
            logger.info(f"Command: {' '.join(eval_cmd)}")
            logger.info("")
            
            # Run evaluation
            try:
                result = process_manager.run(
                    name="lm_eval",
                    command=eval_cmd,
                    log_file="eval",
                    timeout=3600  # 1 hour timeout
                )
                
                if result.returncode == 0:
                    logger.info("")
                    logger.info("=" * 50)
                    logger.info("Evaluation complete!")
                    logger.info("=" * 50)
                    logger.info(f"Results: {results_file}")
                    logger.info(f"Server log: {log_manager.get_log_path('server')}")
                    logger.info(f"Eval log: {log_manager.get_log_path('eval')}")
                    return 0
                else:
                    logger.error("")
                    logger.error("=" * 50)
                    logger.error(f"Evaluation failed with status: {result.returncode}")
                    logger.error("=" * 50)
                    return result.returncode
            
            except Exception as e:
                logger.error(f"Evaluation error: {e}")
                return 1
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
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
    
    # Create work function that captures args and managers
    def work_func():
        return run_gsm8k_work(args, log_manager, process_manager)
    
    # Run with appropriate UI mode
    panes = {
        "server-pane": "server",
        "eval-pane": "eval",
        "script-pane": "script",
    }
    
    exit_code = run_with_ui(ui_manager, work_func, panes)
    
    # Print summary after UI closes
    print()
    print("=" * 60)
    if exit_code == 0:
        print("✅ Evaluation completed successfully!")
        print()
        
        # Try to read and display results
        import json
        import glob
        
        # lm_eval creates results in format: model_name/results_*.json or results_YYYY-MM-DD_HH-MM-SS/
        results_found = False
        
        # Try finding results files in subdirectories first
        result_files = sorted(glob.glob(str(out_base / "*" / "results_*.json")), reverse=True)
        
        # Also try legacy format: results_YYYY-MM-DD_HH-MM-SS/results.json
        result_files.extend(sorted(glob.glob(str(out_base / "results_*" / "results.json")), reverse=True))
        result_files.extend(sorted(glob.glob(str(out_base / "results_*" / "gsm8k.json")), reverse=True))
        
        for result_file in result_files:
            json_path = Path(result_file)
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        results = data.get('results', {})
                        
                        # GSM8K can be under different keys
                        gsm8k = results.get('gsm8k', results.get('gsm8k_cot', {}))
                        
                        if gsm8k:
                            print("📊 GSM8K Results:")
                            print("   " + "-" * 55)
                            
                            # Try different possible metric names
                            for metric_key in ['exact_match,strict-match', 'exact_match,flexible-extract', 'exact_match']:
                                if metric_key in gsm8k:
                                    acc = gsm8k[metric_key]
                                    stderr_key = metric_key.replace('exact_match', 'exact_match_stderr')
                                    stderr = gsm8k.get(stderr_key, 0)
                                    
                                    metric_name = metric_key.replace(',', ' ')
                                    if isinstance(acc, float):
                                        print(f"   {metric_name:40s}: {acc*100:6.2f}% ± {stderr*100:5.2f}%")
                                    else:
                                        print(f"   {metric_name:40s}: {acc}")
                            
                            # Show other metrics if present
                            for key, value in gsm8k.items():
                                if 'stderr' not in key and 'alias' not in key.lower() and not key.startswith('exact_match'):
                                    if isinstance(value, float):
                                        print(f"   {key:40s}: {value*100:6.2f}%")
                            
                            print("   " + "-" * 55)
                            results_found = True
                            break
                except Exception as e:
                    print(f"   ⚠️  Error reading results: {e}")
                
            if results_found:
                break
        
        if not results_found:
            print("   ⚠️  Results file not found or empty")
        
        print()
        print(f"📁 Results directory: {out_base}")
    else:
        print("❌ Evaluation failed!")
    
    print(f"📋 Logs: {log_dir}")
    print(f"   Server log: {log_dir}/server.log")
    print(f"   Eval log: {log_dir}/eval.log")
    print("=" * 60)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
