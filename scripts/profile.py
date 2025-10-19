#!/usr/bin/env python3
"""Profiling Script for vLLM using torch profiler."""

import argparse
import sys
import os
import glob
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.vllm_test_infra import (
    Config,
    LogManager,
    ProcessManager,
    VLLMServer,
    note,
    setup_signal_handlers,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile vLLM server using torch profiler"
    )
    
    parser.add_argument(
        "--model",
        required=True,
        help="Model to profile (HF id or local path)"
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
        "--random-input-len",
        type=int,
        default=1000,
        help="Random input length (default: 1000)"
    )
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=100,
        help="Random output length (default: 100)"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=50,
        help="Number of prompts (default: 50)"
    )
    parser.add_argument(
        "-tp",
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)"
    )
    parser.add_argument(
        "--server-args",
        default="",
        help="Additional vllm serve arguments (quoted string)"
    )
    parser.add_argument(
        "--profile-dir",
        default="./vllm-profiles",
        help="Output directory for profiler traces (default: ./vllm-profiles)"
    )
    parser.add_argument(
        "--venv",
        default=".venv",
        help="Path to virtual environment (default: .venv)"
    )
    
    return parser.parse_args()


def check_vllm_bench_available() -> bool:
    """Check if vllm bench is available."""
    import shutil
    return shutil.which("vllm") is not None


def run_profiling(args, log_manager: LogManager, process_manager: ProcessManager) -> int:
    """Run profiling work.
    
    Args:
        args: Parsed command line arguments.
        log_manager: LogManager instance.
        process_manager: ProcessManager instance.
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    logger = log_manager.get_logger("profile", log_to_file="script")
    
    # Setup profile directory
    profile_dir = Path(args.profile_dir).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean old traces
    logger.info(f"Cleaning old traces in {profile_dir}")
    for old_trace in profile_dir.glob("*.pt.trace.json.gz"):
        old_trace.unlink()
        logger.info(f"Removed old trace: {old_trace.name}")
    
    # Set environment variables for profiling
    os.environ["VLLM_TORCH_PROFILER_DIR"] = str(profile_dir)
    os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
    
    logger.info("=" * 60)
    logger.info("Profiling Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Tensor Parallel: {args.tensor_parallel_size}")
    logger.info(f"  Input Length: {args.random_input_len}")
    logger.info(f"  Output Length: {args.random_output_len}")
    logger.info(f"  Num Prompts: {args.num_prompts}")
    logger.info(f"  Profile Dir: {profile_dir}")
    logger.info("=" * 60)
    
    try:
        # Build server args with TP
        server_args_list = ["-tp", str(args.tensor_parallel_size)]
        if args.server_args:
            server_args_list.extend(args.server_args.split())
        server_args = " ".join(server_args_list)
        
        # Log server command
        server_cmd = f"vllm serve {args.model} --host {args.host} --port {args.port} {server_args}"
        logger.info("=" * 60)
        logger.info("Server Command (for manual reproduction):")
        logger.info(f"  export VLLM_TORCH_PROFILER_DIR={profile_dir}")
        logger.info(f"  export VLLM_ALLREDUCE_USE_SYMM_MEM=0")
        logger.info(f"  {server_cmd}")
        logger.info("=" * 60)
        
        # Start vLLM server
        logger.info("Starting vLLM server with profiling enabled...")
        server = VLLMServer(
            model=args.model,
            host=args.host,
            port=args.port,
            venv_path=args.venv,
            log_manager=log_manager,
            process_manager=process_manager
        )
        
        with server:
            server.start(args=server_args, log_file="server")
            
            # Wait for server to be ready
            if not server.wait_for_ready(timeout=600):
                logger.error("Server failed to start within timeout")
                logger.error(f"Check server log: {log_manager.get_log_path('server')}")
                return 1
            
            logger.info("")
            logger.info("=" * 60)
            logger.info("Running profiler benchmark...")
            logger.info("=" * 60)
            logger.info("")
            
            # Build vllm bench command
            vllm_cmd = os.path.join(args.venv, "bin", "vllm")
            bench_cmd = [
                vllm_cmd, "bench", "serve",
                "--model", args.model,
                "--backend", "openai",
                "--base-url", f"http://{args.host}:{args.port}",
                "--endpoint", "/v1/completions",
                "--random-input-len", str(args.random_input_len),
                "--random-output-len", str(args.random_output_len),
                "--num-prompts", str(args.num_prompts),
                "--profile",
            ]
            
            logger.info("=" * 60)
            logger.info("Benchmark Command (for manual reproduction):")
            logger.info(f"  export VLLM_ALLREDUCE_USE_SYMM_MEM=0")
            logger.info(f"  {' '.join(bench_cmd)}")
            logger.info("=" * 60)
            logger.info("")
            
            # Run benchmark with profiling
            try:
                result = process_manager.run(
                    name="vllm_bench",
                    command=bench_cmd,
                    log_file="bench",
                    timeout=600  # 10 minute timeout
                )
                
                if result.returncode == 0:
                    logger.info("")
                    logger.info("=" * 60)
                    logger.info("Profiling complete!")
                    logger.info("=" * 60)
                    
                    # Check for profile traces
                    traces = list(profile_dir.glob("*.pt.trace.json.gz"))
                    if traces:
                        logger.info(f"Found {len(traces)} trace files:")
                        for trace in sorted(traces):
                            size = trace.stat().st_size
                            size_mb = size / (1024 * 1024)
                            logger.info(f"  {trace.name}: {size_mb:.2f} MB")
                            
                            if size == 0:
                                logger.warning(f"  ⚠️  WARNING: {trace.name} is empty (0 bytes)!")
                        
                        # Check for rank-specific traces
                        rank_traces = list(profile_dir.glob("*-rank-*.pt.trace.json.gz"))
                        if rank_traces:
                            logger.info(f"Found {len(rank_traces)} rank-specific traces")
                        else:
                            logger.warning("No rank-specific traces found (expected for TP>1)")
                    else:
                        logger.warning("⚠️  No trace files found!")
                        logger.warning(f"Expected traces in: {profile_dir}")
                    
                    logger.info(f"Profile directory: {profile_dir}")
                    logger.info(f"Server log: {log_manager.get_log_path('server')}")
                    logger.info(f"Bench log: {log_manager.get_log_path('bench')}")
                    return 0
                else:
                    logger.error("")
                    logger.error("=" * 60)
                    logger.error(f"Profiling failed with status: {result.returncode}")
                    logger.error("=" * 60)
                    return result.returncode
            
            except Exception as e:
                logger.error(f"Profiling error: {e}")
                return 1
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    # Setup signal handlers FIRST for graceful shutdown
    setup_signal_handlers()
    
    args = parse_args()
    
    # Setup configuration
    config = Config(venv_path=args.venv)
    config.activate_venv()
    
    # Determine output paths
    profile_dir = Path(args.profile_dir).resolve()
    log_dir = profile_dir / "logs"
    
    # Setup logging
    log_manager = LogManager(str(log_dir))
    log_manager.setup()
    log_manager.init_log_file("server")
    log_manager.init_log_file("bench")
    log_manager.init_log_file("script")
    
    # Get logger for script
    logger = log_manager.get_logger("profile", log_to_file="script")
    
    logger.info("=" * 60)
    logger.info("vLLM Profiling")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Profile Dir: {profile_dir}")
    logger.info(f"Server: http://{args.host}:{args.port}")
    logger.info("")
    
    # Create process manager
    process_manager = ProcessManager(log_manager)
    
    # Check if vllm bench is available
    if not check_vllm_bench_available():
        logger.error("vllm command not found!")
        logger.error("Please ensure vllm is installed in your environment.")
        return 1
    
    # Run profiling
    exit_code = run_profiling(args, log_manager, process_manager)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

