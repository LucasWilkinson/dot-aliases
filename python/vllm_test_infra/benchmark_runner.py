"""Shared benchmark runner utilities for vLLM benchmarks."""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from .config import Config
from .git import GitManager
from .logging import LogManager
from .process import ProcessManager
from .server import VLLMServer
from .utils import parse_variants, split_args_string


class BaseBenchmarkRunner:
    """Base class for benchmark runners with shared functionality."""
    
    # Always-on server args
    ALWAYS_SERVER_ARGS = [
        "--no-enable-prefix-caching",
        "--disable-log-stats",
        "--trust-remote-code",
    ]
    
    def __init__(self, args):
        """Initialize BaseBenchmarkRunner.
        
        Args:
            args: Parsed command line arguments.
        """
        self.args = args
        
        # Setup configuration
        self.config = Config(venv_path=args.venv)
        self.config.activate_venv()
        
        # Derive terse model name
        if hasattr(args, 'terse_name') and args.terse_name:
            self.terse_model_name = args.terse_name
        else:
            self.terse_model_name = Path(args.model).name.replace("/", "_")
        
        # Parse rates
        self.rates = [float(r.strip()) for r in args.rates.split(',')]
        
        # Setup output directories
        self._setup_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Setup managers
        self.process_manager = ProcessManager(self.log_manager)
        self.git_manager = GitManager(args.repo_dir, venv_path=args.venv)
        
        # Build base server args
        self.server_args_base = self._build_base_server_args()
    
    def _setup_directories(self):
        """Setup output directories."""
        raise NotImplementedError("Subclass must implement _setup_directories")
    
    def _setup_logging(self):
        """Setup logging."""
        self.log_manager = LogManager(str(self.log_dir))
        self.log_manager.setup()
        self.log_manager.init_log_file("server")
        self.log_manager.init_log_file("bench")
        self.log_manager.init_log_file("script")
        self.log_manager.init_log_file("summary_current")
        
        self.logger = self.log_manager.get_logger("benchmark", log_to_file="script")
    
    def _parse_variant_spec(self, spec: Optional[str], default_args: str) -> List[Tuple[str, str, str]]:
        """Parse variant specification.
        
        Args:
            spec: Variant specification string.
            default_args: Default args if no spec provided.
        
        Returns:
            List of (label, args, env_csv) tuples.
        """
        if spec:
            return parse_variants(spec)
        else:
            # No variants specified, use single "base" variant with default args
            return [("base", default_args, "")]
    
    def _build_base_server_args(self) -> str:
        """Build base server arguments."""
        base_args = split_args_string(self.args.server_args_base)
        
        # Add -tp if specified and not already in args
        if hasattr(self.args, 'tensor_parallel_size') and self.args.tensor_parallel_size > 1:
            if not any('-tp' in arg or '--tensor-parallel-size' in arg for arg in base_args):
                base_args.extend(["-tp", str(self.args.tensor_parallel_size)])
        
        # Add always-on args
        base_args.extend(self.ALWAYS_SERVER_ARGS)
        
        return ' '.join(base_args)
    
    def _get_result_filename(self, variant: str, rate: float) -> str:
        """Generate result filename.
        
        Args:
            variant: Variant label.
            rate: Request rate.
        
        Returns:
            Result filename.
        """
        num_prompts = int(rate * self.args.run_seconds)
        ds_safe = re.sub(r'[^A-Za-z0-9._-]+', '-', self.args.dataset)
        
        suffix = f"_{self.args.label_suffix}" if hasattr(self.args, 'label_suffix') and self.args.label_suffix else ""
        
        return (
            f"bench_model-{self.terse_model_name}_rate-{rate}_v-{variant}_"
            f"np-{num_prompts}_in-{self.args.random_in}_out-{self.args.random_out}_"
            f"ds-{ds_safe}{suffix}.json"
        )
    
    def _result_exists(self, results_dir: Path, variant: str, rate: float) -> bool:
        """Check if result file exists and is non-empty.
        
        Args:
            results_dir: Results directory.
            variant: Variant label.
            rate: Request rate.
        
        Returns:
            True if result exists.
        """
        filename = self._get_result_filename(variant, rate)
        filepath = results_dir / filename
        return filepath.exists() and filepath.stat().st_size > 0
    
    def run_client_for_rate(
        self,
        results_dir: Path,
        branch_label: str,
        variant: str,
        rate: float,
        force_rerun: bool = False
    ) -> bool:
        """Run benchmark client for a specific rate.
        
        Args:
            results_dir: Directory to save results.
            branch_label: Branch label for logging.
            variant: Variant label.
            rate: Request rate.
            force_rerun: Force rerun even if result exists.
        
        Returns:
            True if successful.
        """
        import os
        
        num_prompts = int(rate * self.args.run_seconds)
        filename = self._get_result_filename(variant, rate)
        filepath = results_dir / filename
        
        # Check if we should skip
        if self._result_exists(results_dir, variant, rate) and not force_rerun:
            self.logger.info(f"Reusing existing result for {branch_label}/{variant} rate={rate}")
            return True
        
        # Build vllm bench command
        vllm_cmd = os.path.join(self.args.venv, "bin", "vllm")
        
        command = [
            vllm_cmd, "bench", "serve",
            "--model", self.args.model,
            "--host", self.args.host,
            "--port", str(self.args.port),
            "--dataset-name", self.args.dataset,
            "--random-input-len", str(self.args.random_in),
            "--random-output-len", str(self.args.random_out),
            "--random-range-ratio", str(self.args.random_range_ratio),
            "--num-prompts", str(num_prompts),
            "--request-rate", str(rate),
            "--save-result",
            "--result-dir", str(results_dir),
            "--result-filename", filename,
            "--seed", "42",
            "--ignore-eos",
            "--trust-remote-code",
        ]
        
        self.logger.info(f"Running client: {branch_label}/{variant} rate={rate} prompts={num_prompts}")
        self.logger.info("=" * 60)
        self.logger.info(f"Benchmark Command (for manual reproduction):")
        self.logger.info(f"  {' '.join(command)}")
        self.logger.info("=" * 60)
        
        try:
            result = self.process_manager.run(
                name=f"bench_{branch_label}_{variant}_{rate}",
                command=command,
                log_file="bench",
                timeout=self.args.run_seconds * 3  # 3x safety margin
            )
            
            if result.returncode == 0:
                self.logger.info(f"Completed {branch_label}/{variant} rate={rate}")
                return True
            else:
                self.logger.error(f"Benchmark failed for {branch_label}/{variant} rate={rate}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error running benchmark: {e}")
            return False
    
    def run_variant(
        self,
        results_dir: Path,
        branch_label: str,
        variant_label: str,
        variant_args: str,
        variant_env: str,
        force_rerun: bool = False
    ) -> None:
        """Run benchmarks for a single variant.
        
        Args:
            results_dir: Results directory.
            branch_label: Branch label for logging.
            variant_label: Variant label.
            variant_args: Variant-specific server args.
            variant_env: Variant-specific environment variables.
            force_rerun: Force rerun even if results exist.
        """
        self.logger.info(f"Starting variant: {branch_label}/{variant_label}")
        
        # Check if all results already exist and we're not forcing rerun
        if not force_rerun and hasattr(self.args, 'resume') and self.args.resume:
            all_exist = all(
                self._result_exists(results_dir, variant_label, rate)
                for rate in self.rates
            )
            
            if all_exist:
                self.logger.info(f"Skipping {branch_label}/{variant_label} (all results present)")
                return
        
        # Build full server args
        full_args = f"{self.server_args_base} {variant_args}".strip()
        
        # Start server
        server = VLLMServer(
            model=self.args.model,
            host=self.args.host,
            port=self.args.port,
            venv_path=self.args.venv,
            log_manager=self.log_manager,
            process_manager=self.process_manager
        )
        
        with server:
            self.logger.info(f"Starting server for {branch_label}/{variant_label}")
            
            # Log the exact server command for reproducibility
            server_cmd = f"vllm serve {self.args.model} --host {self.args.host} --port {self.args.port} {full_args}"
            if variant_env:
                env_vars = " ".join([f"{k}={v}" for k, v in [pair.split("=", 1) for pair in variant_env.split(",") if pair]])
                server_cmd = f"{env_vars} {server_cmd}"
            self.logger.info("=" * 60)
            self.logger.info(f"Server Command (for manual reproduction):")
            self.logger.info(f"  {server_cmd}")
            self.logger.info("=" * 60)
            
            server.start(args=full_args, env_csv=variant_env, log_file="server")
            
            if not server.wait_for_ready(timeout=600):
                self.logger.error(f"Server failed to start for {branch_label}/{variant_label}")
                return
            
            # Run benchmarks for each rate
            for rate in self.rates:
                self.run_client_for_rate(results_dir, branch_label, variant_label, rate, force_rerun)
        
        self.logger.info(f"Completed variant: {branch_label}/{variant_label}")
    
    def run_branch(
        self,
        branch_label: str,
        ref: str,
        variants: List[Tuple[str, str, str]],
        results_dir: Path,
        build: bool,
        pull: bool,
        force_rerun: bool = False
    ) -> None:
        """Run benchmarks for a branch.
        
        Args:
            branch_label: Branch label (MAIN, PR, etc.).
            ref: Git reference.
            variants: List of (label, args, env) tuples.
            results_dir: Results directory.
            build: Whether to rebuild after checkout.
            pull: Whether to pull latest changes.
            force_rerun: Force rerun even if results exist.
        """
        self.logger.info(f"=" * 60)
        self.logger.info(f"Running {branch_label} branch: {ref}")
        self.logger.info(f"=" * 60)
        
        # Checkout branch
        self.git_manager.checkout(ref)
        
        # Pull if requested
        if pull:
            self.git_manager.pull(ref)
        
        # Build if requested
        if build:
            self.git_manager.build()
        
        # Run each variant
        for variant_label, variant_args, variant_env in variants:
            self.run_variant(results_dir, branch_label, variant_label, variant_args, variant_env, force_rerun)

