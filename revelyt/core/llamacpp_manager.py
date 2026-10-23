"""
llamacpp_manager — Embedded llama.cpp server management for Revelyt.

Handles the full lifecycle:
1. Install llama.cpp (Homebrew on macOS, pip on Linux)
2. Download model from Hugging Face if not present
3. Start llama-server as background subprocess
4. Wait for server to be ready
5. Clean up on exit

This means users can run:
  revelyt scan ./repo --provider llama-cpp-auditor

And everything just works — no manual setup required.
"""

import os
import platform
import shutil
import subprocess
import time
import signal
import logging
from pathlib import Path
from typing import Optional

from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

# Default configuration
DEFAULT_MODEL_REPO = "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF"
DEFAULT_MODEL_FILE = "qwen2.5-coder-14b-instruct-q4_k_m.gguf"

SUMMARY_MODEL_REPO = "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"
SUMMARY_MODEL_FILE = "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
DEFAULT_PORT = 8080
DEFAULT_CTX_PER_SLOT = 8192  # per-slot context; total = per_slot × parallel
DEFAULT_THREADS = 10
DEFAULT_GPU_LAYERS = 40

# KV-cache quantization — 4-bit "TurboQuant" style optimization
# Reduces KV memory footprint massively, enabling much higher concurrency.
DEFAULT_CACHE_TYPE_K = "q4_0"
DEFAULT_CACHE_TYPE_V = "q4_0"

# Auditor-specific parameters (matches qwen2.5-auditor Modelfile)
AUDITOR_PARAMS = {
    "temperature": 0.0,     # deterministic — critical for security auditing
    "top_k": 10,
    "top_p": 1.0,
    "repeat_penalty": 1.15,
    "repeat_last_n": 256,
    "seed": 42,
}

# Server startup timeout (seconds)
SERVER_STARTUP_TIMEOUT = 120


class LlamaCppManager:
    """Manages the embedded llama.cpp server lifecycle."""

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        port: int = DEFAULT_PORT,
        auto_install: bool = True,
        auto_download: bool = True,
        parallel: int = 5,          # bumped to 5! (5 slots * ~0.25GB KV) = ~1.25 GB KV overhead + 9GB Model
        model_repo: str = DEFAULT_MODEL_REPO,
        model_file: str = DEFAULT_MODEL_FILE,
    ):
        self.model_dir = model_dir or Path.home() / ".revelyt" / "models"
        self.port = port
        self.auto_install = auto_install
        self.auto_download = auto_download
        self.parallel = parallel
        self.model_repo = model_repo
        self.model_file = model_file
        self.model_path = self.model_dir / self.model_file
        self.server_process: Optional[subprocess.Popen] = None
        self.server_url = f"http://localhost:{port}"

        # Ensure model directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def ensure_ready(self) -> str:
        """
        Ensure llama.cpp server is running and ready.

        Returns:
            Server URL (e.g., "http://localhost:8080")
        """
        # Step 1: Check/install llama.cpp
        if not self._is_llamacpp_installed():
            if self.auto_install:
                self._install_llamacpp()
            else:
                raise RuntimeError(
                    "llama.cpp is not installed. "
                    "Install it manually: brew install llama.cpp (macOS) "
                    "or pip install llama-cpp-python (Linux)"
                )

        # Step 2: Check/download model
        if not self.model_path.exists():
            if self.auto_download:
                self._download_model()
            else:
                raise RuntimeError(
                    f"Model not found: {self.model_path}\n"
                    f"Download it manually from {DEFAULT_MODEL_REPO}"
                )

        # Step 3: Check if server is already running WITH adequate context
        if self._is_server_healthy():
            ctx_ok = self._check_server_context()
            if ctx_ok:
                console.print(
                    f"   ↳ ⚡ [green]llama.cpp server already running on port {self.port}[/green]"
                )
                return self.server_url
            else:
                # Server running but with wrong context — kill it first
                console.print(
                    f"   ↳ [yellow]Existing server has insufficient context — restarting...[/yellow]"
                )
                self._kill_existing_server()

        # Step 4: Start server (port is guaranteed free at this point)
        self._start_server()
        return self.server_url

    def shutdown(self):
        """Gracefully shutdown the llama.cpp server."""
        if self.server_process:
            console.print("   ↳ [yellow]Shutting down llama.cpp server...[/yellow]")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            console.print("   ↳ ✅ [green]llama.cpp server stopped[/green]")
            self.server_process = None

    def load_model(self, model_repo: str, model_file: str) -> str:
        """
        Dynamically swap the loaded model.
        Shuts down the current server, downloads the new model if necessary,
        and starts a new server instance.
        """
        if self.model_repo == model_repo and self.model_file == model_file and self._is_server_healthy():
             return self.server_url # Already running with correct model
             
        console.print(f"   ↳ 🔄 [cyan]Swapping model to {model_file}...[/cyan]")
        
        self.shutdown()
        
        self.model_repo = model_repo
        self.model_file = model_file
        self.model_path = self.model_dir / self.model_file
        
        return self.ensure_ready()

    def _is_llamacpp_installed(self) -> bool:
        """Check if llama-server binary is available."""
        return shutil.which("llama-server") is not None

    def _install_llamacpp(self):
        """Install llama.cpp via Homebrew (macOS) or pip (Linux)."""
        system = platform.system()

        if system == "Darwin":
            console.print("   ↳ [cyan]Installing llama.cpp via Homebrew...[/cyan]")
            try:
                subprocess.run(
                    ["brew", "install", "llama.cpp"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                console.print("   ↳ ✅ [green]llama.cpp installed successfully[/green]")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to install llama.cpp via Homebrew: {e.stderr}"
                ) from e
        elif system == "Linux":
            console.print("   ↳ [cyan]Installing llama-cpp-python via pip...[/cyan]")
            try:
                subprocess.run(
                    ["pip", "install", "llama-cpp-python"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                console.print(
                    "   ↳ ✅ [green]llama-cpp-python installed successfully[/green]"
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to install llama-cpp-python: {e.stderr}"
                ) from e
        else:
            raise RuntimeError(
                f"Unsupported platform: {system}. Please install llama.cpp manually."
            )

    def _download_model(self):
        """Download the model from Hugging Face."""
        console.print(
            f"   ↳ [cyan]Downloading model {self.model_file} from Hugging Face...[/cyan]"
        )
        console.print(
            "   ↳ [dim]This is a one-time download. Subsequent runs will use the cached model.[/dim]"
        )

        try:
            from huggingface_hub import hf_hub_download

            downloaded_path = hf_hub_download(
                repo_id=self.model_repo,
                filename=self.model_file,
                local_dir=self.model_dir,
            )
            console.print(
                f"   ↳ ✅ [green]Model downloaded to {downloaded_path}[/green]"
            )
        except ImportError:
            raise RuntimeError(
                "huggingface-hub is required for model download.\n"
                "Install it: pip install huggingface-hub"
            ) from None
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}") from e

    def _is_server_healthy(self) -> bool:
        """Pure health check — returns True if a server responds on the port. No side effects."""
        try:
            import urllib.request

            url = f"{self.server_url}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _check_server_context(self) -> bool:
        """
        Check if the running server has sufficient context size.

        Returns True if context is adequate or if we can't determine it.
        Returns False if context is confirmed to be too small.
        """
        try:
            import urllib.request
            import json

            props_url = f"{self.server_url}/props"
            with urllib.request.urlopen(props_url, timeout=2) as resp:
                props = json.loads(resp.read())
                actual_ctx = props.get("default_generation_settings", {}).get("n_ctx", 0)
                required_total = DEFAULT_CTX_PER_SLOT * self.parallel
                if actual_ctx and actual_ctx < required_total * 0.8:
                    logger.warning(
                        "Existing server has n_ctx=%d, need >=%d (per-slot %d × %d parallel)",
                        actual_ctx, required_total, DEFAULT_CTX_PER_SLOT, self.parallel,
                    )
                    console.print(
                        f"   ↳ [yellow]Existing server n_ctx={actual_ctx}, "
                        f"need {required_total} ({DEFAULT_CTX_PER_SLOT}/slot × {self.parallel} parallel)[/yellow]"
                    )
                    return False
        except Exception:
            pass  # /props not available — assume context is fine
        return True

    def _kill_existing_server(self):
        """Kill any existing listening process on our port and wait for it to release."""
        try:
            result = subprocess.run(
                ["lsof", "-tiTCP", f":{self.port}", "-sTCP:LISTEN"],
                capture_output=True, text=True,
            )
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid.strip():
                    os.kill(int(pid.strip()), signal.SIGTERM)
                    logger.info("Killed existing llama-server process: PID %s", pid.strip())
            time.sleep(2)  # Wait for port to be released
        except Exception as e:
            logger.warning("Could not kill existing server: %s", e)

    def _start_server(self):
        """Start llama-server as a background subprocess."""
        # Ensure port is free before starting (belt-and-suspenders)
        self._kill_existing_server()

        console.print(
            f"   ↳ [cyan]Starting llama.cpp server on port {self.port}...[/cyan]"
        )

        cmd = [
            "llama-server",
            # ── Model ────────────────────────────────────────────────
            "--model",          str(self.model_path),
            # ── Hardware ─────────────────────────────────────────────
            "--threads",        str(DEFAULT_THREADS),
            "--gpu-layers",     str(DEFAULT_GPU_LAYERS),
            # ── Context / concurrency ────────────────────────────────
            "--port",           str(self.port),
            "--ctx-size",       str(DEFAULT_CTX_PER_SLOT * self.parallel),
            "--parallel",       str(self.parallel),
            "-cb",              # continuous batching
            # ── KV-cache quantization (halves KV memory vs FP16) ─────
            "--cache-type-k",   DEFAULT_CACHE_TYPE_K,
            "--cache-type-v",   DEFAULT_CACHE_TYPE_V,
            # ── Sampling (auditor-tuned, deterministic) ───────────────
            "--temp",           str(AUDITOR_PARAMS["temperature"]),
            "--top-k",          str(AUDITOR_PARAMS["top_k"]),
            "--top-p",          str(AUDITOR_PARAMS["top_p"]),
            "--repeat-penalty", str(AUDITOR_PARAMS["repeat_penalty"]),
            "--repeat-last-n",  str(AUDITOR_PARAMS["repeat_last_n"]),
            "--seed",           str(AUDITOR_PARAMS["seed"]),
            "--log-disable",    # suppress server log noise in CLI output
        ]

        # Log the exact command for debuggability
        logger.info("llama-server command: %s", " ".join(cmd))

        # Start server in background, logging to /tmp/llama_server.log
        self.server_log = open("/tmp/llama_server.log", "w")
        self.server_process = subprocess.Popen(
            cmd,
            stdout=self.server_log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if platform.system() != "Windows" else None,
        )

        # Wait for server to be ready (pure health check only — no kill logic)
        console.print("   ↳ [dim]Waiting for server to load model...[/dim]")
        start_time = time.time()
        while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
            if self._is_server_healthy():
                elapsed = time.time() - start_time
                console.print(
                    f"   ↳ ✅ [green]llama.cpp server ready ({elapsed:.0f}s)[/green]"
                )
                return
            time.sleep(1)

        # Timeout — shut down cleanly and surface a clear error
        self.shutdown()
        raise RuntimeError(
            f"llama.cpp server failed to start within {SERVER_STARTUP_TIMEOUT}s. "
            f"Check if port {self.port} is available and model file is valid. "
            f"Server log: /tmp/llama_server.log"
        )

    def __enter__(self):
        self.ensure_ready()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False