import contextlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def sandbox_session():
    """
    Layer 1: /tmp isolation with a context manager.
    Guarantees cleanup of all cloned repos and SQLite databases on exit. [cite: 167, 168]
    """
    session_id = str(uuid.uuid4())
    
    # .resolve() is absolutely critical on macOS to avoid symlink comparison bugs [cite: 180, 241]
    base_dir = Path(tempfile.gettempdir()).resolve() 
    sandbox_dir = base_dir / f"revelyt-{session_id}" 
    sandbox_dir.mkdir(parents=True, exist_ok=True) 
    logger.info("Sandbox created: %s", sandbox_dir)

    try:
        yield sandbox_dir, session_id 
    finally:
        # Layer 1 Cleanup: This finally block always runs, even if the scan crashes or 
        # the developer hits Ctrl+C[cite: 172, 186].
        shutil.rmtree(sandbox_dir, ignore_errors=True)
        logger.info("Sandbox cleaned up: %s", sandbox_dir)

def run_scan_subprocess(repo_url: str, llm_provider: str, api_key: str, sandbox_dir: Path):
    """
    Layer 2: Subprocess with a completely stripped environment.
    Prevents the scan process from accessing developer credentials like AWS_SECRET_KEY. [cite: 188-190]
    """
    # We define a clean slate environment. Nothing from os.environ is inherited. [cite: 205, 207]
    clean_env = {
        'REVELYT_REPO_URL': repo_url, 
        'REVELYT_LLM_PROVIDER': llm_provider, 
        'REVELYT_API_KEY': api_key,  # Passed explicitly, never read from shell [cite: 198-199]
        'REVELYT_SANDBOX_DIR': str(sandbox_dir), 
        'PATH': '/usr/bin:/bin',       # Minimal PATH for git and python only [cite: 200, 203]
        # Notice what is missing: HOME is not passed, blocking ~/.ssh and ~/.aws [cite: 209]
    }

    # env=clean_env completely replaces the parent process's environment [cite: 205]
    return subprocess.run(
        [sys.executable, '-m', 'revelyt._internal.runner'], 
        env=clean_env, 
        capture_output=True, 
        text=True 
    )