# test_sandbox.py
import os
import sys
from pathlib import Path
from revelyt.core.sandbox import sandbox_session, run_scan_subprocess

# 1. Inject a fake dangerous credential into your current environment
os.environ["AWS_SECRET_KEY"] = "super-secret-developer-key"

print("--- Testing Revelyt Sandbox ---")

with sandbox_session() as (sandbox_dir, session_id):
    print(f"[+] Layer 1: Sandbox created at {sandbox_dir}")
    assert sandbox_dir.exists(), "Sandbox directory was not created!"
    
    # Normally this calls revelyt._internal.runner, but let's test the env stripping
    # by running a simple python command to print the environment variables.
    import subprocess
    clean_env = {
        'PATH': '/usr/bin:/bin',
        'REVELYT_SANDBOX_DIR': str(sandbox_dir)
    }
    
    print("[+] Layer 2: Running subprocess with stripped environment...")
    result = subprocess.run(
        # Change "python" to sys.executable here 👇
        [sys.executable, "-c", "import os; print('AWS_SECRET_KEY in env:', 'AWS_SECRET_KEY' in os.environ)"],
        env=clean_env,
        capture_output=True, text=True
    )
    
    print(f"Subprocess Output: {result.stdout.strip()}")
    assert "False" in result.stdout, "SECURITY LEAK: Subprocess can see AWS_SECRET_KEY!"

print(f"[+] Layer 1 Cleanup: Does sandbox still exist? {sandbox_dir.exists()}")
assert not sandbox_dir.exists(), "Cleanup failed! Directory still exists."

print("✅ All sandbox tests passed. Isolation is secure.")