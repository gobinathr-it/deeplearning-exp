"""Run one or all experiments.

Usage:
    python run_all.py            # run every experiment sequentially
    python run_all.py ex04       # run experiments whose folder starts with 'ex04'
"""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).parent
prefix = sys.argv[1] if len(sys.argv) > 1 else "ex"
failed = []

for folder in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)):
    script = next((f for f in folder.glob("*.py")), None)
    if script is None:
        continue
    print(f"\n{'=' * 70}\n{folder.name}\n{'=' * 70}")
    code = subprocess.call([sys.executable, script.name], cwd=folder)
    if code != 0:
        failed.append(folder.name)

print("\nSummary:", "all experiments completed" if not failed else f"failed -> {failed}")
sys.exit(1 if failed else 0)
