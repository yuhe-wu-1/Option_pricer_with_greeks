"""Run the unit tests and every analysis script; each script's output is also saved to results/<name>.txt.
Usage (from the repo root):  python run_all.py"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
(ROOT / "results").mkdir(exist_ok=True)
steps = [("tests", [sys.executable, "-m", "pytest", "-q", "tests"])] + [
    (p.stem, [sys.executable, str(p)]) for p in sorted((ROOT / "analysis").glob("0*.py"))]

failed = []
for name, cmd in steps:
    print(f"\n{'=' * 30} {name} {'=' * 30}", flush=True)
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = res.stdout + res.stderr
    print(out)
    (ROOT / "results" / f"{name}.txt").write_text(out)
    if res.returncode != 0:
        failed.append(name)
print("\nAll steps finished." if not failed else f"\nFAILED: {failed}")
sys.exit(1 if failed else 0)
