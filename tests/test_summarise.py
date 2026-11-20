"""The summary script runs on the committed result files and prints both tables."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_summarise_runs_on_committed_results():
    out = subprocess.run([sys.executable, str(ROOT / "paper1" / "summarise.py")],
                         capture_output=True, text=True, check=True).stdout
    assert "| Model | RMSE 60 |" in out
    assert "Mondrian T1DM" in out and "CVAE twin" in out
