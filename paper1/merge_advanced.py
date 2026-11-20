"""Merge the per-process result files of p1_advanced.py (seeds 0-3 and 4-7) into one file."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parts = sorted((ROOT / "results").glob("paper1_adv_[a-z].json"))
runs, cfg = [], None
for p in parts:
    d = json.load(open(p)); cfg = cfg or d["config"]; runs += d["runs"]
runs.sort(key=lambda r: r["seed"])
seeds = [r["seed"] for r in runs]
assert len(seeds) == len(set(seeds)), "duplicate seeds"
json.dump(dict(config=cfg, dataset="ShanghaiT1DM+ShanghaiT2DM", runs=runs), open(ROOT / "results" / "paper1_advanced.json", "w"))
print("merged", len(runs), "seeds:", seeds, "from", [p.name for p in parts])
