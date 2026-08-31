#!/usr/bin/env python3
"""Shuffle method outputs into blinded A/B/C/D files so the judge cannot see
which clustering method produced which summary. Position bias is large in
LLM judging; the reveal key is written separately."""
import argparse, json, random, string
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--resultsdir", default="bench/results")
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()

d = Path(a.resultsdir)
methods = sorted(f.stem for f in d.glob("*.md"))
random.seed(a.seed)
order = methods[:]
random.shuffle(order)

blind = d / "blinded"
blind.mkdir(exist_ok=True)
for f in blind.glob("*.md"):
    f.unlink()

key = {}
for letter, method in zip(string.ascii_uppercase, order):
    text = (d / f"{method}.md").read_text(encoding="utf-8")
    # strip the method name from the heading
    body = "\n".join(l for l in text.splitlines() if not l.startswith("# "))
    n = sum(1 for l in body.splitlines() if l.startswith("## "))
    (blind / f"summary_{letter}.md").write_text(
        f"# Candidate {letter} ({n} themes)\n{body}", encoding="utf-8")
    key[letter] = method

(d / "reveal_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
print("blinded ->", ", ".join(f"{k}={v}" for k, v in key.items()))
print(f"files in {blind}/")
