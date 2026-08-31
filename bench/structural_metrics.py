#!/usr/bin/env python3
"""Objective structural metrics per method output, as a cross-check on the LLM judge.

near_dupes: section pairs whose embeddings exceed a cosine threshold -> under-merging.
max_section_chars / bloat: unusually long sections -> a grab-bag from over-merging.
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from cluster import _generate_embeddings
from sklearn.metrics.pairwise import cosine_similarity

def sections(md):
    out, name, buf = [], None, []
    for line in md.splitlines():
        m = re.match(r"^## \*\*(.+?)\*\*\s*$", line)
        if m:
            if name: out.append((name, "\n".join(buf).strip()))
            name, buf = m.group(1), []
        elif name is not None:
            buf.append(line)
    if name: out.append((name, "\n".join(buf).strip()))
    return out

rows = []
for f in sorted(Path("bench/results").glob("*.md")):
    secs = sections(f.read_text(encoding="utf-8"))
    if not secs: continue
    texts = [f"Theme: {n}\nSummary: {b[:240]}" for n, b in secs]
    emb = _generate_embeddings(texts, "mxbai-embed-large")
    S = cosine_similarity(emb)
    np.fill_diagonal(S, 0)
    pairs = {}
    for thr in (0.75, 0.80, 0.85):
        idx = np.argwhere(np.triu(S) >= thr)
        pairs[thr] = [(secs[i][0], secs[j][0], round(float(S[i, j]), 3)) for i, j in idx]
    lens = [len(b) for _, b in secs]
    rows.append({
        "method": f.stem, "sections": len(secs),
        "near_dupes_0.75": len(pairs[0.75]), "near_dupes_0.80": len(pairs[0.80]),
        "near_dupes_0.85": len(pairs[0.85]),
        "mean_chars": int(np.mean(lens)), "max_chars": max(lens),
        "p90_chars": int(np.percentile(lens, 90)),
        "worst_pairs": sorted(pairs[0.80], key=lambda x: -x[2])[:4],
    })

Path("bench/results/structural.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"{'method':<9}{'secs':>5}{'dup.75':>8}{'dup.80':>8}{'dup.85':>8}{'mean':>7}{'p90':>6}{'max':>6}")
print("-" * 57)
for r in sorted(rows, key=lambda r: r["method"]):
    print(f"{r['method']:<9}{r['sections']:>5}{r['near_dupes_0.75']:>8}{r['near_dupes_0.80']:>8}"
          f"{r['near_dupes_0.85']:>8}{r['mean_chars']:>7}{r['p90_chars']:>6}{r['max_chars']:>6}")
print("\nTop near-duplicate pairs still present at cosine >= 0.80:")
for r in sorted(rows, key=lambda r: r["method"]):
    print(f"\n  [{r['method']}]")
    for a, b, s in r["worst_pairs"] or []:
        print(f"    {s}  {a[:44]!r} <-> {b[:44]!r}")
    if not r["worst_pairs"]: print("    (none)")
