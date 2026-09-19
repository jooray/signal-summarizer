#!/usr/bin/env python3
"""Compare theme-merging methods on a frozen theme fixture.

Every method sees byte-identical input themes, so differences in the output are
attributable to the clustering/merging stage alone.
"""
import argparse
import copy
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import group_summarizer as gs  # noqa: E402
from group_summarizer import setup_logging, load_config, get_group_config  # noqa: E402
from llm_util import LLMUtil, ConversationThemes  # noqa: E402
from decision_util import DecisionClient, DECISION_PROVIDERS  # noqa: E402

# Parameters mirror run-summarization-test.sh so the comparison matches the
# knobs that script sweeps.
METHODS = {
    "none": None,  # control: no merging at all
    "dbscan": {"method": "dbscan", "eps": 0.3, "min_samples": 2},
    "hdbscan": {"method": "hdbscan", "min_cluster_size": 2, "hdbscan_min_samples": 1},
    "louvain": {"method": "louvain", "similarity_threshold": 0.5, "resolution": 1.0},
    "nn-llm": {"method": "nn-llm"},
    "decision": {"engine": "decision"},  # Jev pair scoring; needs a venice-decision model
}


class CallCounter:
    """Wrap LLMUtil.generate_structured_output to count calls per Pydantic class."""

    def __init__(self):
        self.calls = {}
        self._orig = LLMUtil.generate_structured_output

    def __enter__(self):
        counter = self

        def wrapped(slf, *a, **kw):
            cls = kw.get("pydantic_class")
            if cls is None and len(a) >= 3:
                cls = a[2]
            name = getattr(cls, "__name__", "unknown")
            counter.calls[name] = counter.calls.get(name, 0) + 1
            return counter._orig(slf, *a, **kw)

        LLMUtil.generate_structured_output = wrapped
        return self

    def __exit__(self, *exc):
        LLMUtil.generate_structured_output = self._orig

    @property
    def total(self):
        return sum(self.calls.values())


def render(themes, title):
    lines = [f"# {title}\n"]
    for t in themes.themes:
        lines.append(f"## **{gs.capitalize_theme_name(t.name)}**\n{t.summary}")
        if t.dissenting_opinions:
            lines.append(f"\n**Dissenting opinions:** {t.dissenting_opinions}")
    return "\n\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", required=True)
    p.add_argument("--config", default="config.json")
    p.add_argument("--outdir", default="bench/results")
    p.add_argument("--methods", nargs="*", default=list(METHODS))
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    setup_logging(args.log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    fx = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    themesets = [ConversationThemes(**ts) for ts in fx["themesets"]]
    n_in = sum(len(ts.themes) for ts in themesets)

    config = load_config(args.config)
    base_group_config = get_group_config(config, fx["group_id"])
    llm_dict = {n: (DecisionClient(c) if c.get("provider") in DECISION_PROVIDERS else LLMUtil(c))
                for n, c in base_group_config.get("models", {}).items()}
    decision_model = base_group_config["themes_recombination"].get("decision", {}).get("model")
    decision_client = llm_dict.get(decision_model)
    recomb_llm = llm_dict[base_group_config["themes_recombination"]["model"]]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    for method in args.methods:
        overrides = METHODS[method]
        if overrides and overrides.get("engine") == "decision" and decision_client is None:
            print(f"\nskipping {method}: no venice-decision model in themes_recombination.decision.model")
            continue
        print(f"\n{'=' * 60}\n  {method}\n{'=' * 60}")

        random.seed(args.seed)
        gc = copy.deepcopy(base_group_config)
        # Fresh copies: merging mutates theme objects in place.
        sets = [ConversationThemes(**ts) for ts in fx["themesets"]]

        if overrides is None:
            merged = ConversationThemes(
                themes=[t for ts in sets for t in ts.themes]
            )
            elapsed, calls, detail = 0.0, 0, {}
        else:
            if overrides.get("engine") == "decision":
                gc["themes_recombination"]["engine"] = "decision"
            else:
                gc["themes_recombination"]["engine"] = "llm"
                ec = gc.setdefault("embedding_clustering", {})
                ec["enabled"] = True
                ec.update(overrides)
            t0 = time.time()
            with CallCounter() as cc:
                merged = gs.recombine_themes(sets, gc, recomb_llm, decision=decision_client)
            elapsed, calls, detail = time.time() - t0, cc.total, dict(cc.calls)

        n_out = len(merged.themes) if merged else 0
        (outdir / f"{method}.md").write_text(
            render(merged, f"{method} — {n_out} themes"), encoding="utf-8"
        )
        rows.append({
            "method": method,
            "themes_in": n_in,
            "themes_out": n_out,
            "reduction_pct": round(100 * (1 - n_out / n_in), 1) if n_in else 0,
            "llm_calls": calls,
            "seconds": round(elapsed, 1),
            "calls_detail": detail,
        })
        print(f"  {n_in} -> {n_out} themes  ({calls} LLM calls, {elapsed:.0f}s)")

    (outdir / "metrics.json").write_text(
        json.dumps({"fixture": fx["group_id"], "n_messages": fx["n_messages"],
                    "n_chunks": fx["n_chunks"], "results": rows},
                   indent=2), encoding="utf-8")

    print(f"\n{'method':<10} {'in':>4} {'out':>4} {'reduce':>7} {'calls':>6} {'secs':>6}")
    print("-" * 44)
    for r in rows:
        print(f"{r['method']:<10} {r['themes_in']:>4} {r['themes_out']:>4} "
              f"{r['reduction_pct']:>6}% {r['llm_calls']:>6} {r['seconds']:>6}")
    print(f"\nOutputs in {outdir}/")


if __name__ == "__main__":
    main()
