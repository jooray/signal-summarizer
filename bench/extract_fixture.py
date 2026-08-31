#!/usr/bin/env python3
"""Extract themes once and freeze them as a benchmark fixture.

The merge/clustering stage can then be compared across methods on an identical
input, which is the only way the comparison measures merging rather than
run-to-run variation in extraction.
"""
import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from group_summarizer import (  # noqa: E402
    setup_logging,
    load_config,
    get_group_config,
    fetch_messages,
    build_conversation_chunks,
    generate_themes,
    apply_theme_actions,
)
from llm_util import LLMUtil, ConversationTheme, ConversationThemes, ThemeActions  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--group", required=True)
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--out", default="bench/fixtures/themes.json")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--request-timeout", type=int, default=120,
                   help="Per-request timeout (s). The config default of 600 means "
                        "a single hung request stalls the pipeline for 10 minutes.")
    args = p.parse_args()

    setup_logging(args.log_level)
    config = load_config(args.config)
    group_config = get_group_config(config, args.group)
    group_config["group_id"] = args.group

    import datetime

    until = datetime.datetime.now()
    since = until - datetime.timedelta(days=args.days)

    database = group_config.get("database", "messages.db")
    messages = fetch_messages(database, args.group, since, until)
    logging.info(f"Fetched {len(messages)} messages")

    models_cfg = copy.deepcopy(group_config.get("models", {}))
    for c in models_cfg.values():
        c["request_timeout"] = args.request_timeout
    llm_dict = {n: LLMUtil(c) for n, c in models_cfg.items()}
    themes_llm = llm_dict[group_config["themes"]["model"]]

    chunks = build_conversation_chunks(messages, group_config, llm_dict)
    logging.info(f"Built {len(chunks)} conversation chunks")

    sw = group_config.get("themes", {}).get("sliding_window", {})
    sw_enabled = sw.get("enabled", False)
    sw_size = sw.get("window_size", 5)

    t0 = time.time()
    all_themes: List[ConversationTheme] = []
    recent_themes: List[ConversationTheme] = []
    themesets = []

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def snapshot(done):
        sets = ([ConversationThemes(themes=all_themes)] if sw_enabled else themesets)
        out.write_text(json.dumps({
            "group_id": args.group, "config": args.config, "days": args.days,
            "since": since.isoformat(), "until": until.isoformat(),
            "n_messages": len(messages), "n_chunks": len(chunks),
            "chunks_done": done, "complete": done == len(chunks),
            "sliding_window": sw_enabled,
            "extraction_seconds": round(time.time() - t0, 1),
            "themesets": [ts.model_dump() for ts in sets],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    for i, chunk in enumerate(chunks):
        logging.info(f"Chunk {i + 1}/{len(chunks)}")
        if sw_enabled:
            result = generate_themes(
                chunk, group_config, themes_llm,
                recent_themes=recent_themes if recent_themes else None,
            )
            if result is None:
                continue
            if isinstance(result, ThemeActions):
                new = apply_theme_actions(result, recent_themes, all_themes)
                recent_themes.extend(new)
            elif isinstance(result, ConversationThemes):
                for t in result.themes:
                    all_themes.append(t)
                    recent_themes.append(t)
            if len(recent_themes) > sw_size:
                recent_themes = recent_themes[-sw_size:]
        else:
            res = generate_themes(chunk, group_config, themes_llm)
            if res:
                themesets.append(res)
        snapshot(i + 1)

    if sw_enabled:
        themesets = [ConversationThemes(themes=all_themes)]

    n = sum(len(ts.themes) for ts in themesets)
    logging.info(f"Extracted {n} themes in {time.time() - t0:.0f}s")

    snapshot(len(chunks))
    print(f"\nWrote {n} themes to {out}")


if __name__ == "__main__":
    main()
