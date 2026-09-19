#!/usr/bin/env python3
"""Benchmark: can Jev (Venice's typed-decision model, POST /decisions) do the
topic-matching work in this pipeline?

Three questions, each a stage, all on the frozen 61-theme Paraguay fixture and
the 1095 source messages it was extracted from:

  pairs         theme <-> theme: does Jev separate "same concrete topic" from
                "same domain, distinct discussion" better than cosine similarity
                and as well as the LLM pair verdict it would replace?
  assign        message -> topic: can Jev route every message to one of the
                extracted topics (or "none"), and does a chat model agree?
  e2e           merge topics with Jev, route messages with Jev, regenerate each
                topic's description from its messages with the extraction LLM,
                and judge the result blind against the incumbent nn-llm merge.

Nothing here touches the summarizer. Every API response is cached under
bench/results/jev-<date>/cache so re-runs are free and reproducible.

  uv run python bench/jev_bench.py pairs
  uv run python bench/jev_bench.py pairs-judge
  uv run python bench/jev_bench.py assign
  uv run python bench/jev_bench.py assign-judge
  uv run python bench/jev_bench.py e2e --threshold 3.0
  uv run python bench/jev_bench.py e2e-judge
"""
import argparse
import copy
import hashlib
import json
import logging
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import group_summarizer as gs  # noqa: E402
from llm_util import ConversationTheme, ConversationThemes, LLMUtil, PairVerdicts  # noqa: E402
from langchain.output_parsers import PydanticOutputParser  # noqa: E402

DATE = "2026-09-19"
OUT = ROOT / "bench" / "results" / f"jev-{DATE}"
CACHE = OUT / "cache"
FIXTURE = ROOT / "bench" / "fixtures" / "paraguay-180d.json"
EXTRACT_LOG = ROOT / "bench" / "fixtures" / "extract.log"
REVIEW_DIR = ROOT / "bench" / "results" / "quality-review-2026-09-09"
VENICE = "https://api.venice.ai/api/v1"
JEV = "jev-latest"
JEV_PRICE_IN = 0.042  # USD per 1M input tokens; output is free

# Production roles from config-venice.json (extraction: glm-flash, merge
# verdicts + merge prompt: gemma-4-31b). Judges are stronger models.
EXTRACT_MODEL = ("z-ai-glm-5-3-flash", {"temperature": 0.2, "effort": "low"})
VERDICT_MODELS = {
    "gemma4-31b": ("google-gemma-4-31b-it", {"disable_thinking": True}),
    "glm-flash": ("z-ai-glm-5-3-flash", {"temperature": 0.2, "effort": "low"}),
}
JUDGES = {
    "astra": ("openai-gpt-6-astra", {"effort": "low"}),
    "luna": ("openai-gpt-56-luna", {"effort": "low"}),
}
TIEBREAK = ("claude-opus-5", {"effort": "low"})

RUBRIC = [
    "Unrelated",
    "Same broad domain only; should stay separate",
    "Related but still distinct; should stay separate",
    "Same concrete topic, one a narrow subset or example of the other",
    "Essentially the same discussion or a direct continuation",
]
SAME_Q = ("Is this CANDIDATE THEME the same concrete discussion topic as the "
          "TARGET THEME, so that merging them into one summary section would "
          "lose no important specificity?")
RELATE_Q = "How does the TARGET THEME relate to this CANDIDATE THEME?"


def log(msg):
    print(msg, flush=True)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def venice_key():
    key = os.environ.get("VENICE_API_KEY")
    if key:
        return key
    for model in load(ROOT / "config-venice.json")["defaults"]["models"].values():
        if model.get("provider") == "venice":
            return model["apiKey"]
    raise SystemExit("No Venice key: set VENICE_API_KEY")


class Pacer:
    """Space request starts so the 100 req/min limit is never hit.

    Venice locks the key for 30 s after 50 non-success responses, so pacing
    client-side beats retrying 429s.
    """

    def __init__(self, min_interval):
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.next_start = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.min_interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)


JEV_PACER = Pacer(0.65)
CHAT_PACER = Pacer(0.25)
_MODELS_CACHE = {}


def model_price(model_id):
    if not _MODELS_CACHE:
        path = OUT / "models.json"
        if not path.exists():
            r = httpx.get(f"{VENICE}/models?type=all",
                          headers={"Authorization": f"Bearer {venice_key()}"}, timeout=60)
            r.raise_for_status()
            save(path, r.json())
        for m in load(path)["data"]:
            _MODELS_CACHE[m["id"]] = m.get("model_spec", {}).get("pricing", {})
    p = _MODELS_CACHE.get(model_id, {})
    return (p.get("input", {}).get("usd", 0) or 0, p.get("output", {}).get("usd", 0) or 0)


def _post_with_retry(url, payload, pacer, timeout, tag):
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                         .encode()).hexdigest()
    path = CACHE / tag / f"{key}.json"
    if path.exists():
        return load(path), True
    headers = {"Authorization": f"Bearer {venice_key()}", "Content-Type": "application/json"}
    last = None
    for attempt in range(8):
        pacer.wait()
        started = time.monotonic()
        try:
            r = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(min(30, 2 * (attempt + 1)))
            continue
        elapsed = time.monotonic() - started
        if r.status_code == 200:
            body = r.json()
            body["_meta"] = {"seconds": round(elapsed, 3), "cached_at": time.time()}
            save(path, body)
            return body, False
        last = f"HTTP {r.status_code}: {r.text[:300]}"
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(40, 3 * 1.8 ** attempt))
            continue
        raise RuntimeError(f"{tag} request failed: {last}")
    raise RuntimeError(f"{tag} request failed after retries: {last}")


def decide(state, questions):
    """One Jev request. Returns the raw response body (answers, usage, _meta)."""
    payload = {"model": JEV, "state": state, "questions": questions}
    body, _ = _post_with_retry(f"{VENICE}/decisions", payload, JEV_PACER, 120, "jev")
    return body


def chat(model_id, prompt, effort=None, temperature=None, disable_thinking=False,
         max_tokens=6000):
    payload = {"model": model_id,
               "messages": [{"role": "user", "content": prompt}],
               "max_completion_tokens": max_tokens,
               "venice_parameters": {"include_venice_system_prompt": False,
                                     "strip_thinking_response": True}}
    if disable_thinking:
        payload["venice_parameters"]["disable_thinking"] = True
    if effort:
        payload["reasoning_effort"] = effort
    if temperature is not None:
        payload["temperature"] = temperature
    body, _ = _post_with_retry(f"{VENICE}/chat/completions", payload, CHAT_PACER, 600, "chat")
    usage = body.get("usage", {})
    pin, pout = model_price(model_id)
    cost = (usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout) / 1e6
    return {"text": body["choices"][0]["message"].get("content", ""),
            "finish_reason": body["choices"][0].get("finish_reason"),
            "usage": usage, "cost_usd": cost, "seconds": body["_meta"]["seconds"]}


def chat_spec(spec, prompt, **kw):
    model_id, opts = spec
    return chat(model_id, prompt, **{**opts, **kw})


def add_usage(counter, usage):
    """Sum only the integer fields; Venice nests *_tokens_details dicts."""
    for k, v in (usage or {}).items():
        if isinstance(v, (int, float)):
            counter[k] += v


def parse_json_loose(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


# ---------------------------------------------------------------- fixtures --

def load_themes():
    fx = load(FIXTURE)
    themes = [t for ts in fx["themesets"] for t in ts["themes"]]
    for i, t in enumerate(themes):
        t["idx"] = i
        t["key"] = f"T{i + 1:02d}"
    return fx, themes


def theme_text(t):
    return f"Name: {t['name']}\nSummary: {t['summary']}"


def theme_short(t, limit=200):
    summary = re.sub(r"\s+", " ", t["summary"].strip())
    first = re.split(r"(?<=[.!?])\s+", summary)
    short = first[0]
    if len(first) > 1 and len(short) + len(first[1]) < limit:
        short = f"{short} {first[1]}"
    if len(short) > limit:
        short = short[:limit - 3].rstrip() + "..."
    return f"{t['name']}: {short}"


def load_messages(fx):
    import datetime
    since = datetime.datetime.fromisoformat(fx["since"])
    until = datetime.datetime.fromisoformat(fx["until"])
    db = load(ROOT / "config-venice.json")["defaults"].get("database", "./messages.db")
    msgs = gs.fetch_messages(str(ROOT / db), fx["group_id"], since, until)
    assert len(msgs) == fx["n_messages"], (len(msgs), fx["n_messages"])
    return msgs


def message_line(m, with_time=True):
    import datetime
    ts = datetime.datetime.fromtimestamp(m["timestamp"] / 1000).strftime("%Y-%m-%d %H:%M")
    body = (m.get("message") or "").strip()
    parts = [f"[{ts}] {m['sourceName']}: {body}" if with_time else f"{m['sourceName']}: {body}"]
    if m.get("quoteText"):
        parts.append(f'  (in reply to: "{m["quoteText"][:200]}")')
    for d in m.get("attachmentDescriptions") or []:
        parts.append(f"  [attachment: {d[:300]}]")
    return "\n".join(parts)


def chunk_map(fx, themes, messages):
    """theme idx -> set of chunk indices that created or updated it, and
    message position -> chunk index. Reconstructed from the extraction log and
    a deterministic re-split of the conversation text."""
    creates = 0
    lines = EXTRACT_LOG.read_text(encoding="utf-8").splitlines()
    events = []
    for line in lines:
        m = re.search(r"Chunk (\d+)/(\d+)", line)
        if m:
            events.append(("chunk", int(m.group(1))))
            continue
        m = re.search(r"Updated theme #(\d+): (.*?)\x1b", line + "\x1b")
        if m:
            events.append(("update", int(m.group(1)), m.group(2).strip()))
            continue
        m = re.search(r"New theme: (.*?)\x1b", line + "\x1b")
        if m:
            events.append(("new", m.group(1).strip()))
            creates += 1
    n_first = len(themes) - creates
    all_idx = list(range(n_first))
    recent = list(range(n_first))
    names = {i: None for i in range(n_first)}
    touched = defaultdict(set)
    for i in range(n_first):
        touched[i].add(1)
    chunk = 1
    for ev in events:
        if ev[0] == "chunk":
            if ev[1] != 1:
                recent = recent[-5:]
            chunk = ev[1]
        elif ev[0] == "update":
            k = ev[1] - 1
            if 0 <= k < len(recent):
                idx = recent[k]
                names[idx] = ev[2]
                touched[idx].add(chunk)
            else:
                idx = len(all_idx)
                all_idx.append(idx)
                recent.append(idx)
                names[idx] = ev[2]
                touched[idx].add(chunk)
        else:
            idx = len(all_idx)
            all_idx.append(idx)
            recent.append(idx)
            names[idx] = ev[1]
            touched[idx].add(chunk)
    mismatches = [(i, names[i], themes[i]["name"]) for i in range(len(themes))
                  if names[i] is not None and names[i] != themes[i]["name"]]
    assert len(all_idx) == len(themes), (len(all_idx), len(themes))
    assert not mismatches, mismatches[:5]

    # Re-split exactly like build_conversation_chunks (max_chunk_size from the
    # config the fixture was made with, overlap 100).
    cfg = load(ROOT / fx["config"])["defaults"]
    max_chunk = cfg["themes"].get("max_chunk_size", 5000)
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    texts, keep = [], []
    for pos, msg in enumerate(messages):
        body = (msg.get("message") or "").strip()
        parts = [f"[{__import__('datetime').datetime.fromtimestamp(msg['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M')}] {msg['sourceName']}:" + (f" {body}" if body else "")]
        if msg.get("quoteText"):
            parts.append(f'(In reply to: "{msg["quoteText"]}")')
        if msg.get("attachmentDescriptions"):
            parts.extend(msg["attachmentDescriptions"])
        if not body and len(parts) == 1:
            continue
        texts.append("\n".join(parts))
        keep.append(pos)
    combined = "\n".join(texts)
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk, chunk_overlap=100,
                                              length_function=len,
                                              separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_text(combined)
    assert len(chunks) == fx["n_chunks"], (len(chunks), fx["n_chunks"])
    # Locate each message's header in the combined text, then map offsets to
    # chunk starts (chunks are contiguous slices apart from the overlap).
    starts = []
    cursor = 0
    for c in chunks:
        at = combined.find(c[:80], cursor)
        assert at >= 0
        starts.append(at)
        cursor = at + 1
    msg_chunk = {}
    offset = 0
    for text, pos in zip(texts, keep):
        at = combined.find(text, offset)
        assert at >= 0
        offset = at + len(text)
        ci = max(i for i, s in enumerate(starts) if s <= at)
        msg_chunk[pos] = ci + 1
    return {i: sorted(touched[i]) for i in range(len(themes))}, msg_chunk


# --------------------------------------------------------------- baselines --

def review_embeddings():
    out = {}
    for name in ("mxbai-embed-large", "nomic-embed-text", "snowflake-arctic-embed"):
        p = REVIEW_DIR / f"embeddings-{name}.json"
        if p.exists():
            out[name] = np.asarray(load(p))
    return out


def cosine(emb):
    n = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return n @ n.T


def gold_pairs(themes):
    from bench.quality_review import RELATED_PAIRS, DISTINCT_PAIRS  # noqa
    by_name = {}
    for t in themes:
        by_name.setdefault(t["name"], t["idx"])
    pos = {tuple(sorted((by_name[a], by_name[b]))) for a, b in RELATED_PAIRS}
    neg = {tuple(sorted((by_name[a], by_name[b]))) for a, b in DISTINCT_PAIRS}
    return pos, neg


def auc(scores_pos, scores_neg):
    """Probability a random positive outranks a random negative (ties 0.5)."""
    if not scores_pos or not scores_neg:
        return float("nan")
    wins = 0.0
    for p in scores_pos:
        for n in scores_neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(scores_pos) * len(scores_neg))


def topk_union(sim, k):
    n = len(sim)
    pairs = set()
    for i in range(n):
        order = [int(j) for j in np.argsort(-sim[i], kind="stable") if j != i][:k]
        pairs.update(tuple(sorted((i, j))) for j in order)
    return pairs


# ------------------------------------------------------------- stage pairs --

def stage_pairs(args):
    fx, themes = load_themes()
    n = len(themes)
    log(f"{n} themes; batch shape: one request per target with {2 * (n - 1)} questions")

    S = np.full((n, n), np.nan)
    N = np.full((n, n), np.nan)
    P = {}  # directed (i, j) -> score probabilities
    usage = Counter()
    seconds = []

    def one(i):
        qs = {}
        for j, t in enumerate(themes):
            if j == i:
                continue
            cand = f"\nCANDIDATE THEME:\n{theme_text(t)}"
            qs[f"s{j}"] = {"type": "score", "instructions": RELATE_Q + cand, "criteria": RUBRIC}
            qs[f"n{j}"] = {"type": "noul", "instructions": SAME_Q + cand}
        body = decide(f"TARGET THEME:\n{theme_text(themes[i])}", qs)
        return i, body

    with ThreadPoolExecutor(max_workers=6) as pool:
        for fut in as_completed([pool.submit(one, i) for i in range(n)]):
            i, body = fut.result()
            usage["input_tokens"] += body["usage"]["input_tokens"]
            usage["requests"] += 1
            seconds.append(body["_meta"]["seconds"])
            for j in range(n):
                if j == i:
                    continue
                a = body["answers"]
                S[i, j] = a[f"s{j}"]["score"]
                N[i, j] = a[f"n{j}"]["noul"]
                P[f"{i},{j}"] = a[f"s{j}"]["probabilities"]
    log(f"batch: {usage['requests']} requests, {usage['input_tokens']} input tokens "
        f"(${usage['input_tokens'] * JEV_PRICE_IN / 1e6:.4f}), "
        f"mean {np.mean(seconds):.2f}s/request (cached calls report cached latency)")

    S_sym = (S + S.T) / 2
    N_sym = (N + N.T) / 2
    asym = np.abs(S - S.T)
    iu = np.triu_indices(n, 1)
    log(f"directional asymmetry |S_ij - S_ji|: mean {np.nanmean(asym[iu]):.3f}, "
        f"p90 {np.nanpercentile(asym[iu], 90):.3f}, max {np.nanmax(asym[iu]):.3f}")

    save(OUT / "jev-matrix.json", {"themes": [t["name"] for t in themes],
                                   "score_directed": np.round(S, 3).tolist(),
                                   "noul_directed": np.round(N, 3).tolist(),
                                   "rubric": RUBRIC, "probabilities": P,
                                   "usage": dict(usage)})

    # Evaluation set: gold 16 + union of top-3 candidates by mxbai (the
    # production retrieval) + union of top-3 by Jev symmetric score.
    embs = review_embeddings()
    cos = {name: cosine(e) for name, e in embs.items()}
    pos, neg = gold_pairs(themes)
    eval_pairs = set(pos) | set(neg)
    eval_pairs |= topk_union(cos["mxbai-embed-large"], 3)
    S_ret = np.where(np.isnan(S_sym), -1, S_sym)
    eval_pairs |= topk_union(S_ret, 3)
    eval_pairs = sorted(eval_pairs)
    log(f"evaluation set: {len(eval_pairs)} pairs (gold {len(pos) + len(neg)})")

    # Pair shape: both themes in the state, two questions.
    def one_pair(pair):
        i, j = pair
        state = f"THEME A:\n{theme_text(themes[i])}\n\nTHEME B:\n{theme_text(themes[j])}"
        qs = {"relate": {"type": "score", "criteria": RUBRIC,
                         "instructions": "How does THEME A relate to THEME B?"},
              "same": {"type": "noul",
                       "instructions": "Are THEME A and THEME B the same concrete discussion "
                                       "topic, so that merging them into one summary section "
                                       "would lose no important specificity?"}}
        return pair, decide(state, qs)

    pair_shape = {}
    pusage = Counter()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for fut in as_completed([pool.submit(one_pair, p) for p in eval_pairs]):
            pair, body = fut.result()
            pusage["input_tokens"] += body["usage"]["input_tokens"]
            pusage["requests"] += 1
            pair_shape[pair] = {"score": body["answers"]["relate"]["score"],
                                "noul": body["answers"]["same"]["noul"],
                                "probabilities": body["answers"]["relate"]["probabilities"]}
    log(f"pair shape: {pusage['requests']} requests, {pusage['input_tokens']} input tokens "
        f"(${pusage['input_tokens'] * JEV_PRICE_IN / 1e6:.4f})")

    rows = []
    for i, j in eval_pairs:
        row = {"i": i, "j": j, "a": themes[i]["name"], "b": themes[j]["name"],
               "gold": "related" if (i, j) in pos else "distinct" if (i, j) in neg else None,
               "jev_batch_score": round(float(S_sym[i, j]), 3),
               "jev_batch_ij": round(float(S[i, j]), 3), "jev_batch_ji": round(float(S[j, i]), 3),
               "jev_batch_noul": round(float(N_sym[i, j]), 3),
               "jev_pair_score": round(pair_shape[(i, j)]["score"], 3),
               "jev_pair_noul": round(pair_shape[(i, j)]["noul"], 3),
               "jev_pair_probs": pair_shape[(i, j)]["probabilities"]}
        for name, c in cos.items():
            row[f"cos_{name.split('-')[0]}"] = round(float(c[i, j]), 4)
        rows.append(row)
    save(OUT / "pairs.json", rows)

    # Gold-set separation.
    scorers = ["jev_batch_score", "jev_pair_score", "jev_batch_noul", "jev_pair_noul",
               "cos_mxbai", "cos_nomic", "cos_snowflake"]
    log("\nGold set (8 related merge candidates vs 8 hard negatives):")
    log(f"{'scorer':<18}{'AUC':>6}{'min pos':>9}{'max neg':>9}{'gap':>7}")
    summary = {}
    for s in scorers:
        p = [r[s] for r in rows if r["gold"] == "related"]
        q = [r[s] for r in rows if r["gold"] == "distinct"]
        a = auc(p, q)
        summary[s] = {"auc": round(a, 3), "min_pos": min(p), "max_neg": max(q)}
        log(f"{s:<18}{a:>6.3f}{min(p):>9.3f}{max(q):>9.3f}{min(p) - max(q):>7.3f}")
    log("\nPer gold pair (jev batch score / pair score / mxbai cosine):")
    for r in sorted(rows, key=lambda r: (r["gold"] or "z", -r["jev_batch_score"])):
        if r["gold"]:
            log(f"  {r['gold']:<8} {r['jev_batch_score']:.2f} {r['jev_pair_score']:.2f} "
                f"{r['cos_mxbai']:.3f}  {r['a'][:40]} <-> {r['b'][:40]}")
    save(OUT / "pairs-summary.json", {"gold_auc": summary, "eval_pairs": len(eval_pairs),
                                      "batch_usage": dict(usage), "pair_usage": dict(pusage),
                                      "asymmetry": {"mean": float(np.nanmean(asym[iu])),
                                                    "p90": float(np.nanpercentile(asym[iu], 90)),
                                                    "max": float(np.nanmax(asym[iu]))}})


# ------------------------------------------------------- stage pairs-judge --

def verdict_prompt(themes, batch):
    """Exactly the production nn_llm_merge prompt for a batch of pairs."""
    fmt = PydanticOutputParser(pydantic_object=PairVerdicts).get_format_instructions()
    numbered = {t["idx"]: f"{t['idx'] + 1}. {gs.capitalize_theme_name(t['name'])}\n"
                          f"Signature: {gs.build_theme_pair_signature(ConversationTheme(**{k: t[k] for k in ('name', 'summary', 'dissenting_opinions')}))}\n"
                          f"Summary: {t['summary']}" for t in themes}
    involved = sorted({i for pair in batch for i in pair})
    context = "\n\n".join(numbered[i] for i in involved)
    pairs_text = "\n".join(f"[{i + 1}, {j + 1}]" for i, j in batch)
    return gs.PAIR_VERDICT_PROMPT.format(format_instructions=fmt, context=context,
                                         pairs=pairs_text)


def run_verdicts(themes, pairs, spec, label, batch_size=10):
    """Returns {pair: label} plus usage/cost, via the production prompt."""
    out, usage, cost, fails = {}, Counter(), 0.0, 0

    def one(batch):
        res = chat_spec(spec, verdict_prompt(themes, batch))
        try:
            data = parse_json_loose(res["text"])
            verdicts = PairVerdicts(**data).verdicts
        except Exception as exc:  # noqa
            return batch, None, res, str(exc)
        got = {}
        for v in verdicts:
            if len(v.pair) == 2:
                got[tuple(sorted((v.pair[0] - 1, v.pair[1] - 1)))] = v.decision
        return batch, got, res, None

    batches = [pairs[k:k + batch_size] for k in range(0, len(pairs), batch_size)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fut in as_completed([pool.submit(one, b) for b in batches]):
            batch, got, res, err = fut.result()
            add_usage(usage, res["usage"])
            cost += res["cost_usd"]
            if err:
                fails += 1
                log(f"  {label}: batch parse failure: {err[:120]}")
                continue
            for pair in batch:
                out[pair] = got.get(pair, "missing")
    return out, {"usage": dict(usage), "cost_usd": round(cost, 4), "batch_failures": fails,
                 "batches": len(batches)}


def stage_pairs_judge(args):
    fx, themes = load_themes()
    rows = load(OUT / "pairs.json")
    pairs = [(r["i"], r["j"]) for r in rows]
    random.seed(11)
    random.shuffle(pairs)  # batch composition should not follow score order

    results, meta = {}, {}
    for label, spec in {**JUDGES, **VERDICT_MODELS}.items():
        log(f"verdicts: {label} ({spec[0]}) on {len(pairs)} pairs")
        results[label], meta[label] = run_verdicts(themes, pairs, spec, label)
        log(f"  {meta[label]}")

    # Silver label: the two judges agree, otherwise a third breaks the tie.
    silver = {}
    ties = [p for p in pairs if results["astra"].get(p) != results["luna"].get(p)
            or results["astra"].get(p) in (None, "missing")]
    log(f"judge disagreement on {len(ties)}/{len(pairs)} pairs; tie-break with {TIEBREAK[0]}")
    tb = {}
    if ties:
        tb, meta["tiebreak"] = run_verdicts(themes, ties, TIEBREAK, "tiebreak")
    for p in pairs:
        votes = [results[j].get(p) for j in ("astra", "luna")]
        if p in tb:
            votes.append(tb[p])
        votes = [v for v in votes if v in ("same", "related", "unrelated")]
        c = Counter(votes).most_common()
        if not c:
            silver[p] = None
        elif len(c) > 1 and c[0][1] == c[1][1]:
            silver[p] = "related"  # unresolved 3-way split: conservative middle
        else:
            silver[p] = c[0][0]

    for r in rows:
        p = (r["i"], r["j"])
        r["silver"] = silver[p]
        for label in results:
            r[f"verdict_{label}"] = results[label].get(p)
        r["verdict_tiebreak"] = tb.get(p)
    save(OUT / "pairs.json", rows)
    save(OUT / "pairs-judge-meta.json", meta)

    # Report
    dist = Counter(r["silver"] for r in rows)
    log(f"\nsilver label distribution: {dict(dist)}")
    gold_agree = [(r["gold"], r["silver"]) for r in rows if r["gold"]]
    ok = sum(1 for g, s in gold_agree
             if (g == "related" and s == "same") or (g == "distinct" and s != "same"))
    log(f"silver vs hand labels (related->same, distinct->not same): {ok}/{len(gold_agree)}")
    for g, s in gold_agree:
        pass

    log("\nScorer separation against silver labels (AUC):")
    log(f"{'scorer':<18}{'same vs rest':>14}{'same vs related':>17}{'unrel vs rest':>15}")
    scorers = ["jev_batch_score", "jev_pair_score", "jev_batch_noul", "jev_pair_noul",
               "cos_mxbai", "cos_nomic", "cos_snowflake"]
    labelled = [r for r in rows if r["silver"]]
    report = {"silver_distribution": dict(dist), "auc": {}, "incumbents": {}, "thresholds": {}}
    for s in scorers:
        same = [r[s] for r in labelled if r["silver"] == "same"]
        rel = [r[s] for r in labelled if r["silver"] == "related"]
        unr = [r[s] for r in labelled if r["silver"] == "unrelated"]
        a1 = auc(same, rel + unr)
        a2 = auc(same, rel)
        a3 = auc(rel + same, unr)
        report["auc"][s] = {"same_vs_rest": round(a1, 3), "same_vs_related": round(a2, 3),
                            "notunrelated_vs_unrelated": round(a3, 3)}
        log(f"{s:<18}{a1:>14.3f}{a2:>17.3f}{a3:>15.3f}")

    log("\nIncumbent LLM verdicts vs silver (accuracy on 3 labels; precision/recall of 'same'):")
    for label in list(VERDICT_MODELS) + list(JUDGES):
        v = results[label]
        agree = sum(1 for r in labelled if v.get((r["i"], r["j"])) == r["silver"])
        tp = sum(1 for r in labelled if v.get((r["i"], r["j"])) == "same" and r["silver"] == "same")
        fp = sum(1 for r in labelled if v.get((r["i"], r["j"])) == "same" and r["silver"] != "same")
        fn = sum(1 for r in labelled if v.get((r["i"], r["j"])) != "same" and r["silver"] == "same")
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        report["incumbents"][label] = {"accuracy": round(agree / len(labelled), 3),
                                       "same_precision": round(prec, 3), "same_recall": round(rec, 3),
                                       "tp": tp, "fp": fp, "fn": fn, "cost_usd": meta[label]["cost_usd"]}
        log(f"  {label:<12} acc {agree / len(labelled):.3f}  same P {prec:.2f} R {rec:.2f} "
            f"(tp {tp} fp {fp} fn {fn})  ${meta[label]['cost_usd']:.4f}")

    log("\nJev batch score thresholds for 'same' (precision / recall vs silver):")
    n_same = sum(1 for r in labelled if r["silver"] == "same")
    for thr in (2.5, 2.75, 3.0, 3.25, 3.5):
        for s in ("jev_batch_score", "jev_pair_score"):
            tp = sum(1 for r in labelled if r[s] >= thr and r["silver"] == "same")
            fp = sum(1 for r in labelled if r[s] >= thr and r["silver"] != "same")
            fp_unrel = sum(1 for r in labelled if r[s] >= thr and r["silver"] == "unrelated")
            prec = tp / (tp + fp) if tp + fp else float("nan")
            rec = tp / n_same if n_same else float("nan")
            report["thresholds"][f"{s}@{thr}"] = {"precision": round(prec, 3), "recall": round(rec, 3),
                                                 "tp": tp, "fp": fp, "fp_unrelated": fp_unrel}
            log(f"  {s:<16} >= {thr:<5} P {prec:.2f} R {rec:.2f} (tp {tp} fp {fp}, of which unrelated {fp_unrel})")
    save(OUT / "pairs-report.json", report)

    log("\nDisagreements between Jev (batch score >= 3.0) and silver:")
    for r in sorted(labelled, key=lambda r: -r["jev_batch_score"]):
        jev_same = r["jev_batch_score"] >= 3.0
        if jev_same != (r["silver"] == "same"):
            log(f"  jev {r['jev_batch_score']:.2f} silver {r['silver']:<9} "
                f"gemma {r.get('verdict_gemma4-31b')!s:<9} {r['a'][:38]} <-> {r['b'][:38]}")


# ------------------------------------------------------------ stage assign --

NONE_KEY = "none"


def topic_criteria(themes):
    crit = {t["key"]: theme_short(t) for t in themes}
    crit[NONE_KEY] = ("Off-topic chit-chat, a greeting, a joke, a bare link or image without "
                      "discussion, or a subject that matches none of the listed topics")
    return crit


def stage_assign(args):
    fx, themes = load_themes()
    messages = load_messages(fx)
    crit = topic_criteria(themes)
    W, CTX = args.window, 3
    log(f"{len(messages)} messages, window {W}, {len(crit)} options per question")

    # Window shape: one request per block of W messages, one choice per message.
    blocks = [list(range(s, min(s + W, len(messages)))) for s in range(0, len(messages), W)]

    def one_block(block):
        first = block[0]
        lines = []
        if first > 0:
            lines.append("CONTEXT (earlier messages, for reference only):")
            for p in range(max(0, first - CTX), first):
                lines.append(f"  {message_line(messages[p])}")
            lines.append("")
        lines.append("MESSAGES TO CLASSIFY:")
        for k, p in enumerate(block, 1):
            lines.append(f"M{k}: {message_line(messages[p])}")
        state = "\n".join(lines)
        qs = {}
        for k, p in enumerate(block, 1):
            qs[f"m{p}"] = {"type": "choice",
                           "instructions": f"Which topic does message M{k} belong to? Judge the "
                                           f"message itself, using the surrounding messages only to "
                                           f"resolve what it refers to.",
                           "criteria": crit}
        return block, decide(state, qs)

    window, usage, secs = {}, Counter(), []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for fut in as_completed([pool.submit(one_block, b) for b in blocks]):
            block, body = fut.result()
            usage["input_tokens"] += body["usage"]["input_tokens"]
            usage["requests"] += 1
            secs.append(body["_meta"]["seconds"])
            for p in block:
                a = body["answers"][f"m{p}"]
                top = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:3]
                window[p] = {"choice": a["choice"], "confidence": a["confidence"],
                             "top3": [(k, round(v, 3)) for k, v in top]}
    log(f"window shape: {usage['requests']} requests, {usage['input_tokens']} input tokens "
        f"(${usage['input_tokens'] * JEV_PRICE_IN / 1e6:.4f}), mean {np.mean(secs):.2f}s")

    # Single shape on a fixed sample, for shape agreement and for the judges.
    random.seed(7)
    sample = sorted(random.sample(range(len(messages)), args.sample))

    def one_single(p):
        lines = []
        if p > 0:
            lines.append("CONTEXT (earlier messages, for reference only):")
            for q in range(max(0, p - CTX), p):
                lines.append(f"  {message_line(messages[q])}")
            lines.append("")
        lines.append(f"MESSAGE TO CLASSIFY:\n{message_line(messages[p])}")
        qs = {"m": {"type": "choice", "criteria": crit,
                    "instructions": "Which topic does the MESSAGE TO CLASSIFY belong to? Judge "
                                    "the message itself, using the context only to resolve what "
                                    "it refers to."}}
        return p, decide("\n".join(lines), qs)

    single, susage = {}, Counter()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for fut in as_completed([pool.submit(one_single, p) for p in sample]):
            p, body = fut.result()
            susage["input_tokens"] += body["usage"]["input_tokens"]
            susage["requests"] += 1
            a = body["answers"]["m"]
            top = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:3]
            single[p] = {"choice": a["choice"], "confidence": a["confidence"],
                         "top3": [(k, round(v, 3)) for k, v in top]}
    log(f"single shape: {susage['requests']} requests, {susage['input_tokens']} input tokens "
        f"(${susage['input_tokens'] * JEV_PRICE_IN / 1e6:.4f})")

    # Chat baselines on the same sample: production extraction model and the
    # production verdict model, same options, JSON answer.
    topic_list = "\n".join(f"{t['idx'] + 1}. {theme_short(t)}" for t in themes)

    def chat_prompt(p):
        ctx = "\n".join(f"  {message_line(messages[q])}" for q in range(max(0, p - CTX), p))
        return (f"Assign one chat message to a topic from the list, or 0 if it is off-topic "
                f"chit-chat, a greeting, a joke, a bare link or image without discussion, or "
                f"fits none of the topics. Judge the message itself; use the context only to "
                f"resolve what it refers to. The chat is in Slovak/Czech/English.\n\n"
                f"TOPICS:\n{topic_list}\n\nCONTEXT (earlier messages):\n{ctx or '  (none)'}\n\n"
                f"MESSAGE:\n{message_line(messages[p])}\n\n"
                f'Answer with JSON only: {{"topic": <number or 0>, "reason": "<one short sentence>"}}')

    chat_results = {}
    for label, spec in VERDICT_MODELS.items():
        got, cusage, cost, bad = {}, Counter(), 0.0, 0

        def one_chat(p, spec=spec):
            return p, chat_spec(spec, chat_prompt(p), max_tokens=400)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for fut in as_completed([pool.submit(one_chat, p) for p in sample]):
                p, res = fut.result()
                add_usage(cusage, res["usage"])
                cost += res["cost_usd"]
                try:
                    num = int(parse_json_loose(res["text"])["topic"])
                    got[p] = NONE_KEY if num == 0 else themes[num - 1]["key"]
                except Exception:  # noqa
                    bad += 1
                    got[p] = None
        chat_results[label] = got
        log(f"chat {label}: {len(sample)} messages, ${cost:.4f}, parse failures {bad}")

    # Diagnostics.
    theme_chunks, msg_chunk = chunk_map(fx, themes, messages)
    by_key = {t["key"]: t for t in themes}
    dist = Counter(v["choice"] for v in window.values())
    none_rate = dist[NONE_KEY] / len(messages)
    conf = [v["confidence"] for v in window.values()]
    in_home = near_home = 0
    assigned = 0
    for p, v in window.items():
        if v["choice"] == NONE_KEY or v["choice"] not in by_key:
            continue
        assigned += 1
        home = theme_chunks[by_key[v["choice"]]["idx"]]
        c = msg_chunk.get(p)
        if c is None:
            continue
        if c in home:
            in_home += 1
        if any(abs(c - h) <= 1 for h in home):
            near_home += 1
    agree_shape = sum(1 for p in sample if single[p]["choice"] == window[p]["choice"])
    agree_chat = {label: sum(1 for p in sample if got[p] == window[p]["choice"])
                  for label, got in chat_results.items()}
    agree_chat_single = {label: sum(1 for p in sample if got[p] == single[p]["choice"])
                         for label, got in chat_results.items()}
    chat_none = {label: sum(1 for p in sample if got[p] == NONE_KEY) for label, got in chat_results.items()}
    log(f"\nwindow: none rate {none_rate:.1%}, topics used {sum(1 for k in dist if k != NONE_KEY)}/{len(themes)}, "
        f"confidence mean {np.mean(conf):.2f} p10 {np.percentile(conf, 10):.2f}")
    log(f"temporal plausibility of assigned messages: in home chunk {in_home}/{assigned} "
        f"({in_home / assigned:.1%}), within +-1 chunk {near_home}/{assigned} ({near_home / assigned:.1%})")
    log(f"sample {len(sample)}: window vs single agreement {agree_shape / len(sample):.1%}; "
        f"chat vs window: {', '.join(f'{k} {v / len(sample):.1%}' for k, v in agree_chat.items())}; "
        f"chat none rate: {', '.join(f'{k} {v / len(sample):.1%}' for k, v in chat_none.items())}")
    log("most used topics: " + ", ".join(f"{by_key[k]['name'][:30]} ({c})" for k, c in dist.most_common(8) if k != NONE_KEY))

    save(OUT / "assign.json", {
        "window": {str(p): v for p, v in window.items()},
        "single": {str(p): v for p, v in single.items()},
        "chat": {label: {str(p): v for p, v in got.items()} for label, got in chat_results.items()},
        "sample": sample, "msg_chunk": {str(p): c for p, c in msg_chunk.items()},
        "theme_chunks": {str(i): c for i, c in theme_chunks.items()},
        "diagnostics": {"none_rate": none_rate, "topics_used": sum(1 for k in dist if k != NONE_KEY),
                        "confidence_mean": float(np.mean(conf)),
                        "in_home_chunk": in_home, "near_home_chunk": near_home, "assigned": assigned,
                        "shape_agreement": agree_shape, "chat_window_agreement": agree_chat,
                        "chat_single_agreement": agree_chat_single, "chat_none": chat_none,
                        "window_usage": dict(usage), "single_usage": dict(susage)}})


# ------------------------------------------------------ stage assign-judge --

def stage_assign_judge(args):
    fx, themes = load_themes()
    messages = load_messages(fx)
    A = load(OUT / "assign.json")
    by_key = {t["key"]: t for t in themes}
    sample = A["sample"]
    methods = {"jev_window": {int(p): v["choice"] for p, v in A["window"].items()},
               "jev_single": {int(p): v["choice"] for p, v in A["single"].items()}}
    for label, got in A["chat"].items():
        methods[f"chat_{label}"] = {int(p): v for p, v in got.items()}

    def desc(key):
        if key == NONE_KEY:
            return "NONE: off-topic, chit-chat, or no listed topic fits"
        t = by_key[key]
        return f"{t['name']}: {t['summary']}"

    def one(p, judge_label):
        cands = []
        for m, got in methods.items():
            k = got.get(p)
            if k and k not in cands:
                cands.append(k)
        random.seed(p)
        random.shuffle(cands)
        letters = "ABCDEFG"
        cand_text = "\n".join(f"{letters[i]}. {desc(k)}" for i, k in enumerate(cands))
        ctx = "\n".join(f"  {message_line(messages[q])}" for q in range(max(0, p - 5), p))
        nxt = "\n".join(f"  {message_line(messages[q])}" for q in range(p + 1, min(len(messages), p + 3)))
        prompt = (
            "A chat message from a Slovak/Czech/English group has been assigned to a topic by "
            "several classifiers. Decide which candidate topic the message actually belongs to. "
            "Judge the message itself; use the surrounding messages only to resolve what it "
            "refers to. A message belongs to a topic when it contributes to that discussion, "
            "not merely when it shares a keyword. If the message is chit-chat, a bare reaction, "
            "or fits none of the candidates, prefer NONE if offered, otherwise say none fit.\n\n"
            f"EARLIER MESSAGES:\n{ctx or '  (none)'}\n\nMESSAGE:\n{message_line(messages[p])}\n\n"
            f"LATER MESSAGES:\n{nxt or '  (none)'}\n\nCANDIDATE TOPICS:\n{cand_text}\n\n"
            'Answer with JSON only: {"best": "<letter or null if none fit>", '
            '"acceptable": ["<letters that are defensible assignments>"], '
            '"reason": "<one sentence>"}')
        res = chat_spec(JUDGES[judge_label], prompt, max_tokens=500)
        try:
            data = parse_json_loose(res["text"])
            best = data.get("best")
            best_key = cands[letters.index(best)] if best in letters[:len(cands)] else None
            acc = [cands[letters.index(x)] for x in data.get("acceptable", []) if x in letters[:len(cands)]]
        except Exception as exc:  # noqa
            return p, judge_label, None, res, str(exc)
        return p, judge_label, {"best": best_key, "acceptable": acc, "candidates": cands,
                                "reason": data.get("reason")}, res, None

    judged = defaultdict(dict)
    cost = Counter()
    fails = Counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(one, p, j) for p in sample for j in JUDGES]
        for fut in as_completed(futs):
            p, j, data, res, err = fut.result()
            cost[j] += res["cost_usd"]
            if err:
                fails[j] += 1
                continue
            judged[j][p] = data
    log(f"judged {len(sample)} messages x {list(JUDGES)}; cost {dict(cost)}; parse failures {dict(fails)}")

    report = {}
    log(f"\n{'method':<18}" + "".join(f"{j + ' best':>12}{j + ' ok':>10}" for j in JUDGES) + f"{'both best':>11}")
    for m, got in methods.items():
        row = {}
        for j in JUDGES:
            ps = [p for p in sample if p in judged[j]]
            best = sum(1 for p in ps if got.get(p) == judged[j][p]["best"])
            ok = sum(1 for p in ps if got.get(p) in judged[j][p]["acceptable"] or got.get(p) == judged[j][p]["best"])
            row[j] = {"best": best / len(ps), "acceptable": ok / len(ps), "n": len(ps)}
        both = [p for p in sample if all(p in judged[j] for j in JUDGES)]
        row["both_best"] = sum(1 for p in both if all(got.get(p) == judged[j][p]["best"] for j in JUDGES)) / len(both)
        report[m] = row
        log(f"{m:<18}" + "".join(f"{row[j]['best']:>12.1%}{row[j]['acceptable']:>10.1%}" for j in JUDGES)
            + f"{row['both_best']:>11.1%}")
    both = [p for p in sample if all(p in judged[j] for j in JUDGES)]
    jagree = sum(1 for p in both if judged["astra"][p]["best"] == judged["luna"][p]["best"]) / len(both)
    none_best = {j: sum(1 for p in both if judged[j][p]["best"] in (None, NONE_KEY)) / len(both) for j in JUDGES}
    log(f"judge-judge agreement on best: {jagree:.1%}; judge says none/NONE: "
        + ", ".join(f"{j} {v:.1%}" for j, v in none_best.items()))
    report["_judges"] = {"agreement": jagree, "none_best": none_best, "cost": dict(cost)}
    save(OUT / "assign-judge.json", {"report": report,
                                     "judged": {j: {str(p): v for p, v in d.items()} for j, d in judged.items()}})

    log("\nSample of Jev window misses (judge best differs, both judges agree):")
    shown = 0
    for p in both:
        b = judged["astra"][p]["best"]
        if b == judged["luna"][p]["best"] and methods["jev_window"].get(p) != b and shown < 12:
            shown += 1
            got = methods["jev_window"].get(p)
            log(f"  msg {p}: jev={by_key[got]['name'][:34] if got in by_key else got!s:<36} "
                f"judge={by_key[b]['name'][:34] if b in by_key else b!s:<36} | "
                f"{(messages[p].get('message') or '')[:70]!r}")


# --------------------------------------------------------------- stage e2e --

def union_find_groups(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def linkage_groups(S_sym, thr, method):
    """Agglomerative groups over a similarity matrix on the rubric scale."""
    n = len(S_sym)
    if method == "single":
        edges = [(i, j) for i, j in combinations(range(n), 2) if S_sym[i, j] >= thr]
        return union_find_groups(n, edges)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    D = (len(RUBRIC) - 1) - S_sym
    np.fill_diagonal(D, 0.0)
    D = np.clip((D + D.T) / 2, 0, None)
    Z = linkage(squareform(D, checks=False), method=method)
    labels = fcluster(Z, t=(len(RUBRIC) - 1) - thr, criterion="distance")
    groups = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[lab].append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def render(themes, title):
    lines = [f"# {title}\n"]
    for t in themes:
        lines.append(f"## **{gs.capitalize_theme_name(t['name'])}**\n{t['summary']}")
        if t.get("dissenting_opinions"):
            lines.append(f"\n**Dissenting opinions:** {t['dissenting_opinions']}")
    return "\n\n".join(lines) + "\n"


def production_llm(role_model):
    cfg = load(ROOT / "config-venice.json")["defaults"]
    mc = copy.deepcopy(cfg["models"][role_model])
    mc["request_timeout"] = 300
    return LLMUtil(mc), cfg


def stage_e2e(args):
    fx, themes = load_themes()
    messages = load_messages(fx)
    M = load(OUT / "jev-matrix.json")
    S = np.asarray(M["score_directed"], dtype=float)
    S_sym = (S + S.T) / 2
    n = len(themes)
    thr = args.threshold
    veto = args.veto

    # Jev merge decision. A pair whose weaker direction sits in "unrelated"
    # territory is vetoed (score pinned to 0), mirroring the production
    # "unrelated" veto. Single linkage = union-find like nn_llm_merge and
    # chains A~B, B~C into one group; complete linkage requires every pair in
    # a group to clear the threshold; average linkage sits between.
    S_eff = S_sym.copy()
    for i, j in combinations(range(n), 2):
        if min(S[i, j], S[j, i]) < veto:
            S_eff[i, j] = S_eff[j, i] = 0.0
    groups = linkage_groups(S_eff, thr, args.linkage)
    multi = [g for g in groups if len(g) > 1]
    log(f"jev merge @>={thr} ({args.linkage} linkage, veto <{veto}): {len(groups)} topics "
        f"({len(multi)} merged groups, largest {max(map(len, groups))})")
    for g in multi:
        log("  + " + " | ".join(themes[i]["name"][:34] for i in g))

    merge_llm, cfg = production_llm(cfg_role("themes_recombination"))
    merging_prompt = cfg["themes_recombination"]["merging_prompt"]

    # Variant A: Jev replaces only the pair-verdict step; the merged summary is
    # still written from the theme summaries with the production merge prompt.
    def merge_group(g):
        if len(g) == 1:
            return dict(themes[g[0]])
        merged = gs.merge_themes_with_prompt(
            [ConversationTheme(**{k: themes[i][k] for k in ("name", "summary", "dissenting_opinions")}) for i in g],
            merge_llm, merging_prompt)
        return merged.model_dump()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        variant_a = list(pool.map(merge_group, groups))
    log(f"variant jev-verdict: {len(variant_a)} topics in {time.time() - t0:.0f}s")
    (OUT / "jev-verdict.md").write_text(render(variant_a, f"jev-verdict — {len(variant_a)} themes"), encoding="utf-8")

    # Variant B: same groups, but every topic's text is regenerated from the
    # messages Jev routed to it (window shape).
    A = load(OUT / "assign.json")
    by_key = {t["key"]: t["idx"] for t in themes}
    theme_group = {}
    for gi, g in enumerate(groups):
        for i in g:
            theme_group[i] = gi
    group_msgs = defaultdict(list)
    for p, v in A["window"].items():
        k = v["choice"]
        if k in by_key:
            group_msgs[theme_group[by_key[k]]].append(int(p))
    routed = sum(len(v) for v in group_msgs.values())
    log(f"routed {routed}/{len(messages)} messages into {len(group_msgs)} topics; "
        f"{sum(1 for g in range(len(groups)) if len(group_msgs[g]) < args.min_messages)} topics below "
        f"{args.min_messages} messages keep their original text")

    fmt = PydanticOutputParser(pydantic_object=ConversationTheme).get_format_instructions()

    def regenerate(gi):
        ps = sorted(group_msgs[gi])
        if len(ps) < args.min_messages:
            t = dict(variant_a[gi])
            t["_source"] = f"kept ({len(ps)} msgs)"
            return t
        names = "; ".join(themes[i]["name"] for i in groups[gi])
        convo = "\n".join(message_line(messages[p]) for p in ps)
        if len(convo) > 60000:
            convo = convo[:60000] + "\n[... truncated ...]"
        prompt = (
            "Below are the messages from a group chat that were classified as belonging to one "
            f"discussion topic (working title: {names}). Write the theme for the summary: a short, "
            "specific, concrete name; a 2-4 sentence summary of what was actually said, keeping "
            "concrete facts (prices, documents, decisions, recommendations, corrections); and "
            "dissenting_opinions only when there is a real disagreement (otherwise an empty "
            "string). Do not invent anything that is not in the messages. Ignore messages that "
            "were misclassified and clearly belong elsewhere. Output in English even though the "
            f"messages are Slovak/Czech/English.\n\n{fmt}\n\nMessages:\n{convo}")
        res = chat_spec(EXTRACT_MODEL, prompt, max_tokens=1500)
        try:
            t = ConversationTheme(**parse_json_loose(res["text"])).model_dump()
        except Exception as exc:  # noqa
            log(f"  regenerate parse failure for group {gi}: {exc}")
            t = dict(variant_a[gi])
        t["_source"] = f"regenerated ({len(ps)} msgs, ${res['cost_usd']:.4f})"
        t["_cost"] = res["cost_usd"]
        return t

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        variant_b = list(pool.map(regenerate, range(len(groups))))
    cost_b = sum(t.get("_cost", 0) for t in variant_b)
    log(f"variant jev-e2e: {len(variant_b)} topics, regenerated "
        f"{sum(1 for t in variant_b if t['_source'].startswith('regen'))}, ${cost_b:.4f}, {time.time() - t0:.0f}s")
    (OUT / "jev-e2e.md").write_text(render(variant_b, f"jev-e2e — {len(variant_b)} themes"), encoding="utf-8")

    # Incumbent: production nn-llm merge on the same fixture with config-venice.
    inc = OUT / "nn-llm.md"
    if not inc.exists() or args.rerun_incumbent:
        group_config = gs.get_group_config(load(ROOT / "config-venice.json"), fx["group_id"])
        sets = [ConversationThemes(**ts) for ts in fx["themesets"]]
        t0 = time.time()
        merged = gs.recombine_themes(sets, group_config, merge_llm)
        inc_themes = [t.model_dump() for t in merged.themes]
        log(f"incumbent nn-llm: {len(inc_themes)} topics in {time.time() - t0:.0f}s")
        inc.write_text(render(inc_themes, f"nn-llm — {len(inc_themes)} themes"), encoding="utf-8")
    save(OUT / "e2e.json", {"threshold": thr, "veto": veto, "groups": groups,
                            "group_messages": {str(g): sorted(v) for g, v in group_msgs.items()},
                            "variant_a": variant_a, "variant_b": variant_b,
                            "cost_regenerate_usd": cost_b})


def cfg_role(role):
    return load(ROOT / "config-venice.json")["defaults"][role]["model"]


# --------------------------------------------------------- stage e2e-judge --

E2E_JUDGE_PROMPT = """You compare two candidate summaries of the same group chat. The source
conversation is given in full. The summaries' methods are hidden. Judge only against the source.

Criteria, each decided separately:
- faithfulness: fewer claims that the source does not support, fewer attribution errors,
  no strengthened certainty. Quote up to 5 problematic passages per candidate with the line
  numbers that contradict or fail to support them.
- specificity: concrete facts a member could act on (prices, documents, waiting times,
  decisions, corrections, named services) rather than vague description.
- structure: topics are distinct, not duplicated or over-merged into umbrellas; each
  section is about one concrete discussion.
- coverage: the substantive discussions in the source are represented.

Do not reward length. Return JSON only:
{{"faithfulness": {{"winner": "A|B|tie", "issues_A": [{{"quote": "...", "lines": [1], "why": "..."}}],
  "issues_B": [...]}}, "specificity": {{"winner": "A|B|tie", "why": "..."}},
 "structure": {{"winner": "A|B|tie", "duplicated_or_overmerged_A": ["..."], "duplicated_or_overmerged_B": ["..."]}},
 "coverage": {{"winner": "A|B|tie", "missing_from_A": ["..."], "missing_from_B": ["..."]}},
 "overall": "A|B|tie", "why": "..."}}

SOURCE (line-numbered):
{source}

CANDIDATE A:
{a}

CANDIDATE B:
{b}
"""


def stage_e2e_judge(args):
    fx, themes = load_themes()
    messages = load_messages(fx)
    source = "\n".join(f"L{p + 1:04d} {message_line(messages[p])}" for p in range(len(messages)))
    cands = {name: (OUT / f"{name}.md").read_text(encoding="utf-8")
             for name in ("nn-llm", "jev-verdict", "jev-e2e")}
    strip = lambda md: "\n".join(l for l in md.splitlines() if not l.startswith("# "))  # noqa
    matchups = list(combinations(cands, 2))
    jobs = [(x, y, order, j) for x, y in matchups for order in (0, 1) for j in JUDGES]

    def one(job):
        x, y, order, j = job
        a, b = (x, y) if order == 0 else (y, x)
        prompt = E2E_JUDGE_PROMPT.format(source=source, a=strip(cands[a]), b=strip(cands[b]))
        res = chat_spec(JUDGES[j], prompt, max_tokens=4000)
        try:
            data = parse_json_loose(res["text"])
        except Exception as exc:  # noqa
            return job, a, b, None, res, str(exc)
        return job, a, b, data, res, None

    results = []
    cost = 0.0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for fut in as_completed([pool.submit(one, j) for j in jobs]):
            job, a, b, data, res, err = fut.result()
            cost += res["cost_usd"]
            if err:
                log(f"  judge parse failure {job}: {err[:100]}")
                continue
            results.append({"judge": job[3], "A": a, "B": b, "verdict": data,
                            "usage": res["usage"], "cost_usd": res["cost_usd"]})
    log(f"{len(results)}/{len(jobs)} judgments, ${cost:.3f}")
    save(OUT / "e2e-judge.json", results)

    crit = ("faithfulness", "specificity", "structure", "coverage", "overall")
    wins = {c: Counter() for c in crit}
    issues = Counter()
    for r in results:
        for c in crit:
            w = r["verdict"].get(c) if c == "overall" else r["verdict"].get(c, {}).get("winner")
            if w == "A":
                wins[c][r["A"]] += 1
            elif w == "B":
                wins[c][r["B"]] += 1
            else:
                wins[c]["tie"] += 1
        f = r["verdict"].get("faithfulness", {})
        issues[r["A"]] += len(f.get("issues_A", []))
        issues[r["B"]] += len(f.get("issues_B", []))
    log(f"\n{'criterion':<14}" + "".join(f"{c:>13}" for c in cands) + f"{'tie':>6}")
    for c in crit:
        log(f"{c:<14}" + "".join(f"{wins[c][name]:>13}" for name in cands) + f"{wins[c]['tie']:>6}")
    log("faithfulness issues flagged per candidate (summed over judgments where it appeared): "
        + ", ".join(f"{k} {v}" for k, v in issues.items()))
    for r in results:
        log(f"\n[{r['judge']}] A={r['A']} B={r['B']} overall={r['verdict'].get('overall')}: "
            f"{str(r['verdict'].get('why', ''))[:300]}")


# ------------------------------------------------------------------- main --

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["pairs", "pairs-judge", "assign", "assign-judge", "e2e", "e2e-judge"])
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--sample", type=int, default=150)
    p.add_argument("--threshold", type=float, default=3.0)
    p.add_argument("--veto", type=float, default=1.5)
    p.add_argument("--linkage", choices=["single", "complete", "average"], default="complete")
    p.add_argument("--min-messages", type=int, default=3)
    p.add_argument("--rerun-incumbent", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    {"pairs": stage_pairs, "pairs-judge": stage_pairs_judge, "assign": stage_assign,
     "assign-judge": stage_assign_judge, "e2e": stage_e2e, "e2e-judge": stage_e2e_judge}[args.stage](args)


if __name__ == "__main__":
    main()
