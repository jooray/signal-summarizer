# Jev as a topic-matching engine — benchmark, 19 September 2026

Question: can Venice's typed-decision model Jev (`jev-latest`, `POST /decisions`;
see `~/projects/server-documentation/experiments/jev.md`) take over the
matching work in this pipeline — deciding which extracted themes are the same
topic, and routing messages to topics — with the chat LLM kept only for writing
text? Nothing in the summarizer was changed. Harness: `bench/jev_bench.py`;
raw outputs, every API response and the three candidate summaries are in
`bench/results/jev-2026-09-19/` (gitignored).

**Verdict.** Yes for both matching jobs, as an optional engine.

- **Theme-pair matching:** Jev separates the 8 hand-labelled merge candidates
  from the 8 hard negatives perfectly (AUC 1.00); the best embedding model
  manages 0.94. It never ranks a judge-unrelated pair above the merge range.
  Scoring all 1,830 pairs of the 61-theme fixture took 61 requests, 80 s and
  4 cents.
- **Message routing:** with 8 messages per request and 62 options, Jev picks
  the topic two strong judges also pick 84% of the time, ahead of the
  production chat models (79–82%), at 15 cents and ~3.5 minutes for 1,095
  messages, rate-limit bound.
- **End to end:** using Jev only for the merge decision (summaries still merged
  by the production prompt) beat the incumbent nn-llm output in 4 of 4 blind
  matchups. Regenerating each topic from the messages Jev routed to it wins on
  specificity and coverage 8–0 but adds attribution errors; the two judges
  split on whether that trade is worth it. It needs the routing tightened
  before it is a candidate.

The absolute scale of Jev's scores is compressed and must be calibrated on our
own labels (the doc warned about this): "same topic" lives at 2.4–3.2 on a 0–4
rubric, not at 3.5+. Use complete linkage over the score matrix, not
union-find: single linkage chains residency into crypto tax at every threshold
below 3.0.

## 1. Setup

- Fixture: `bench/fixtures/paraguay-180d.json`, 61 themes from 32 chunks of
  1,095 messages (Slovak/Czech/English), the same input the 9 September review
  used. Gold labels: its 8 `RELATED_PAIRS` (merge candidates) and 8
  `DISTINCT_PAIRS` (same domain, must stay apart) in `bench/quality_review.py`.
- Production roles from `config-venice.json`: extraction `z-ai-glm-5-3-flash`
  (low effort, T=0.2), merge verdicts and merge prompt `google-gemma-4-31b-it`.
  Retrieval baselines: the cached mxbai / nomic / snowflake embeddings from the
  review.
- Judges: `openai-gpt-6-astra` and `openai-gpt-56-luna` at low effort;
  `claude-opus-5` breaks ties on pair labels. Astra is $10/M input and was
  $8.36 of the $9.06 total spend; Jev was $0.22 for 559 requests (5.2M input
  tokens), latency p50 0.81 s, p90 1.50 s regardless of question count.
- Rate limit is the real throughput ceiling: 100 requests/min per key, shared
  with every other Jev user of the key. The harness paces at 0.65 s between
  request starts and never saw a 429.

## 2. Theme ↔ theme matching

### 2.1 Request shapes

Two shapes, both with the same 5-level rubric as `criteria` (unrelated / same
broad domain / related but distinct / same concrete topic, one a subset / same
discussion) and a yes/no `noul` "same concrete topic, merging loses nothing":

- **batch**: state = one target theme; 120 questions per request, a `score` and
  a `noul` for each of the 60 candidates with the candidate text inside the
  question. 61 requests cover the full directed matrix; scores are averaged
  over the two directions. 16k input tokens and 1.3 s per request.
- **pair**: state = both themes, two questions. One request per pair; used on
  the 211-pair evaluation set.

The two shapes agree closely (Spearman 0.96, mean absolute difference 0.15 on
the 0–4 scale). Directional asymmetry of the batch shape is small (mean 0.13,
p90 0.32), so averaging the two directions is safe. Use the batch shape: it is
30x cheaper per pair and covers every pair, which removes the retrieval step
(top-k by embedding) entirely.

The `noul` probability is nearly useless as an absolute value — it sits at
0.06–0.17 for *every* pair, gold positives included — but its ranking is as
good as the score's. Keep the `score` and store the probability distribution.

### 2.2 Gold set (8 merge candidates vs 8 hard negatives)

| scorer | AUC | lowest positive | highest negative | gap |
|---|---:|---:|---:|---:|
| **Jev batch score** | **1.000** | 2.42 | 2.08 | +0.35 |
| **Jev pair score** | **1.000** | 2.40 | 1.78 | +0.62 |
| Jev noul (either shape) | 1.000 | 0.12 | 0.07 | +0.05 |
| cosine mxbai (production) | 0.906 | 0.687 | 0.799 | −0.11 |
| cosine nomic | 0.938 | 0.798 | 0.846 | −0.05 |
| cosine snowflake | 0.938 | 0.627 | 0.678 | −0.05 |

Every embedding model puts at least one hard negative above at least one merge
candidate (Asunción travel ↔ Encarnación travel scores 0.80 cosine on mxbai,
higher than five of the eight true merges). Jev puts that pair at 2.08 and the
banking-forex ↔ banking-security pair at 1.33. This is the failure mode the
review identified — embeddings cannot tell "same domain" from "same
discussion" — and Jev does not have it.

Calibrated threshold from the gold set: **2.4** catches all 8 merge candidates
with a 0.35 margin over the worst negative. The 16-pair gold set is small; treat
2.4 as the starting point and re-check after any change in Jev's answers.

### 2.3 Silver labels on 211 candidate pairs

Evaluation set: gold 16 ∪ top-3 mxbai neighbours (the production retrieval)
∪ top-3 Jev neighbours. Astra and Luna each labelled every pair with the
production `PAIR_VERDICT_PROMPT` (same / related / unrelated), Opus broke the
26 disagreements.

The result says more about the prompt than about Jev: with its "when in doubt
prefer related" rule the strong judges call **2 of 211** pairs "same" and 168
"related" — including 7 of the 8 hand-labelled merge candidates. So "merge or
not" at this fixture's granularity is a policy choice, and the silver set is
too thin on "same" to measure merge precision. What it can measure:

| scorer | AUC same vs rest (n=2) | AUC not-unrelated vs unrelated (170 vs 41) | unrelated pairs above the median related pair |
|---|---:|---:|---:|
| Jev batch score | 0.983 | **0.888** | **2 / 41** |
| Jev pair score | 0.981 | 0.895 | — |
| cosine mxbai | 0.978 | 0.765 | 8 / 41 |
| cosine nomic | 0.990 | 0.822 | 3 / 41 |
| cosine snowflake | 0.976 | 0.711 | — |

Jev is clearly better at the retrieval boundary (unrelated vs everything
else); on the 2-pair "same" set nothing can be concluded.

The incumbent verdict models against the silver labels:

| model | 3-label accuracy | "same" verdicts | of which silver same / related / unrelated |
|---|---:|---:|---|
| gemma-4-31b (production) | 0.763 | 31 | 2 / 29 / 0 |
| glm-flash | 0.825 | 10 | 2 / 8 / 0 |
| astra | 0.967 | 1 | 1 / 0 / 0 |
| luna | 0.910 | 1 | 1 / 0 / 0 |

Gemma merges far more than the strong judges would, and its 31 "same" pairs
span Jev scores 1.88–3.21, i.e. it says "same" for pairs Jev puts at "same
broad domain". Jev at ≥2.4 accepts 19 of the 211 pairs, all silver same or
related, never unrelated; its top 9 are all pairs gemma also calls "same", and
6 of those 9 glm-flash calls "same" as well. In other words Jev gives a stable ordering; the threshold sets
how aggressive the merge is, and 2.4 is more conservative than the production
model while still catching every hand-labelled merge.

### 2.4 Groups at the calibrated threshold

Union-find (what `nn_llm_merge` does with "same" edges) chains transitively:
at 2.4 it builds a 9-theme group joining Paraguay residency, crypto tax
reporting and DNIT; at 2.2 a 15-theme umbrella. Complete linkage (every pair
in a group ≥ threshold) at 2.4 gives 49 topics, 11 merged groups, largest 3,
and every group is a plausible duplicate set (residency ↔ residency
requirements, driver's licences ↔ licence renewal, the three crypto-tax
reporting themes, SIM ↔ roaming, Graphene NFC ↔ digital payments NFC, …).
Average linkage is nearly identical. Use complete.

## 3. Message → topic routing

Choice question with 62 options (61 themes as `name: first sentence` plus
`none`), 3 preceding messages as context.

| shape | requests | input tokens | cost | wall |
|---|---:|---:|---:|---:|
| window: 8 messages per request, one choice each | 137 | 3.55M | $0.149 | ~3.5 min paced |
| single: one message per request (150-message sample) | 150 | 0.55M | $0.023 | — |

Quality on a fixed 150-message sample (seed 7). Each judge saw the message,
5 earlier and 2 later messages, and the distinct candidate topics chosen by
the four methods (shuffled, lettered), and named the best fit plus the
acceptable ones. Judge–judge agreement on "best": 89.7%.

| method | astra best | astra acceptable | luna best | luna acceptable | both judges' best |
|---|---:|---:|---:|---:|---:|
| **Jev window** | **84.2%** | **87.7%** | **83.8%** | 83.8% | **79.3%** |
| Jev single | 73.3% | 76.7% | 75.7% | 75.7% | 70.3% |
| gemma-4-31b (chat, JSON) | 78.8% | 82.9% | 81.8% | 83.8% | 75.9% |
| glm-flash (chat, JSON) | 80.8% | 83.6% | 77.0% | 78.4% | 73.8% |

The window shape is not just cheaper, it is more accurate than one message per
request: the neighbouring messages inside the state resolve what a reply is
about. Jev window agrees with gemma on 74% of messages and with its own single
shape on 76%, so the remaining disagreement is at model-noise level.

Diagnostics on the full 1,095 messages: 24.5% routed to `none` (the chat
models on the sample: 27%), all 61 topics used, mean confidence 0.74.
Confidence is informative: assignments with confidence ≥ 0.8 fall in a chunk
that created or updated the theme 78% of the time; below 0.6 it is 33%, with a
median distance of 3–7 chunks. Two failure modes seen:

- **Bare reactions get a topic.** 18 of the 37 messages of ≤12 characters
  without quote or attachment (":)", "Lol", "Aha", "imo") were routed to a
  topic instead of `none`, at confidence 0.3–0.9. A length gate before the call
  fixes this for free.
- **Near-duplicate themes are attractors.** "Tax Residency and LLC Risks"
  (home chunk 14) collected 69 messages spread over 10 chunks; "Paraguay
  Driver's Licenses" (chunk 2) took the later licence-renewal messages. These
  are the sliding window's duplicate themes, not routing errors; routing to
  merged topics rather than raw themes removes most of it.

## 4. End to end, judged blind

Three candidates from the same fixture:

- **nn-llm** — incumbent: mxbai top-3 retrieval → gemma pair verdicts → gemma
  merge prompt. 38 topics, 329 s.
- **jev-verdict** — Jev batch matrix, complete linkage at 2.4 (veto if either
  direction < 1.5) → gemma merge prompt per group. 49 topics, 16 s after the
  80 s matrix.
- **jev-e2e** — same groups; every group with ≥ 3 routed messages gets its
  name/summary/dissent rewritten by glm-flash from those messages (42 of 49
  regenerated, 827 messages routed, $0.02, 58 s). 49 topics, 3.7x the text of
  the others.

Each judge saw the full line-numbered source (~65k tokens) and two candidates,
both orders, 12 judgments, $6.22.

| criterion | nn-llm | jev-verdict | jev-e2e | tie |
|---|---:|---:|---:|---:|
| faithfulness | 5 | 3 | 3 | 1 |
| specificity | 0 | 4 | 8 | 0 |
| structure | 4 | 4 | 4 | 0 |
| coverage | 0 | 4 | 8 | 0 |
| **overall** | **1** | **6** | **5** | 0 |

Head to head (both judges, both orders):

- jev-verdict beats nn-llm **4–0**. Both judges call it narrow: better
  separation of concrete discussions and a few more actionable facts, "not
  through length". nn-llm's consolidation "loses distinct discussions without
  fixing the shared" problems. nn-llm has slightly fewer unsupported claims but
  also invents (a "Form 623" that is not in the source).
- jev-e2e beats nn-llm **3–1**; astra flipped once on order.
- jev-verdict vs jev-e2e is a **judge split**: Luna prefers e2e twice
  ("substantially more specific… concrete prices, thresholds, services,
  procedures and corrections"), astra prefers verdict twice ("repeatedly turns
  replies into authorship, assigns statements to the wrong people, merges
  unrelated procedures"). Flagged faithfulness issues: luna 13 / 13 / 19,
  astra 20 / 20 / 20 for nn-llm / verdict / e2e, so per unit of text e2e is
  cleaner, but its errors are of a worse kind — a quoted question becoming
  someone's experience, a scam victim swapped, a telecom plan described as a
  tax regime. Those come from misrouted or context-stripped messages being
  summarized as if they belonged.

Structural cross-check (mxbai cosine ≥ 0.80 between section texts): nn-llm 0
near-duplicate pairs, jev-verdict 7, jev-e2e 5. The review already noted this
metric mislabels distinct-but-adjacent topics; the jev pairs it flags are
residency ↔ residency-documentation (Jev 2.64, split only because complete
linkage refused a third member) and crypto-tax ↔ DNIT-resolution — the
granularity question again, not a Jev error.

## 5. What this means for the summarizer

1. **Jev as the merge decision is ready to try as an optional engine.** It
   replaces both embedding retrieval and the LLM pair verdict: one request per
   theme, full matrix, complete linkage at a configurable threshold (start at
   2.4), veto below 1.5 in either direction, then the existing merge prompt.
   It is faster (80 s vs 329 s on this fixture), 10x cheaper than the verdict
   calls it replaces, and won the blind comparison. Keep the current path as
   the default; Jev is beta, `anonymized` tier only (no E2EE build), and
   `jev-latest` can move.
2. **Jev routing is good enough to build on**, but regenerate-from-messages is
   not a drop-in improvement yet. To make it one: gate short messages to
   `none` before the call, route to merged topics rather than raw themes, keep
   only assignments with confidence ≥ 0.6 (or send low-confidence ones to
   `none`), and give the regeneration prompt the reply structure and a hard
   2–4 sentence cap. The attribution errors are exactly the class the review
   proposed a claim-level verifier for; routing gives that verifier the
   evidence IDs it needs.
3. **Do not use the `noul` probability as an absolute.** Use `score` and
   calibrate on labelled pairs; recalibrate when the answer distribution moves.
4. **Rate limit, not latency, bounds throughput.** 100 requests/min per key
   is shared across projects on the same key; a 1,095-message group is ~140
   routing requests, so a monthly run over all groups is minutes, but two
   concurrent benchmarks will collide. Pace client-side.

## 6. Reproduce

```bash
uv run python bench/jev_bench.py pairs          # 61 + 211 Jev requests, ~3 min
uv run python bench/jev_bench.py pairs-judge    # 4 verdict models + tiebreak on 211 pairs
uv run python bench/jev_bench.py assign         # 137 + 150 Jev requests + 300 chat calls
uv run python bench/jev_bench.py assign-judge   # 300 judge calls (~$1.7, mostly astra)
uv run python bench/jev_bench.py e2e --threshold 2.4 --linkage complete
uv run python bench/jev_bench.py e2e-judge      # 12 judge calls (~$6.2)
```

Every request is cached by payload hash under `bench/results/jev-2026-09-19/cache/`,
so re-running any stage is free; delete a cache subdirectory to re-measure.
Needs `VENICE_API_KEY` in the environment (falls back to `config-venice.json`)
and local Ollama with `mxbai-embed-large` for the incumbent run and the
structural check.
