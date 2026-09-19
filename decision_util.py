# File: decision_util.py
"""Client for typed-decision models (Venice's Jev, `POST /decisions`).

A decision model is not a chat model: it takes a `state` and a map of typed
questions and returns probabilities, never text. Configure it in `models`
with `"provider": "venice-decision"`; the summarizer uses it for judgments it
acts on directly (which themes are the same topic), and keeps a chat model
for everything that produces text.

Question types (see docs.venice.ai/guides/features/decisions):
  noul   {"type": "noul", "instructions": ...}                  -> {"noul": P(yes)}
  choice {"type": "choice", "instructions": ..., "criteria": {option: desc}}
                                                               -> {"choice", "probabilities", "confidence"}
  score  {"type": "score", "instructions": ..., "criteria": [level0, ...]}
                                                               -> {"score", "probabilities", "confidence"}
"""
import logging
import os
import random
import threading
import time
from typing import Any, Dict

import httpx

DECISION_PROVIDERS = ("venice-decision",)


class DecisionError(Exception):
    """A decision request failed after retries or returned a malformed body."""


class _Pacer:
    """Space request starts so a per-key rate limit is never hit.

    Venice allows 100 requests/min per key and locks the key for 30 s after
    50 non-success responses, so pacing client-side beats retrying 429s.
    Thread-safe: parallel callers share one schedule.
    """

    def __init__(self, min_interval):
        self.min_interval = max(0.0, float(min_interval))
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


class DecisionClient:
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.provider = model_config.get("provider", "venice-decision")
        if self.provider not in DECISION_PROVIDERS:
            raise ValueError(f"Unsupported decision provider: {self.provider}")
        self.logger = logging.getLogger("decision_util")
        self.model = model_config.get("model", "jev-latest")
        self.api_base = model_config.get(
            "apiBase", "https://api.venice.ai/api/v1"
        ).rstrip("/")
        self.api_key = model_config.get("apiKey") or os.environ.get("VENICE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Decision model needs an apiKey (or VENICE_API_KEY in the environment)"
            )
        self.request_timeout = model_config.get("request_timeout", 120)
        self.max_retries = int(model_config.get("max_retries", 6))
        # 0.65 s between request starts keeps one process under 100 req/min.
        # Lower it only if nothing else shares the key.
        self._pacer = _Pacer(model_config.get("min_request_interval", 0.65))
        self._client = httpx.Client(timeout=self.request_timeout)
        self.requests = 0
        self.input_tokens = 0
        self._stats_lock = threading.Lock()

    def effective_params(self):
        return {
            "provider": self.provider,
            "model": self.model,
            "request_timeout": self.request_timeout,
            "min_request_interval": self._pacer.min_interval,
        }

    def decide(self, state, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """One request. Returns the `answers` map keyed by the question ids.

        Retries 429/5xx and transport errors with backoff; any other HTTP
        status (a 400 on a schema change, for instance) raises immediately so
        the caller can fall back rather than hammer the endpoint.
        """
        payload = {"model": self.model, "state": state, "questions": questions}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last = None
        for attempt in range(self.max_retries):
            self._pacer.wait()
            try:
                response = self._client.post(
                    f"{self.api_base}/decisions", json=payload, headers=headers
                )
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
                self.logger.warning(
                    "Decision request failed (attempt %d/%d): %s",
                    attempt + 1, self.max_retries, last,
                )
                time.sleep(min(30, 2 * (attempt + 1)) + random.random())
                continue
            if response.status_code == 200:
                body = response.json()
                answers = body.get("answers")
                if not isinstance(answers, dict):
                    raise DecisionError(
                        f"Decision response without answers: {str(body)[:300]}"
                    )
                usage = body.get("usage") or {}
                with self._stats_lock:
                    self.requests += 1
                    self.input_tokens += int(usage.get("input_tokens", 0) or 0)
                return answers
            last = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code == 429 or response.status_code >= 500:
                self.logger.warning(
                    "Decision request rejected (attempt %d/%d): %s",
                    attempt + 1, self.max_retries, last,
                )
                time.sleep(min(40, 3 * 1.8 ** attempt) + random.random())
                continue
            raise DecisionError(f"Decision request failed: {last}")
        raise DecisionError(
            f"Decision request failed after {self.max_retries} attempts: {last}"
        )
