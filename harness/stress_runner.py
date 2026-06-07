"""
Stress Test Harness
Feeds synthetic content into sentiment analysis pipelines and records
how algorithmic strategies respond to controlled sentiment scenarios.
"""

from __future__ import annotations
import json
import time
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Callable, Optional
from collections import defaultdict


@dataclass
class SentimentSignal:
    """Normalized signal fed to the trading strategy under test."""
    ticker: str
    timestamp: str
    raw_score: float        # -1.0 to +1.0
    source_type: str        # news_article / tweet / reddit_post / analyst_note
    intensity: float
    polarity: str
    scenario_id: str


@dataclass
class StrategyResponse:
    """Records how a strategy reacted to a sentiment signal."""
    signal: SentimentSignal
    action: str             # BUY / SELL / HOLD / REDUCE
    confidence: float       # 0.0–1.0
    latency_ms: float
    rationale: str


@dataclass
class StressTestResult:
    scenario_id: str
    total_signals: int
    strategy_responses: list[StrategyResponse]
    metrics: dict = field(default_factory=dict)

    def summary(self) -> dict:
        actions = defaultdict(int)
        for r in self.strategy_responses:
            actions[r.action] += 1

        latencies = [r.latency_ms for r in self.strategy_responses]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0

        return {
            "scenario_id": self.scenario_id,
            "total_signals": self.total_signals,
            "action_distribution": dict(actions),
            "avg_latency_ms": round(avg_lat, 2),
            "metrics": self.metrics,
        }


class StressTestHarness:
    """
    Drives synthetic signals through a pluggable strategy function.

    The `strategy_fn` is a callable that represents your algo:
        strategy_fn(signal: SentimentSignal) -> StrategyResponse

    This lets you test multiple strategies against the same synthetic dataset.
    """

    def __init__(self, strategy_fn: Callable[[SentimentSignal], StrategyResponse]):
        self.strategy_fn = strategy_fn
        self.results: list[StressTestResult] = []

    def run_scenario(
        self,
        scenario_id: str,
        generated_contents: list,       # list[GeneratedContent]
        inject_delay: bool = False,     # Simulate real-time feed timing
    ) -> StressTestResult:
        signals = self._contents_to_signals(generated_contents)
        responses: list[StrategyResponse] = []

        for signal in signals:
            if inject_delay:
                # Simulate realistic inter-event timing (Poisson process)
                time.sleep(random.expovariate(5.0) * 0.01)

            t0 = time.perf_counter()
            response = self.strategy_fn(signal)
            latency_ms = (time.perf_counter() - t0) * 1000
            response.latency_ms = latency_ms
            responses.append(response)

        result = StressTestResult(
            scenario_id=scenario_id,
            total_signals=len(signals),
            strategy_responses=responses,
            metrics=self._compute_metrics(responses, signals),
        )
        self.results.append(result)
        return result

    def run_multi_scenario(
        self,
        scenario_batches: dict[str, list],
    ) -> list[StressTestResult]:
        return [
            self.run_scenario(sid, contents)
            for sid, contents in scenario_batches.items()
        ]

    def export_results(self, path: str) -> None:
        export = [r.summary() for r in self.results]
        with open(path, "w") as f:
            json.dump(export, f, indent=2)
        print(f"Results exported to {path}")

    # ── Signal conversion ─────────────────────────────────────────────────────

    def _contents_to_signals(self, contents: list) -> list[SentimentSignal]:
        signals = []
        for c in contents:
            if not c.metadata.get("validation", {}).get("safe_to_use", True):
                continue  # Skip content that failed validation
            signals.append(SentimentSignal(
                ticker=c.request.ticker,
                timestamp=c.timestamp,
                raw_score=c.sentiment_score or 0.0,
                source_type=c.request.content_format,
                intensity=c.request.intensity,
                polarity=c.request.polarity,
                scenario_id=c.request.scenario_id,
            ))
        return signals

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        responses: list[StrategyResponse],
        signals: list[SentimentSignal],
    ) -> dict:
        if not responses:
            return {}

        bull_signals = [s for s in signals if s.polarity == "bullish"]
        bear_signals = [s for s in signals if s.polarity == "bearish"]

        buy_on_bull = sum(
            1 for r in responses
            if r.signal.polarity == "bullish" and r.action == "BUY"
        )
        sell_on_bear = sum(
            1 for r in responses
            if r.signal.polarity == "bearish" and r.action == "SELL"
        )

        return {
            "signal_alignment_rate": round(
                (buy_on_bull + sell_on_bear) / max(len(responses), 1), 3
            ),
            "bullish_signal_count": len(bull_signals),
            "bearish_signal_count": len(bear_signals),
            "strategy_buy_rate": round(
                sum(1 for r in responses if r.action == "BUY") / max(len(responses), 1), 3
            ),
            "avg_confidence": round(
                sum(r.confidence for r in responses) / max(len(responses), 1), 3
            ),
        }


# ── Example strategy implementations for testing ─────────────────────────────

def naive_threshold_strategy(signal: SentimentSignal) -> StrategyResponse:
    """
    Simple threshold-based strategy. Good baseline for comparison.
    BUY if score > 0.4, SELL if score < -0.4, else HOLD.
    """
    if signal.raw_score > 0.4:
        action, confidence = "BUY", min(1.0, signal.raw_score)
        rationale = f"Positive sentiment score {signal.raw_score:.2f} exceeds BUY threshold"
    elif signal.raw_score < -0.4:
        action, confidence = "SELL", min(1.0, abs(signal.raw_score))
        rationale = f"Negative sentiment score {signal.raw_score:.2f} exceeds SELL threshold"
    else:
        action, confidence = "HOLD", 0.5
        rationale = "Score within neutral band"

    return StrategyResponse(
        signal=signal, action=action, confidence=confidence,
        latency_ms=0.0, rationale=rationale,
    )


def source_weighted_strategy(signal: SentimentSignal) -> StrategyResponse:
    """
    Weights sentiment signal by source credibility.
    News articles and analyst notes carry more weight than social media.
    """
    source_weights = {
        "news_article": 1.0,
        "analyst_note": 1.2,
        "reddit_post": 0.4,
        "tweet": 0.3,
        "stocktwits": 0.25,
    }
    weight = source_weights.get(signal.source_type, 0.5)
    adjusted_score = signal.raw_score * weight

    if adjusted_score > 0.45:
        action, confidence = "BUY", min(1.0, adjusted_score)
    elif adjusted_score < -0.35:
        action, confidence = "SELL", min(1.0, abs(adjusted_score))
    elif adjusted_score < -0.15:
        action, confidence = "REDUCE", 0.4
    else:
        action, confidence = "HOLD", 0.5

    return StrategyResponse(
        signal=signal, action=action, confidence=confidence,
        latency_ms=0.0,
        rationale=f"Source-weighted score: {adjusted_score:.2f} (weight {weight})",
    )
