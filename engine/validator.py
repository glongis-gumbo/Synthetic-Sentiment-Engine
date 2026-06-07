"""
Content Validator: ensures synthetic content is safe, well-formed, and
sentiment-accurate relative to the generation request parameters.
"""

from __future__ import annotations
import re
import json
import anthropic
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.generator import GeneratedContent

# Real ticker patterns to catch accidental inclusion
REAL_TICKER_PATTERN = re.compile(
    r"\b(AAPL|GOOGL|MSFT|AMZN|TSLA|NVDA|META|NFLX|JPM|GS|BRK|SPY|QQQ)\b"
)

REAL_COMPANY_PATTERN = re.compile(
    r"\b(Apple|Google|Microsoft|Amazon|Tesla|Nvidia|Meta|Netflix|"
    r"Goldman|JPMorgan|Berkshire|BlackRock|Vanguard)\b",
    re.IGNORECASE,
)


@dataclass
class ValidationResult:
    passed: bool
    sentiment_score: float      # -1.0 (extreme bearish) to +1.0 (extreme bullish)
    intensity_delta: float      # abs diff between requested and measured intensity
    flags: list[str]            # safety/quality issues found
    safe_to_use: bool           # False = reject from stress test dataset


class ContentValidator:
    """
    Two-pass validator:
    1. Rule-based safety check (fast, no LLM calls)
    2. LLM-based sentiment scoring to verify intensity accuracy
    """

    def __init__(self, intensity_tolerance: float = 0.2):
        self.client = anthropic.Anthropic()
        self.intensity_tolerance = intensity_tolerance

    def validate(self, content: "GeneratedContent") -> ValidationResult:
        flags: list[str] = []

        # ── Pass 1: Rule-based safety checks ─────────────────────────────────
        flags.extend(self._check_real_entities(content.raw_text))
        flags.extend(self._check_length(content))
        flags.extend(self._check_synthetic_marker_absent(content.raw_text))

        # ── Pass 2: LLM sentiment scoring ────────────────────────────────────
        sentiment_score = self._score_sentiment(content.raw_text, content.request.ticker)

        # Map polarity to expected sign
        expected_sign = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0, "mixed": 0.0}
        expected_direction = expected_sign.get(content.request.polarity, 0.0)
        expected_intensity_signed = expected_direction * content.request.intensity
        intensity_delta = abs(sentiment_score - expected_intensity_signed)

        if intensity_delta > self.intensity_tolerance:
            flags.append(
                f"INTENSITY_DRIFT: requested {content.request.intensity:.2f} "
                f"{content.request.polarity}, measured {sentiment_score:.2f} "
                f"(delta {intensity_delta:.2f})"
            )

        safe = len([f for f in flags if f.startswith("SAFETY")]) == 0

        result = ValidationResult(
            passed=len(flags) == 0,
            sentiment_score=sentiment_score,
            intensity_delta=intensity_delta,
            flags=flags,
            safe_to_use=safe,
        )

        # Attach validation results back to content
        content.sentiment_score = sentiment_score
        content.metadata["validation"] = {
            "passed": result.passed,
            "safe_to_use": result.safe_to_use,
            "flags": flags,
            "intensity_delta": intensity_delta,
        }

        return result

    # ── Rule-based checks ─────────────────────────────────────────────────────

    def _check_real_entities(self, text: str) -> list[str]:
        flags = []
        ticker_match = REAL_TICKER_PATTERN.search(text)
        if ticker_match:
            flags.append(f"SAFETY: Real ticker found: {ticker_match.group()}")
        company_match = REAL_COMPANY_PATTERN.search(text)
        if company_match:
            flags.append(f"SAFETY: Real company name found: {company_match.group()}")
        return flags

    def _check_length(self, content: "GeneratedContent") -> list[str]:
        length = len(content.raw_text)
        if content.request.content_format == "news_article" and length < 300:
            return ["QUALITY: News article too short (<300 chars)"]
        if content.request.content_format in ("tweet", "stocktwits") and length > 300:
            return ["QUALITY: Social post too long for platform format"]
        return []

    def _check_synthetic_marker_absent(self, text: str) -> list[str]:
        # The marker is added by the wrapper, not the LLM output itself.
        # Flag if LLM accidentally included it (means it saw examples).
        if "SYNTHETIC_TEST_CONTENT" in text:
            return ["QUALITY: LLM included synthetic marker verbatim in output"]
        return []

    # ── LLM sentiment scoring ─────────────────────────────────────────────────

    def _score_sentiment(self, text: str, ticker: str) -> float:
        """
        Uses a second LLM call to score actual sentiment of generated content.
        Returns float in [-1.0, +1.0].
        """
        message = self.client.messages.create(
            model="claude-haiku-4-5-20251001",   # Haiku for cost efficiency on scoring
            max_tokens=64,
            messages=[{
                "role": "user",
                "content": f"""Score the financial sentiment of this text about ${ticker}.

Return ONLY a JSON object: {{"score": <float from -1.0 to 1.0>}}
-1.0 = extreme bearish/negative, 0.0 = neutral, +1.0 = extreme bullish/positive

TEXT:
{text[:800]}"""
            }],
        )
        raw = message.content[0].text.strip()
        try:
            # Extract JSON even if surrounded by markdown fences
            json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if json_match:
                return float(json.loads(json_match.group())["score"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return 0.0
