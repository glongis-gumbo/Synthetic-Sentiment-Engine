"""
Synthetic Financial Content Generator
Generates clearly-marked synthetic news and social content for algo stress testing.
All company names are fictional. Content is watermarked and must not be published.
"""

from __future__ import annotations
import json
import random
import re
from dataclasses import dataclass, field
from typing import Literal, Optional
from datetime import datetime, timezone
import anthropic

SYNTHETIC_MARKER = "[SYNTHETIC_TEST_CONTENT — NOT REAL — FOR INTERNAL STRESS TESTING ONLY]"

SentimentPolarity = Literal["bullish", "bearish", "neutral", "mixed"]
ContentFormat = Literal["news_article", "tweet", "reddit_post", "stocktwits", "analyst_note"]


@dataclass
class GenerationRequest:
    scenario_id: str
    company_name: str
    ticker: str
    sector: str
    polarity: SentimentPolarity
    intensity: float                    # 0.0–1.0
    content_format: ContentFormat
    catalyst: str                       # e.g. "Q3 earnings beat 25%"
    quarter: str = "Q3"
    fiscal_year: str = "2025"
    noise_skeptic: bool = False         # Inject contrarian noise
    cascade_event: Optional[str] = None


@dataclass
class GeneratedContent:
    request: GenerationRequest
    raw_text: str
    synthetic_marker: str = SYNTHETIC_MARKER
    sentiment_score: Optional[float] = None   # Post-generation validation score
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "synthetic_marker": self.synthetic_marker,
            "timestamp": self.timestamp,
            "ticker": self.request.ticker,
            "polarity": self.request.polarity,
            "intensity": self.request.intensity,
            "format": self.request.content_format,
            "text": self.raw_text,
            "sentiment_score": self.sentiment_score,
            "metadata": self.metadata,
        }


class SyntheticContentGenerator:
    """
    LLM-backed generator for synthetic financial market content.
    Designed for stress testing algorithmic sentiment analysis pipelines.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model
        self._prompt_cache: dict[str, str] = {}

    # ── Prompt builders ──────────────────────────────────────────────────────

    def _system_prompt(self) -> str:
        return """You are a synthetic financial content generator for internal stress testing.
Your output is CLEARLY FICTIONAL content used by quantitative analysts to test
sentiment analysis algorithms in a controlled research environment.

CRITICAL RULES:
1. All companies are fictional (provided in the request). Never use real company names.
2. All tickers are fictional placeholders. Never use real stock tickers.
3. Never generate content that could constitute market manipulation if published.
4. Never generate specific false price targets or precise financial figures
   that could be mistaken for real analyst research.
5. Content is for internal testing ONLY — it will never be published.
6. Match the requested sentiment intensity precisely on a 0–1 scale.

Intensity guide:
  0.1–0.3: Mildly directional, balanced with caveats
  0.4–0.6: Clearly directional, some nuance remains
  0.7–0.85: Strong conviction, limited hedging
  0.86–1.0: Extreme, capitulation or euphoria language"""

    def _build_news_article_prompt(self, req: GenerationRequest) -> str:
        intensity_desc = self._intensity_label(req.intensity)
        skeptic_note = (
            "Include one short skeptical analyst quote to add realism."
            if req.noise_skeptic else ""
        )
        cascade_note = (
            f"Also briefly mention emerging concern: {req.cascade_event}."
            if req.cascade_event else ""
        )

        return f"""Generate a synthetic financial news article with these exact parameters:

COMPANY: {req.company_name} (FICTIONAL)
TICKER: {req.ticker} (FICTIONAL PLACEHOLDER)
SECTOR: {req.sector}
SENTIMENT: {req.polarity} at intensity {req.intensity:.2f} ({intensity_desc})
CATALYST: {req.catalyst}
QUARTER/PERIOD: {req.quarter} {req.fiscal_year}

FORMAT REQUIREMENTS:
- Headline (punchy, reflects sentiment intensity)
- Dateline: [SYNTHETIC TEST DATE]
- 3–4 paragraph body
- At least one (fictional) analyst quote
- Brief mention of share price reaction (use percentage, not absolute price)
- Standard AP financial journalism style
{skeptic_note}
{cascade_note}

Begin with the headline, then the article body. Do not include any meta-commentary."""

    def _build_social_prompt(self, req: GenerationRequest, platform: str) -> str:
        platform_styles = {
            "tweet": "280-character max, use ${ticker} format, 1–2 hashtags, casual tone",
            "reddit_post": "r/wallstreetbets or r/investing style. Title + 2–3 sentence body. Can include emojis if high intensity bullish.",
            "stocktwits": "Stocktwits style: $TICKER at start, 140 chars, use cashtags, include bull/bear emoji if appropriate",
        }
        style = platform_styles.get(platform, "short-form social post")
        intensity_desc = self._intensity_label(req.intensity)

        return f"""Generate a synthetic {platform} post for internal stress testing:

FICTIONAL TICKER: ${req.ticker}
FICTIONAL COMPANY: {req.company_name}
SENTIMENT: {req.polarity} at intensity {req.intensity:.2f} ({intensity_desc})
CATALYST: {req.catalyst}
STYLE: {style}

Generate ONLY the post text, no meta-commentary. Make it sound authentic to the platform."""

    def _build_analyst_note_prompt(self, req: GenerationRequest) -> str:
        action = "Upgrade" if req.polarity == "bullish" else "Downgrade"
        return f"""Generate a synthetic sell-side analyst note excerpt for stress testing:

FICTIONAL COMPANY: {req.company_name}
FICTIONAL TICKER: {req.ticker}
SECTOR: {req.sector}
ACTION: {action} (fictional)
SENTIMENT INTENSITY: {req.intensity:.2f}
CATALYST: {req.catalyst}

Include:
- Rating change line (use fictional bank name like "Westfield Securities")
- 2–3 sentence investment thesis
- Key risk factors (1–2 bullets)
- Do NOT include specific price targets (regulatory compliance)

Output only the note excerpt."""

    # ── Core generation ───────────────────────────────────────────────────────

    def generate(self, req: GenerationRequest) -> GeneratedContent:
        prompt = self._select_prompt(req)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self._system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = message.content[0].text.strip()

        return GeneratedContent(
            request=req,
            raw_text=raw_text,
            metadata={
                "model": self.model,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "stop_reason": message.stop_reason,
            },
        )

    def generate_batch(
        self,
        requests: list[GenerationRequest],
        validate: bool = True,
    ) -> list[GeneratedContent]:
        """Generate a batch, optionally running the validator pass."""
        results = [self.generate(r) for r in requests]
        if validate:
            from engine.validator import ContentValidator
            validator = ContentValidator()
            for content in results:
                validator.validate(content)
        return results

    # ── Scenario builder helpers ──────────────────────────────────────────────

    def build_scenario_batch(
        self,
        scenario_config: dict,
        n_articles: int = 5,
        n_social: int = 20,
    ) -> list[GenerationRequest]:
        """
        Converts a loaded YAML scenario into a mixed batch of generation requests.
        Applies noise_injection ratios from config to vary polarity within the batch.
        """
        requests = []
        companies = scenario_config["company_templates"]
        polarity: SentimentPolarity = scenario_config["sentiment"]
        base_intensity: float = scenario_config["intensity"]
        skeptic_ratio: float = scenario_config.get("noise_injection", {}).get("skeptic_ratio", 0.1)

        catalyst = scenario_config.get("triggers", {})
        catalyst_str = self._catalyst_string(catalyst, polarity)

        # News articles
        for i in range(n_articles):
            company = random.choice(companies)
            noise = random.random() < skeptic_ratio
            # Add slight intensity jitter for realism
            jittered = min(1.0, max(0.0, base_intensity + random.gauss(0, 0.08)))
            requests.append(GenerationRequest(
                scenario_id=scenario_config["scenario_id"],
                company_name=company["name"],
                ticker=company["ticker"],
                sector=company["sector"],
                polarity="bearish" if (noise and polarity == "bullish") else polarity,
                intensity=jittered,
                content_format="news_article",
                catalyst=catalyst_str,
                noise_skeptic=noise,
            ))

        # Social posts across platforms
        platforms = ["tweet", "reddit_post", "stocktwits"]
        for i in range(n_social):
            company = random.choice(companies)
            platform = random.choice(platforms)
            noise = random.random() < skeptic_ratio
            jittered = min(1.0, max(0.0, base_intensity + random.gauss(0, 0.12)))
            requests.append(GenerationRequest(
                scenario_id=scenario_config["scenario_id"],
                company_name=company["name"],
                ticker=company["ticker"],
                sector=company["sector"],
                polarity="bearish" if (noise and polarity == "bullish") else polarity,
                intensity=jittered,
                content_format=platform,
                catalyst=catalyst_str,
            ))

        return requests

    # ── Utilities ────────────────────────────────────────────────────────────

    def _select_prompt(self, req: GenerationRequest) -> str:
        if req.content_format == "news_article":
            return self._build_news_article_prompt(req)
        elif req.content_format == "analyst_note":
            return self._build_analyst_note_prompt(req)
        else:
            return self._build_social_prompt(req, req.content_format)

    def _intensity_label(self, intensity: float) -> str:
        if intensity < 0.3:
            return "mild/cautious"
        elif intensity < 0.55:
            return "moderate"
        elif intensity < 0.75:
            return "strong"
        elif intensity < 0.9:
            return "very strong"
        return "extreme/capitulation"

    def _catalyst_string(self, triggers: dict, polarity: str) -> str:
        if polarity == "bullish":
            if "eps_beat_pct" in triggers:
                pct = random.choice(triggers["eps_beat_pct"])
                return f"Q3 EPS beat consensus by {pct}%"
        elif polarity == "bearish":
            if "agency" in triggers:
                agency = random.choice(triggers["agency"])
                probe = random.choice(triggers.get("probe_type", ["inquiry"]))
                return f"{agency} {probe} disclosed"
        return "significant corporate event"
