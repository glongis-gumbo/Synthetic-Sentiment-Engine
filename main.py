"""
Synthetic Financial Content Stress Test — Main Runner

Usage:
    python main.py --scenario earnings_beat --articles 5 --social 20
    python main.py --scenario regulatory_probe --strategy source_weighted

IMPORTANT: All generated content is fictional and watermarked.
           Never publish or distribute generated content.
"""

import argparse
import json
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from engine.generator import SyntheticContentGenerator
from engine.validator import ContentValidator
from harness.stress_runner import (
    StressTestHarness,
    naive_threshold_strategy,
    source_weighted_strategy,
)

STRATEGIES = {
    "naive": naive_threshold_strategy,
    "source_weighted": source_weighted_strategy,
}

SCENARIOS_DIR = Path(__file__).parent / "scenarios"


def load_scenario(name: str) -> dict:
    for subdir in ["bullish", "bearish"]:
        path = SCENARIOS_DIR / subdir / f"{name}.yaml"
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"Scenario '{name}' not found in {SCENARIOS_DIR}")


def run_stress_test(
    scenario_name: str,
    n_articles: int,
    n_social: int,
    strategy_name: str,
    output_path: str,
) -> None:
    print(f"\n{'='*60}")
    print(f"  SYNTHETIC SENTIMENT STRESS TEST")
    print(f"  Scenario: {scenario_name} | Strategy: {strategy_name}")
    print(f"  [ALL CONTENT IS FICTIONAL — INTERNAL USE ONLY]")
    print(f"{'='*60}\n")

    # 1. Load scenario config
    scenario = load_scenario(scenario_name)
    print(f"[1/4] Loaded scenario: {scenario['scenario_id']} ({scenario['sentiment']} @ {scenario['intensity']})")

    # 2. Build generation requests
    generator = SyntheticContentGenerator()
    requests = generator.build_scenario_batch(scenario, n_articles=n_articles, n_social=n_social)
    print(f"[2/4] Built {len(requests)} generation requests ({n_articles} articles + {n_social} social posts)")

    # 3. Generate & validate
    print(f"[3/4] Generating synthetic content (this may take ~{len(requests) * 2}s)...")
    contents = generator.generate_batch(requests, validate=True)

    passed = sum(1 for c in contents if c.metadata.get("validation", {}).get("passed"))
    safe = sum(1 for c in contents if c.metadata.get("validation", {}).get("safe_to_use"))
    print(f"      Generated: {len(contents)} | Passed validation: {passed} | Safe: {safe}")

    # 4. Run stress test
    strategy_fn = STRATEGIES[strategy_name]
    harness = StressTestHarness(strategy_fn)
    result = harness.run_scenario(scenario_name, contents)

    # 5. Report
    summary = result.summary()
    print(f"\n[4/4] RESULTS:")
    print(f"      Signals processed:    {summary['total_signals']}")
    print(f"      Action distribution:  {summary['action_distribution']}")
    print(f"      Signal alignment:     {summary['metrics'].get('signal_alignment_rate', 'N/A')}")
    print(f"      Buy rate:             {summary['metrics'].get('strategy_buy_rate', 'N/A')}")
    print(f"      Avg confidence:       {summary['metrics'].get('avg_confidence', 'N/A')}")
    print(f"      Avg latency (ms):     {summary['avg_latency_ms']}")

    harness.export_results(output_path)
    print(f"\nFull results written to: {output_path}")

    # Sample content preview
    print(f"\n--- SAMPLE GENERATED CONTENT (first article) ---")
    articles = [c for c in contents if c.request.content_format == "news_article"]
    if articles:
        a = articles[0]
        print(f"[{a.request.ticker}] Polarity: {a.request.polarity} @ {a.request.intensity:.2f}")
        print(f"Measured sentiment score: {a.sentiment_score:.2f}")
        print(f"{a.synthetic_marker}")
        print(a.raw_text[:500] + "..." if len(a.raw_text) > 500 else a.raw_text)


def main():
    parser = argparse.ArgumentParser(description="Synthetic Financial Content Stress Test Engine")
    parser.add_argument("--scenario", default="earnings_beat",
                        choices=["earnings_beat", "analyst_upgrade", "regulatory_probe", "macro_headwind"])
    parser.add_argument("--articles", type=int, default=3)
    parser.add_argument("--social", type=int, default=10)
    parser.add_argument("--strategy", default="source_weighted", choices=list(STRATEGIES.keys()))
    parser.add_argument("--output", default="stress_test_results.json")
    args = parser.parse_args()

    run_stress_test(args.scenario, args.articles, args.social, args.strategy, args.output)


if __name__ == "__main__":
    main()
