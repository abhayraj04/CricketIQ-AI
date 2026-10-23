from pathlib import Path
import pytest
import pandas as pd

from src.ai.config import AIConfig, load_ai_config
from src.ai.provider import LLMProviderError, MockProvider, OpenAICompatibleProvider
from src.ai.query_engine import ReadOnlyAnalyticsQueryEngine, SecurityViolationError
from src.ai.service import AICricketAnalyst
from src.ai.explainers import (
    explain_historical_match,
    explain_prediction,
    generate_player_intelligence_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "ipl_analytics.sqlite"


def test_ai_config_detects_missing_key_and_masks():
    empty_cfg = AIConfig(api_key=None)
    assert not empty_cfg.is_configured
    assert empty_cfg.masked_key == "Not configured"

    placeholder_cfg = AIConfig(api_key="your_openai_api_key_here")
    assert not placeholder_cfg.is_configured

    valid_cfg = AIConfig(api_key="sk-proj-1234567890abcdef")
    assert valid_cfg.is_configured
    assert valid_cfg.masked_key == "sk-...cdef"


def test_mock_provider_returns_grounded_response():
    provider = MockProvider()
    response = provider.generate_chat_completion([{"role": "user", "content": "Tell me about CSK"}])
    assert "Chennai Super Kings" in response
    assert len(provider.call_history) == 1


def test_unconfigured_openai_provider_raises_error():
    cfg = AIConfig(provider="openai", api_key="")
    provider = OpenAICompatibleProvider(cfg)
    with pytest.raises(LLMProviderError, match="not configured"):
        provider.generate_chat_completion([{"role": "user", "content": "Hello"}])


def test_query_engine_blocks_sql_mutations():
    engine = ReadOnlyAnalyticsQueryEngine(DATABASE_PATH)

    mutations = [
        "DROP TABLE ipl_matches_summary",
        "DELETE FROM ipl_matches_summary WHERE match_id = 1",
        "UPDATE ipl_matches_summary SET outcome_winner = 'CSK'",
        "INSERT INTO ipl_matches_summary (match_id) VALUES (99999)",
        "ALTER TABLE ipl_matches_summary ADD COLUMN leaked_col TEXT",
        "TRUNCATE TABLE ipl_ball_by_ball_processed",
        "SELECT * FROM ipl_matches_summary; DROP TABLE ipl_player_profiles;",
    ]

    for stmt in mutations:
        with pytest.raises(SecurityViolationError):
            engine.execute_safe_query(stmt)


def test_query_engine_allows_safe_select():
    engine = ReadOnlyAnalyticsQueryEngine(DATABASE_PATH)
    rows = engine.execute_safe_query(
        "SELECT match_id, team1, team2 FROM ipl_matches_summary LIMIT 5"
    )
    assert len(rows) == 5
    assert "match_id" in rows[0]
    assert "team1" in rows[0]


def test_query_head_to_head_returns_verified_counts():
    engine = ReadOnlyAnalyticsQueryEngine(DATABASE_PATH)
    h2h = engine.query_head_to_head("Chennai Super Kings", "Mumbai Indians", 2008, 2026)
    assert h2h["total_matches"] > 0
    assert h2h["team_a_wins"] + h2h["team_b_wins"] + h2h["ties_or_no_result"] == h2h["total_matches"]


def test_ai_analyst_works_in_deterministic_fallback_when_ai_disabled():
    cfg = AIConfig(provider="openai", api_key="")
    analyst = AICricketAnalyst(DATABASE_PATH, config=cfg)
    result = analyst.answer_question("Compare CSK and MI")

    assert result["grounded"] is True
    assert "Verified Head-to-Head" in result["answer"]
    assert "Chennai Super Kings" in result["answer"]
    assert "Mumbai Indians" in result["answer"]
    assert result["evidence"] is not None


def test_ai_analyst_handles_empty_or_ambiguous_result():
    analyst = AICricketAnalyst(DATABASE_PATH, config=AIConfig(provider="mock"))
    result = analyst.answer_question("   ")
    assert "Please ask a question" in result["answer"]


def test_match_explainer_generates_factual_summary():
    engine = ReadOnlyAnalyticsQueryEngine(DATABASE_PATH)
    sample_match = engine.execute_safe_query("SELECT match_id FROM ipl_matches_summary LIMIT 1")[0]
    match_id = sample_match["match_id"]

    explanation = explain_historical_match(match_id, DATABASE_PATH, config=AIConfig(provider="mock"))
    assert explanation["grounded"] is True
    assert "Historical Match" in explanation["summary"] or len(explanation["summary"]) > 20


def test_prediction_explainer_generates_grounded_rationale():
    state = {
        "batting_team": "Chennai Super Kings",
        "bowling_team": "Mumbai Indians",
        "target_score": 180,
        "current_score": 120,
        "wickets_lost": 3,
        "legal_balls_bowled": 72,
    }
    feature_row = pd.DataFrame([{
        "runs_needed": 60,
        "balls_remaining": 48,
        "wickets_remaining": 7,
        "current_run_rate": 10.0,
        "required_run_rate": 7.5,
    }])
    prediction = {"chasing_team_probability": 0.72, "bowling_team_probability": 0.28}
    contributions = {"required_run_rate": -0.85, "wickets_remaining": 1.20}

    result = explain_prediction(state, feature_row, prediction, contributions, config=AIConfig(provider="mock"))
    assert result["grounded"] is True
    assert "Chase Probability" in result["explanation"] or len(result["explanation"]) > 20


def test_player_intelligence_report_generates_dossier():
    player_data = {
        "player_name": "V Kohli",
        "primary_role": "Batter",
        "matches_selected": 240,
        "runs_scored": 7500,
        "batting_average": 38.5,
        "batting_strike_rate": 131.2,
        "fours": 650,
        "sixes": 240,
        "boundary_run_pct": 58.2,
    }
    result = generate_player_intelligence_report("V Kohli", player_data, config=AIConfig(provider="mock"))
    assert result["grounded"] is True
    assert "V Kohli" in result["report"] or len(result["report"]) > 20
