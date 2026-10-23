from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.ai.config import AIConfig, load_ai_config
from src.ai.provider import BaseLLMProvider, get_llm_provider
from src.ai.query_engine import ReadOnlyAnalyticsQueryEngine


def explain_historical_match(
    match_id: str,
    database_path: str | Path,
    config: Optional[AIConfig] = None,
    provider: Optional[BaseLLMProvider] = None,
) -> Dict[str, Any]:
    """Generate a data-grounded AI explanation of a historical match."""
    engine = ReadOnlyAnalyticsQueryEngine(database_path)
    details = engine.query_match_details(match_id)
    if not details or not details.get("match"):
        return {
            "summary": "Match details could not be retrieved from the database.",
            "grounded": False,
            "facts": None,
        }

    match = details["match"]
    batters = details.get("top_batters", [])
    bowlers = details.get("top_bowlers", [])
    turning_overs = details.get("turning_point_overs", [])

    date_str = str(match.get("date", "Unknown date"))[:10]
    t1 = match.get("team1", "Team 1")
    t2 = match.get("team2", "Team 2")
    winner = match.get("outcome_winner") or "No winner recorded"
    pom = match.get("player_of_match") or "Not awarded"
    venue = match.get("venue", "Unknown venue")
    toss_winner = match.get("toss_winner", "Unknown")
    toss_decision = match.get("toss_decision", "unknown")

    margin = ""
    if match.get("outcome_by_runs") and match["outcome_by_runs"] > 0:
        margin = f"by {int(match['outcome_by_runs'])} runs"
    elif match.get("outcome_by_wickets") and match["outcome_by_wickets"] > 0:
        margin = f"by {int(match['outcome_by_wickets'])} wickets"

    # Deterministic factual baseline
    batter_lines = [
        f"• {b['batter']} ({b['batting_team']}): {int(b['runs'])} runs off {int(b['balls'])} balls ({int(b['fours'])}x4, {int(b['sixes'])}x6)"
        for b in batters[:3]
    ]
    bowler_lines = [
        f"• {b['bowler']} ({b['bowling_team']}): {int(b['wickets'])}/{int(b['runs_conceded'])} ({int(b['balls'])} balls)"
        for b in bowlers[:3]
    ]
    over_lines = [
        f"• Innings {int(o['innings'])}, Over {int(o['over_number'])} ({o['batting_team']}): {int(o['over_runs'])} runs, {int(o['over_wickets'])} wickets"
        for o in turning_overs[:4]
    ]

    deterministic_summary = (
        f"### Historical Match Intelligence: {t1} vs {t2}\n\n"
        f"**Date & Venue:** {date_str} at {venue}\n\n"
        f"**Toss:** {toss_winner} elected to {toss_decision}.\n\n"
        f"**Match Outcome:** **{winner}** won {margin}. **Player of the Match:** {pom}.\n\n"
        f"#### Top Batting Contributions\n" + ("\n".join(batter_lines) if batter_lines else "No delivery stats.") + "\n\n"
        f"#### Decisive Bowling Spells\n" + ("\n".join(bowler_lines) if bowler_lines else "No delivery stats.") + "\n\n"
        f"#### Turning Points & High Momentum Overs\n" + ("\n".join(over_lines) if over_lines else "No high-scoring (15+ runs) or multi-wicket (2+) overs recorded.")
    )

    cfg = config or load_ai_config()
    llm = provider or get_llm_provider(cfg)

    if not cfg.is_configured and cfg.provider != "mock":
        return {
            "summary": deterministic_summary,
            "grounded": True,
            "facts": details,
        }

    system_prompt = (
        "You are an expert cricket match analyst. Write an articulate, grounded match debrief "
        "using ONLY the verified match details, top performances, and turning point overs provided. "
        "Highlight turning points, standout performers, and key momentum shifts. Do not invent events."
    )
    user_prompt = f"Explain this historical match using the following verified records:\n\n{deterministic_summary}"

    try:
        explanation = llm.generate_chat_completion(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        return {
            "summary": explanation,
            "grounded": True,
            "facts": details,
        }
    except Exception:
        return {
            "summary": deterministic_summary,
            "grounded": True,
            "facts": details,
        }


def explain_prediction(
    state: Dict[str, Any],
    feature_row: pd.DataFrame,
    prediction: Dict[str, Any],
    contributions: Dict[str, float],
    config: Optional[AIConfig] = None,
    provider: Optional[BaseLLMProvider] = None,
) -> Dict[str, Any]:
    """Generate a transparent explanation of the live chase model prediction."""
    chasing_team = state.get("batting_team", "Batting team")
    bowling_team = state.get("bowling_team", "Bowling team")
    chase_prob = prediction.get("chasing_team_probability", 0.5) * 100
    bowl_prob = prediction.get("bowling_team_probability", 0.5) * 100

    row = feature_row.iloc[0]
    runs_needed = int(row.get("runs_needed", 0))
    balls_left = int(row.get("balls_remaining", 0))
    wickets_left = int(row.get("wickets_remaining", 0))
    crr = float(row.get("current_run_rate", 0))
    rrr = float(row.get("required_run_rate", 0))

    # Sort feature contributions
    sorted_factors = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    top_factors_desc = []
    for feat, impact in sorted_factors[:4]:
        direction = "in favor of" if impact > 0 else "against"
        name = feat.replace("_", " ").title()
        top_factors_desc.append(f"• **{name}**: Impact of {impact:+.2f} log-odds ({direction} chase)")

    deterministic_summary = (
        f"### Chase Probability Assessment\n\n"
        f"At this match state, the calibrated logistic regression model projects **{chase_prob:.1f}% win probability** "
        f"for **{chasing_team}** and **{bowl_prob:.1f}%** for **{bowling_team}**.\n\n"
        f"#### Key State Metrics\n"
        f"- **Equation**: {runs_needed} runs needed off {balls_left} balls ({wickets_left} wickets in hand)\n"
        f"- **Run Rates**: Current run rate is {crr:.2f} RPO vs Required run rate of {rrr:.2f} RPO\n\n"
        f"#### Decisive Model Factors (Log-Odds Impact)\n"
        + "\n".join(top_factors_desc)
        + "\n\n"
        f"#### Analytical Caveats & Uncertainty\n"
        f"- Probabilities are historical empirical estimates based on a sigmoid-calibrated model (Test ROC-AUC 0.8995).\n"
        f"- This statistical model assumes a standard 120-ball chase and excludes external factors like pitch deterioration, weather, or unrecorded dropped catches.\n"
        f"- A prediction is an expectation, not a certainty: individual boundary streaks or multi-wicket overs can rapidly shift odds."
    )

    cfg = config or load_ai_config()
    llm = provider or get_llm_provider(cfg)

    if not cfg.is_configured and cfg.provider != "mock":
        return {"explanation": deterministic_summary, "grounded": True}

    system_prompt = (
        "You are an expert sports data scientist explaining a machine learning probability model. "
        "Explain the prediction clearly for an analyst or coach. Ground your reasoning strictly in the "
        "provided state numbers and logistic regression feature weights. Explain why the probability is "
        "high or low, cite the required run rate and wickets in hand, and remind the reader that sports models "
        "express historical likelihood, not guaranteed outcomes."
    )
    user_prompt = f"Explain this live chase prediction:\n\n{deterministic_summary}"

    try:
        text = llm.generate_chat_completion(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        return {"explanation": text, "grounded": True}
    except Exception:
        return {"explanation": deterministic_summary, "grounded": True}


def generate_player_intelligence_report(
    player_name: str,
    player_data: Dict[str, Any],
    config: Optional[AIConfig] = None,
    provider: Optional[BaseLLMProvider] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive scouting and intelligence report for a player."""
    p = player_data
    name = p.get("player_name", player_name)
    role = p.get("primary_role", "Cricket Player")
    matches = p.get("matches_selected", 0)

    # Batting
    runs = p.get("runs_scored", 0)
    avg = p.get("batting_average", 0) or 0
    sr = p.get("batting_strike_rate", 0) or 0
    fours = p.get("fours", 0)
    sixes = p.get("sixes", 0)
    bnd_pct = p.get("boundary_run_pct", 0) or 0
    pp_sr = p.get("powerplay_strike_rate", 0) or 0
    death_sr = p.get("death_strike_rate", 0) or 0
    pace_sr = p.get("sr_vs_pace", 0) or 0
    spin_sr = p.get("sr_vs_spin", 0) or 0

    # Bowling
    wickets = p.get("wickets", 0)
    econ = p.get("bowling_economy", 0) or 0
    bowl_avg = p.get("bowling_average", 0) or 0
    bowl_sr = p.get("bowling_strike_rate", 0) or 0

    report_text = (
        f"### Intelligence Dossier: {name}\n\n"
        f"**Role:** {role} | **Batting Style:** {p.get('bat_style', 'N/A')} | **Bowling Style:** {p.get('bowl_style', 'N/A')}\n"
        f"**Sample Size:** {matches} career matches recorded in profile directory.\n\n"
        f"#### Batting Profile\n"
        f"- **Volume & Productivity:** {runs:,} runs at an average of {avg:.2f} and strike rate of {sr:.2f}.\n"
        f"- **Boundary Impact:** {fours} fours, {sixes} sixes ({bnd_pct:.1f}% runs from boundaries).\n"
        f"- **Phase Breakdown:** Powerplay Strike Rate: {pp_sr:.1f} | Death Overs Strike Rate: {death_sr:.1f}.\n"
        f"- **Tactical Matchups:** Strike rate vs Pace: {pace_sr:.1f} | Strike rate vs Spin: {spin_sr:.1f}.\n\n"
        f"#### Bowling Profile\n"
        f"- **Wickets & Economy:** {wickets} wickets at an economy rate of {econ:.2f}.\n"
        f"- **Averages & Strike Rate:** Average {bowl_avg:.2f} runs per wicket, striking every {bowl_sr:.1f} legal balls.\n"
    )

    cfg = config or load_ai_config()
    llm = provider or get_llm_provider(cfg)

    if not cfg.is_configured and cfg.provider != "mock":
        return {"report": report_text, "grounded": True}

    system_prompt = (
        "You are an elite cricket talent scout and performance director. "
        "Synthesize the provided verified profile numbers into a rigorous, executive-level "
        "scouting report. Highlight tactical strengths, phase advantages, and potential vulnerabilities. "
        "Do not invent facts or numbers outside what is provided."
    )
    user_prompt = f"Generate an executive intelligence report from these facts:\n\n{report_text}"

    try:
        ai_report = llm.generate_chat_completion(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.2,
            max_tokens=700,
        )
        return {"report": ai_report, "grounded": True}
    except Exception:
        return {"report": report_text, "grounded": True}
