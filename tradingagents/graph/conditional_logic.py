# TradingAgents/graph/conditional_logic.py

import json

from tradingagents.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    # Note: should_continue_* methods for analysts removed — tool loops
    # now run inside each analyst node via run_tool_loop(), avoiding
    # shared-state contamination during parallel fan-out.

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 3 rounds of back-and-forth between 2 agents
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_retry_after_evaluation(self, state: AgentState) -> str:
        """Route after evaluator: pass -> Trader, fail -> Retry Gate."""
        eval_retry_count = state.get("eval_retry_count", 0)

        report_str = state.get("evaluation_report", "")
        if not report_str:
            # Fail-closed: missing report = failed evaluation
            return "Trader" if eval_retry_count >= 2 else "Retry Gate"
        try:
            report = json.loads(report_str)
        except (json.JSONDecodeError, ValueError):
            # Fail-closed: unparseable report = failed evaluation
            return "Trader" if eval_retry_count >= 2 else "Retry Gate"

        passed = report.get("pass", False)
        # Bool coercion: handle LLM returning "false" as string
        if isinstance(passed, str):
            passed = passed.lower() in ("true", "1", "yes")

        if passed:
            return "Trader"

        # Failed evaluation: force-proceed after max retries, otherwise retry.
        # The failing evaluation_report is preserved so downstream knows
        # quality was poor.
        return "Trader" if eval_retry_count >= 2 else "Retry Gate"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 rounds of back-and-forth between 3 agents
            return "Risk Judge"
        if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
            return "Conservative Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
            return "Neutral Analyst"
        return "Aggressive Analyst"
