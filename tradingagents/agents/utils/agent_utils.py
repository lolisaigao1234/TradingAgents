from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )


def filter_messages_for_tools(messages, tool_names):
    """Filter messages to only include those belonging to a specific analyst's tool loop.

    During parallel fan-out, all analyst branches share the same messages state.
    This causes cross-contamination: one analyst sees AI tool_calls and ToolMessages
    from other analysts. Vertex AI then rejects the request because function call
    counts don't match function response counts.

    This filter keeps only:
    - HumanMessages (the initial prompt)
    - AIMessages whose tool_calls ALL reference tools in tool_names
    - ToolMessages whose name is in tool_names
    """
    tool_name_set = set(tool_names)
    filtered = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            filtered.append(msg)
        elif isinstance(msg, AIMessage):
            if not msg.tool_calls:
                # AI message with no tool calls (shouldn't appear mid-loop, but keep it)
                filtered.append(msg)
            elif all(tc["name"] in tool_name_set for tc in msg.tool_calls):
                filtered.append(msg)
            # else: tool_calls for another analyst's tools — skip
        elif isinstance(msg, ToolMessage):
            if getattr(msg, "name", None) in tool_name_set:
                filtered.append(msg)
            # else: tool response for another analyst — skip
        else:
            filtered.append(msg)
    return filtered


def run_tool_loop(chain, tools, messages, max_iterations=10):
    """Run an LLM tool-calling loop internally, avoiding shared graph state.

    During parallel fan-out in LangGraph, all branches share the same messages
    state channel. If tool calls and responses are added to the graph state,
    branches contaminate each other — one analyst sees another's tool calls,
    causing Vertex AI to reject the request due to mismatched function
    call/response counts.

    This helper runs the entire tool loop within a single node invocation:
    call LLM → execute tools → call LLM → ... until no more tool calls.
    Only the final AIMessage (with the report) is returned to the graph.

    Returns:
        The final AIMessage (no tool_calls) containing the analyst's report.
    """
    tools_by_name = {t.name: t for t in tools}
    local_messages = list(messages)

    for _ in range(max_iterations):
        result = chain.invoke(local_messages)
        if not result.tool_calls:
            return result

        # Add the AI message with tool calls to local history
        local_messages.append(result)

        # Execute each tool call and add responses to local history
        for tc in result.tool_calls:
            tool_name = tc["name"]
            tool_fn = tools_by_name.get(tool_name)
            if tool_fn is None:
                content = f"Error: {tool_name} is not a valid tool, try one of [{', '.join(tools_by_name.keys())}]."
            else:
                try:
                    content = tool_fn.invoke(tc["args"])
                except Exception as e:
                    content = f"Error executing {tool_name}: {e}"
            local_messages.append(
                ToolMessage(content=str(content), tool_call_id=tc["id"], name=tool_name)
            )

    # If we hit max iterations, return the last result regardless
    return result


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages
