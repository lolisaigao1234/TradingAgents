import json
import re


EVALUATOR_SYSTEM_PROMPT = """你是一个独立的评估者（Evaluator），你的唯一职责是验证其他分析师的输出质量。你不参与任何分析或投资决策。Generator-Evaluator 分离是核心原则。

## 评分维度（每项 0-10 分）

### 1. data_quality（数据质量）
- 数据来源是否明确、可验证？
- 引用的数字是否自洽（例如：同一份报告中同一指标不能出现矛盾数值）？
- **严禁**接受"某分析师认为"、"有观点认为"、"据分析"等模糊引用作为数据来源——这不是证据，这是偷懒
- 每一个关键论断必须有具体数据支撑（日期、数值、来源）
- 如果出现"某分析师认为"类表述作为论据支撑，该项直接 0 分
- 验证每个分析师的报告都包含：具体数据点、数据来源、量化分析
- 如果任何分析师的报告为空或内容模糊（< 50 字），data_quality 直接判 0 分
- 检查以下必要覆盖项：技术指标(RSI/MACD)、财报数据、新闻事件、市场情绪

### 2. reasoning_depth（推理深度）
- 必须完整包含以下推理链：数据 → 观察 → 假设 → 验证 → 结论
- 缺少任何一个环节，该项最高 5 分
- 仅有"数据→结论"的跳跃式推理，最高 3 分
- 推理链中的每一步必须逻辑连贯，不能出现逻辑跳跃
- "因为X所以Y"不算推理链，必须展示中间验证步骤

### 3. consistency（一致性）
- 技术面分析和基本面分析方向是否一致？
- 如果方向相反，是否提供了合理的解释？
- Bull 和 Bear 分析师的论据是否相互矛盾但都有数据支撑？
- 无解释的矛盾直接扣 4 分以上
- 同一报告内数据前后矛盾，直接扣 5 分

### 4. actionability（可操作性）
- 投资建议是否具体、明确？
- 是否包含具体的仓位建议、止损位、目标价？
- "建议观望"、"需要进一步观察"类的模糊建议最高 4 分
- 必须有明确的 entry/exit 条件

## 输出格式

你必须严格按照以下 JSON 格式输出，不要输出任何其他内容：

```json
{
  "data_quality": <0-10>,
  "reasoning_depth": <0-10>,
  "consistency": <0-10>,
  "actionability": <0-10>,
  "total": <0-40>,
  "pass": <true/false>,
  "reasoning": "<评分理由，简明扼要，指出具体缺陷>"
}
```

## 评判标准
- total >= 24：通过（pass: true）
- total < 24：打回重做（pass: false）
- 你的评分必须严格、客观，不能因为分析看起来"努力了"就给高分
- 打回时必须在 reasoning 中指出具体缺陷，给出改进方向"""


def create_evaluator_node(llm_client):
    """Create the evaluator node for the langgraph pipeline.

    The evaluator independently assesses analyst output quality without
    participating in any analysis. Generator-Evaluator separation is the
    core principle.

    Args:
        llm_client: The LLM client to use for evaluation.

    Returns:
        A langgraph node function.
    """

    def evaluator_node(state) -> dict:
        eval_retry_count = state.get("eval_retry_count", 0)

        # Fail-closed after max retries: emit a FAILED report so downstream
        # knows quality was poor.  Routing (force-proceed to Trader) is
        # handled by conditional_logic.should_retry_after_evaluation.
        if eval_retry_count >= 2:
            return {
                "evaluation_report": json.dumps(
                    {
                        "data_quality": 0,
                        "reasoning_depth": 0,
                        "consistency": 0,
                        "actionability": 0,
                        "total": 0,
                        "pass": False,
                        "reasoning": "max retries exceeded - analysis quality insufficient",
                    },
                    ensure_ascii=False,
                ),
                "eval_retry_count": eval_retry_count + 1,
            }

        # Gather analyst reports and debate history
        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")

        # Tag empty reports so the LLM evaluator sees which analysts are missing
        _empty_warn = "[WARNING: 此分析师未产出报告]"
        if not market_report.strip():
            market_report = _empty_warn
        if not sentiment_report.strip():
            sentiment_report = _empty_warn
        if not news_report.strip():
            news_report = _empty_warn
        if not fundamentals_report.strip():
            fundamentals_report = _empty_warn

        investment_debate_state = state.get("investment_debate_state", {})
        bull_history = investment_debate_state.get("bull_history", "")
        bear_history = investment_debate_state.get("bear_history", "")
        judge_decision = investment_debate_state.get("judge_decision", "")

        user_prompt = f"""请评估以下分析师输出的质量。

## 市场分析报告
{market_report}

## 社交媒体情绪报告
{sentiment_report}

## 新闻分析报告
{news_report}

## 基本面分析报告
{fundamentals_report}

## Bull 分析师论点
{bull_history}

## Bear 分析师论点
{bear_history}

## 研究经理决策
{judge_decision}

请严格按照评分标准打分，输出 JSON。注意：
1. "某分析师认为"不是证据，必须追溯到原始数据
2. 推理链必须完整：数据→观察→假设→验证→结论
3. 每个扣分点必须指出具体位置和原因"""

        messages = [
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = llm_client.invoke(messages)
        raw_content = response.content

        # Parse and validate - fail-closed on parse error
        report = _parse_and_validate(raw_content)

        return {
            "evaluation_report": json.dumps(report, ensure_ascii=False),
            "eval_retry_count": eval_retry_count + 1,
        }

    return evaluator_node


def create_retry_gate_node():
    """Create the Retry Gate node — a no-op node for conditional parallel fan-out.

    Resets investment_debate_state to defaults (count=0, clear histories)
    so the debate starts fresh on retry.
    """

    def retry_gate_node(state) -> dict:
        return {
            "investment_debate_state": {
                "bull_history": "",
                "bear_history": "",
                "history": "",
                "current_response": "",
                "judge_decision": "",
                "count": 0,
            },
        }

    return retry_gate_node


def _parse_and_validate(raw_content: str) -> dict:
    """Parse LLM output and apply server-side validation.

    Fail-closed: JSON parse failure = reject.
    Server-side: clamp scores 0-10, recompute total, derive pass from total >= 24.
    Bool coercion: handle LLM returning 'false' as string.
    """
    try:
        # Extract JSON from response (LLM may wrap in markdown code blocks)
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise ValueError("No JSON object found")
        parsed = json.loads(raw_content[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        # Fail-closed: parse failure = reject
        return {
            "data_quality": 0,
            "reasoning_depth": 0,
            "consistency": 0,
            "actionability": 0,
            "total": 0,
            "pass": False,
            "reasoning": f"JSON 解析失败，自动打回。原始输出: {raw_content[:500]}",
        }

    # Server-side score validation: clamp 0-10
    dimensions = ["data_quality", "reasoning_depth", "consistency", "actionability"]
    for dim in dimensions:
        val = parsed.get(dim, 0)
        if isinstance(val, bool):
            val = 0
        else:
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = 0
        parsed[dim] = max(0, min(10, val))

    # Recompute total server-side (don't trust LLM's arithmetic)
    parsed["total"] = sum(parsed[dim] for dim in dimensions)

    # Derive pass from total (don't trust LLM's boolean)
    # Bool coercion: LLM may return "false" as string
    parsed["pass"] = parsed["total"] >= 24

    # Ensure reasoning exists
    if "reasoning" not in parsed or not isinstance(parsed["reasoning"], str):
        parsed["reasoning"] = ""

    return parsed
