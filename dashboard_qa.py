"""
Natural-language question -> chart spec, using Claude tool use + structured
outputs. This is a plain single-call Anthropic API loop (not the Claude Agent
SDK) -- the Agent SDK shells out to the `claude` CLI, which is built for
interactive coding-agent sessions, not a stateless one-shot "answer this
question with a validated JSON chart spec" call from a web backend.

Design: give Claude one tool (query_table) that runs safe, parameterized
pandas aggregations over the four small in-memory tables -- never arbitrary
code execution -- then require the final answer to conform to a JSON schema
(chart_type/title/labels/values/insight) via output_config.format, so the
Streamlit app can render it without guessing at Claude's output shape.
"""
import json
import os

import anthropic
import pandas as pd

from dashboard_data import TABLE_DESCRIPTIONS

MODEL = "claude-sonnet-5"

CHART_SCHEMA = {
    "type": "object",
    "properties": {
        "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "number", "table"]},
        "title": {"type": "string"},
        "x_label": {"type": "string"},
        "y_label": {"type": "string"},
        "categories": {"type": "array", "items": {"type": "string"}},
        "values": {"type": "array", "items": {"type": "number"}},
        "insight": {"type": "string", "description": "1-3 sentence plain-language answer to the user's question."},
    },
    "required": ["chart_type", "title", "x_label", "y_label", "categories", "values", "insight"],
    "additionalProperties": False,
}

QUERY_TOOL = {
    "name": "query_table",
    "description": "Run a grouped aggregation (or a filtered scalar aggregation) over one of the dashboard's data tables.",
    "input_schema": {
        "type": "object",
        "properties": {
            "table": {"type": "string", "enum": ["category", "state", "monthly", "churn_users"]},
            "groupby": {"type": ["string", "null"], "description": "Column to group by, or null for a single scalar result."},
            "metric": {"type": "string", "description": "Column to aggregate. Ignored when agg is 'count'."},
            "agg": {"type": "string", "enum": ["sum", "mean", "count"]},
            "filter_column": {"type": ["string", "null"]},
            "filter_value": {"type": ["string", "number", "null"]},
            "sort_desc": {"type": "boolean"},
            "top_n": {"type": "integer"},
        },
        "required": ["table", "groupby", "metric", "agg", "filter_column", "filter_value", "sort_desc", "top_n"],
    },
}

SYSTEM_PROMPT = f"""You answer questions about Amazon customer purchase/churn
data by calling the query_table tool (never invent numbers), then returning a
chart spec. {TABLE_DESCRIPTIONS}

Pick chart_type "number" for a single scalar answer, "table" only when a list
of category/value pairs doesn't read well as a bar/pie/line, otherwise prefer
"bar" for comparisons across categories and "line" for the monthly trend.
categories and values must be the same length. Keep insight concise and
specific -- cite the actual numbers you found."""


def _run_query(tables: dict[str, pd.DataFrame], **kwargs) -> str:
    table = kwargs["table"]
    df = tables[table]
    filter_column = kwargs.get("filter_column")
    filter_value = kwargs.get("filter_value")
    groupby = kwargs.get("groupby")
    metric = kwargs.get("metric")
    agg = kwargs.get("agg", "sum")
    sort_desc = kwargs.get("sort_desc", True)
    top_n = kwargs.get("top_n", 10)

    try:
        if filter_column:
            if filter_column not in df.columns:
                return json.dumps({"error": f"unknown column '{filter_column}' in table '{table}'"})
            df = df[df[filter_column].astype(str) == str(filter_value)]

        if groupby:
            if groupby not in df.columns:
                return json.dumps({"error": f"unknown column '{groupby}' in table '{table}'"})
            if agg == "count":
                result = df.groupby(groupby).size().reset_index(name="count")
                value_col = "count"
            else:
                if metric not in df.columns:
                    return json.dumps({"error": f"unknown column '{metric}' in table '{table}'"})
                result = df.groupby(groupby)[metric].agg(agg).reset_index()
                value_col = metric
            result = result.sort_values(value_col, ascending=not sort_desc).head(top_n)
            return result.to_json(orient="records")
        else:
            if agg == "count":
                return json.dumps({"count": int(len(df))})
            if metric not in df.columns:
                return json.dumps({"error": f"unknown column '{metric}' in table '{table}'"})
            return json.dumps({agg: float(df[metric].agg(agg))})
    except Exception as e:  # tool errors go back to Claude as text, not a crash
        return json.dumps({"error": str(e)})


def answer_question(question: str, tables: dict[str, pd.DataFrame]) -> dict:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]

    for _ in range(5):  # hard cap on tool round-trips
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[QUERY_TOOL],
            output_config={"format": {"type": "json_schema", "schema": CHART_SCHEMA}},
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result_text = _run_query(tables, **block.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_text})
            messages.append({"role": "user", "content": tool_results})
            continue

        text = next((b.text for b in response.content if b.type == "text"), "{}")
        return json.loads(text)

    return {
        "chart_type": "number", "title": "Error", "x_label": "", "y_label": "",
        "categories": [], "values": [], "insight": "Could not resolve the question in time.",
    }
