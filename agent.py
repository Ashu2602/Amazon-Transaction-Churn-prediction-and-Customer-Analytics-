"""
Claude Agent SDK wiring for the Amazon product RAG/recommendation assistant.

This module only defines tools and agent options -- it does not import
anything that hits the network until run(). Requires either:
  - ANTHROPIC_API_KEY set in the environment, or
  - a locally authenticated `claude` CLI (the Claude Agent SDK shells out to it)

Not runnable yet in this session (no key configured) -- see README "Part 2".
The tool functions themselves (recommend_tools.py) work standalone right now
and are exercised directly in test_recommend_tools.py without needing Claude
at all, so the retrieval/recommendation logic is already verified independent
of whether the agent wiring below has been test-run.
"""
import asyncio
from dotenv import load_dotenv
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    create_sdk_mcp_server,
    tool,
    AssistantMessage,
    TextBlock,
)
from recommend_tools import get_index

load_dotenv()  # loads ANTHROPIC_API_KEY from .env (gitignored) if not already in the environment


@tool(
    "search_products",
    "Semantic search over the Amazon product catalog by free-text description. "
    "Use this when the user describes what they want without naming a specific product.",
    {"query": str, "k": int},
)
async def search_products(args: dict) -> dict:
    idx = get_index()
    results = idx.search_products(args["query"], k=args.get("k", 10))
    return {"content": [{"type": "text", "text": results[["asin", "title", "category", "avg_price", "similarity"]].to_json(orient="records")}]}


@tool(
    "similar_products",
    "Find products similar to a specific known product, given its ASIN.",
    {"asin": str, "k": int},
)
async def similar_products(args: dict) -> dict:
    idx = get_index()
    results = idx.similar_to_asin(args["asin"], k=args.get("k", 10))
    return {"content": [{"type": "text", "text": results[["asin", "title", "category", "avg_price", "similarity"]].to_json(orient="records")}]}


@tool(
    "get_user_purchase_history",
    "Look up a user's recent Amazon purchase history by their Survey ResponseID.",
    {"response_id": str, "n": int},
)
async def get_user_purchase_history(args: dict) -> dict:
    idx = get_index()
    hist = idx.get_user_history(args["response_id"], n=args.get("n", 20))
    cols = ["Order Date", "Title", "Category", "Purchase Price Per Unit", "Quantity"]
    return {"content": [{"type": "text", "text": hist[cols].to_json(orient="records", date_format="iso")}]}


@tool(
    "recommend_for_user",
    "Generate personalized product recommendations for a user based on their "
    "purchase history, biased toward their more recent purchases.",
    {"response_id": str, "k": int},
)
async def recommend_for_user(args: dict) -> dict:
    idx = get_index()
    results = idx.recommend_for_user(args["response_id"], k=args.get("k", 10))
    return {"content": [{"type": "text", "text": results[["asin", "title", "category", "avg_price", "similarity"]].to_json(orient="records")}]}


@tool(
    "category_stats",
    "Get popularity and pricing stats for a product category (e.g. PET_FOOD).",
    {"category": str},
)
async def category_stats(args: dict) -> dict:
    idx = get_index()
    return {"content": [{"type": "text", "text": str(idx.category_stats(args["category"]))}]}


SYSTEM_PROMPT = """You are a shopping assistant with access to a real Amazon
purchase dataset (876K unique products, 5K users' purchase histories). Use the
provided tools to ground every product claim in actual retrieved data -- never
invent ASINs, prices, or purchase history. When recommending, prefer
recommend_for_user for a known user, and search_products for general requests."""


def build_options() -> ClaudeAgentOptions:
    server = create_sdk_mcp_server(
        name="amazon-catalog",
        tools=[search_products, similar_products, get_user_purchase_history, recommend_for_user, category_stats],
    )
    return ClaudeAgentOptions(
        mcp_servers={"amazon": server},
        allowed_tools=[
            "mcp__amazon__search_products",
            "mcp__amazon__similar_products",
            "mcp__amazon__get_user_purchase_history",
            "mcp__amazon__recommend_for_user",
            "mcp__amazon__category_stats",
        ],
        system_prompt=SYSTEM_PROMPT,
        model="claude-sonnet-5",  # cheaper than Opus, plenty capable for tool-calling over a fixed catalog
        max_budget_usd=0.50,      # hard per-session cap -- belt-and-suspenders alongside console-level spend limits
    )


async def ask(prompt: str) -> str:
    """Run one agent turn and return the assistant's final text reply."""
    options = build_options()
    reply_parts = []
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        reply_parts.append(block.text)
    return "".join(reply_parts)


if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) or "Recommend some pet food similar to what R_01vNIayewjIIKMF has bought before."
    print(asyncio.run(ask(prompt)))
