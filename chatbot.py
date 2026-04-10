from __future__ import annotations

import json
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def build_llm_prompt(user_text, cart_items, recommendations):
    return f"""
You are an ecommerce shopping assistant.
Answer naturally and briefly.
Use the recommendation results to explain why the products fit the user.

User message:
{user_text}

Cart items:
{json.dumps(cart_items)}

Recommendations:
{json.dumps(recommendations, indent=2)}
""".strip()


def llm_reply(user_text, cart_items, recommendations):
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if not api_key or OpenAI is None:
        if recommendations:
            ids = [str(item["product_id"]) for item in recommendations[:3]]
            return f"Here are some strong matches for your cart: {', '.join(ids)}."
        return "I could not find strong matches yet. Try adding more items or refining your request."

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model_name,
        input=build_llm_prompt(user_text, cart_items, recommendations),
    )
    return response.output_text.strip()
