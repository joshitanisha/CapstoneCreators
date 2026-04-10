from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class UserIntent:
    category: str | None = None
    style: str | None = None
    brand: str | None = None
    budget_type: str | None = None
    max_price: float | None = None


def extract_intent(
    text: str,
    category_values: set[str],
    style_values: set[str],
    brand_values: set[str],
) -> UserIntent:
    text = (text or "").lower().strip()
    intent = UserIntent()

    price_match = re.search(r"(under|below|less than)\s+(\d+)", text)
    if price_match:
        intent.max_price = float(price_match.group(2))

    if any(word in text for word in ["cheap", "budget", "affordable", "low price"]):
        intent.budget_type = "cheap"
    elif any(word in text for word in ["premium", "luxury", "high-end", "expensive"]):
        intent.budget_type = "premium"

    for value in category_values:
        if value and value in text:
            intent.category = value
            break

    for value in style_values:
        if value and value in text:
            intent.style = value
            break

    for value in brand_values:
        if value and value in text:
            intent.brand = value
            break

    return intent
