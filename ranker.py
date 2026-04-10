from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    ASSOCIATION_WEIGHT,
    BRAND_BOOST,
    BUDGET_BOOST,
    CATEGORY_BOOST,
    INTENT_BRAND_BOOST,
    INTENT_CATEGORY_BOOST,
    INTENT_STYLE_BOOST,
    MIN_FINAL_SCORE,
    SEMANTIC_WEIGHT,
    STYLE_BOOST,
    TOP_K,
)
from intent import extract_intent


def normalize_scores(values: dict[Any, float]) -> dict[Any, float]:
    if not values:
        return {}

    arr = np.array(list(values.values()), dtype=float)
    min_val = arr.min()
    max_val = arr.max()

    if np.isclose(min_val, max_val):
        return {k: 1.0 for k in values}

    return {
        k: float((v - min_val) / (max_val - min_val))
        for k, v in values.items()
    }


def get_product_row(state, product_id: Any):
    idx = state.id_to_idx.get(product_id)
    if idx is None:
        return None
    return state.products.iloc[idx]


def coerce_cart_items(cart_items: list[Any]) -> list[int]:
    normalized: list[int] = []
    for item in cart_items:
        try:
            normalized.append(int(item))
        except (TypeError, ValueError):
            continue
    return normalized


def build_cart_profile(state, cart_items: list[Any]) -> dict[str, set[str]]:
    categories: set[str] = set()
    styles: set[str] = set()
    brands: set[str] = set()

    for pid in cart_items:
        row = get_product_row(state, pid)
        if row is None:
            continue

        if row["category"]:
            categories.add(str(row["category"]))
        if row["style"]:
            styles.add(str(row["style"]))
        if row["brand"]:
            brands.add(str(row["brand"]))

    return {
        "categories": categories,
        "styles": styles,
        "brands": brands,
    }


def association_candidates(state, cart_items: list[Any]) -> dict[Any, float]:
    if state.rules.empty or not cart_items:
        return {}

    cart_set = set(cart_items)
    raw_scores: dict[Any, float] = {}

    for _, rule in state.rules.iterrows():
        antecedents = set(rule["antecedents"])
        consequents = set(rule["consequents"])

        if antecedents.issubset(cart_set):
            score = float(rule["lift"]) + float(rule["confidence"])
            for pid in consequents:
                if pid not in cart_set:
                    raw_scores[pid] = max(raw_scores.get(pid, 0.0), score)

    return normalize_scores(raw_scores)


def semantic_candidates(
    state,
    cart_items: list[Any],
    top_k: int = TOP_K * 3,
) -> dict[Any, float]:
    if not cart_items:
        return {}

    scores: dict[Any, float] = {}

    for pid in cart_items:
        idx = state.id_to_idx.get(pid)
        if idx is None:
            continue

        query_vec = state.embeddings[idx].reshape(1, -1)
        sims = cosine_similarity(query_vec, state.embeddings).flatten()
        ranked_indices = np.argsort(sims)[::-1]

        taken = 0
        for candidate_idx in ranked_indices:
            candidate_id = state.products.iloc[candidate_idx]["product_id"]

            if candidate_id == pid or candidate_id in cart_items:
                continue

            scores[candidate_id] = max(
                scores.get(candidate_id, 0.0),
                float(sims[candidate_idx]),
            )
            taken += 1
            if taken >= top_k:
                break

    return scores


def structured_filter(state, candidate_id: Any, intent=None) -> bool:
    row = get_product_row(state, candidate_id)

    if row is None:
        return False

    # ✅ Keep only relevant product types
    allowed_styles = ["cutlery", "cookware", "homekeeping"]

    return row["style"] in allowed_styles

def structured_boost(state, candidate_id: Any, cart_profile, intent=None):
    row = get_product_row(state, candidate_id)
    if row is None:
        return 0.0, {}

    same_category = str(row["category"]) in cart_profile["categories"]
    same_style = str(row["style"]) in cart_profile["styles"]
    same_brand = str(row["brand"]) in cart_profile["brands"]

    boost = 0.0
    if same_category:
        boost += CATEGORY_BOOST
    if same_style:
        boost += STYLE_BOOST
    if same_brand:
        boost += BRAND_BOOST

    if intent:
        if intent.category and str(row["category"]) == intent.category:
            boost += INTENT_CATEGORY_BOOST
        if intent.style and str(row["style"]) == intent.style:
            boost += INTENT_STYLE_BOOST
        if intent.brand and str(row["brand"]) == intent.brand:
            boost += INTENT_BRAND_BOOST
        if intent.max_price is not None and float(row["price"]) <= intent.max_price:
            boost += BUDGET_BOOST
        elif intent.budget_type == "cheap":
            if float(row["price"]) <= float(state.products["price"].median()):
                boost += BUDGET_BOOST
        elif intent.budget_type == "premium":
            if float(row["price"]) >= float(state.products["price"].median()):
                boost += BUDGET_BOOST

    return boost, {
        "same_category": same_category,
        "same_style": same_style,
        "same_brand": same_brand,
        "matches_intent_category": bool(
            intent and intent.category and str(row["category"]) == intent.category
        ),
        "matches_intent_style": bool(
            intent and intent.style and str(row["style"]) == intent.style
        ),
        "matches_intent_brand": bool(
            intent and intent.brand and str(row["brand"]) == intent.brand
        ),
    }


def popular_fallback(state, cart_items: list[Any], top_k: int):
    items = []

    for pid in state.popular_products:
        if pid in cart_items:
            continue

        row = get_product_row(state, pid)
        if row is None:
            continue

        items.append(
            {
                "product_id": int(pid),
                "category": str(row["category"]),
                "style": str(row["style"]),
                "brand": str(row["brand"]),
                "price": float(row.get("price", 0.0)),
                "semantic_score": 0.0,
                "association_score": 0.0,
                "boost_score": 0.0,
                "final_score": 0.0,
                "fallback": "popular_products",
            }
        )

        if len(items) >= top_k:
            break

    return items


def recommend_items(
    state,
    cart_items: list[Any],
    top_k: int = TOP_K,
    user_text: str = "",
):
    cart_items = coerce_cart_items(cart_items)

    intent = extract_intent(
        user_text,
        state.category_values,
        state.style_values,
        state.brand_values,
    )

    cart_profile = build_cart_profile(state, cart_items)
    assoc_scores = association_candidates(state, cart_items)
    semantic_scores = semantic_candidates(state, cart_items)

    candidate_ids = set(assoc_scores) | set(semantic_scores)

    ranked = []

    for pid in candidate_ids:
        if pid in cart_items:
            continue

        if not structured_filter(state, pid, intent):
            continue

        semantic_score = semantic_scores.get(pid, 0.0)
        association_score = assoc_scores.get(pid, 0.0)
        boost_score, flags = structured_boost(state, pid, cart_profile, intent)

        # ✅ CORRECT SCORING
        final_score = 0.0

        if association_score > 0:
            final_score += 3.0 * association_score

        final_score += boost_score
        final_score += 0.3 * semantic_score

        if final_score < MIN_FINAL_SCORE:
            continue

        row = get_product_row(state, pid)
        if row is None:
            continue

        ranked.append(
            {
                "product_id": int(pid),
                "category": str(row["category"]),
                "style": str(row["style"]),
                "brand": str(row["brand"]),
                "price": float(row.get("price", 0.0)),
                "semantic_score": round(semantic_score, 4),
                "association_score": round(association_score, 4),
                "boost_score": round(boost_score, 4),
                "final_score": round(final_score, 4),
                "explanations": {
                    **flags,
                    "has_behavior_match": association_score > 0,
                    "has_semantic_match": semantic_score > 0,
                },
            }
        )

    # ✅ SORT OUTSIDE LOOP
    ranked.sort(key=lambda x: x["final_score"], reverse=True)

    # ✅ FALLBACK
    if not ranked:
        return popular_fallback(state, cart_items, top_k)

    return ranked[:top_k]
def build_bundle(
    state,
    cart_items: list[Any],
    bundle_size: int = 5,
    user_text: str = "",
):
    recommendations = recommend_items(
        state,
        cart_items,
        top_k=TOP_K * 2,
        user_text=user_text,
    )

    bundle = []
    used_categories = set()

    for item in recommendations:
        if item["category"] in used_categories:
            continue

        used_categories.add(item["category"])
        bundle.append(item)

        if len(bundle) >= bundle_size:
            break

    return bundle