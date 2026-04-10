from __future__ import annotations

from flask import Flask, jsonify, request

from chatbot import llm_reply
from config import EMBEDDING_MODEL, PORT, TOP_K
from data_loader import initialize_state
from intent import extract_intent
from ranker import build_bundle, recommend_items


app = Flask(__name__)
STATE = initialize_state()


def parse_json_body():
    return request.get_json(silent=True) or {}


def validate_cart(cart):
    if cart is None:
        return []
    if not isinstance(cart, list):
        raise ValueError("cart_items must be a list")
    return cart


def abandonment_score(cart_size: int, dwell_time: float) -> float:
    score = 0.2
    if cart_size <= 1:
        score += 0.35
    if dwell_time < 10:
        score += 0.25
    if cart_size == 0:
        score += 0.2
    return min(score, 1.0)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "products": len(STATE.products),
            "purchases": len(STATE.purchases),
            "rules": len(STATE.rules),
            "popular_products": len(STATE.popular_products),
            "embedding_model": EMBEDDING_MODEL,
        }
    )


@app.route("/recommend", methods=["POST"])
def recommend():
    data = parse_json_body()
    try:
        cart_items = validate_cart(data.get("cart_items", []))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    user_text = str(data.get("text", "")).strip()
    top_k = int(data.get("top_k", TOP_K))
    items = recommend_items(STATE, cart_items, top_k=top_k, user_text=user_text)
    return jsonify({"items": items})


@app.route("/bundle", methods=["POST"])
def bundle():
    data = parse_json_body()
    try:
        cart_items = validate_cart(data.get("cart_items", []))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    user_text = str(data.get("text", "")).strip()
    bundle_size = int(data.get("bundle_size", 5))
    items = build_bundle(STATE, cart_items, bundle_size=bundle_size, user_text=user_text)
    return jsonify({"items": items})


@app.route("/abandonment", methods=["POST"])
def abandonment():
    data = parse_json_body()
    try:
        cart_size = int(data.get("cart_size", 0))
        dwell_time = float(data.get("dwell_time", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "cart_size must be int and dwell_time must be numeric"}), 400

    return jsonify({"risk": abandonment_score(cart_size, dwell_time)})


@app.route("/chat", methods=["POST"])
def chat():
    data = parse_json_body()
    try:
        cart_items = validate_cart(data.get("cart_items", []))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    user_text = str(data.get("text", "")).strip()
    top_k = int(data.get("top_k", TOP_K))

    recommendations = recommend_items(STATE, cart_items, top_k=top_k, user_text=user_text)
    intent = extract_intent(
        user_text,
        STATE.category_values,
        STATE.style_values,
        STATE.brand_values,
    )
    reply = llm_reply(user_text, cart_items, recommendations)

    return jsonify(
        {
            "reply": reply,
            "items": recommendations,
            "intent": intent.__dict__,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5051, debug=False)


