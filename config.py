import os

PRODUCTS_PATH = os.getenv("PRODUCTS_PATH", "products.json")
PURCHASES_PATH = os.getenv("PURCHASES_PATH", "purchases.csv")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

MIN_SUPPORT = float(os.getenv("MIN_SUPPORT", "0.02"))
MIN_LIFT = float(os.getenv("MIN_LIFT", "1.0"))
TOP_K = int(os.getenv("TOP_K", "10"))
MIN_FINAL_SCORE = float(os.getenv("MIN_FINAL_SCORE", "0.3"))

CATEGORY_BOOST = 0.3
STYLE_BOOST = 0.2
BRAND_BOOST = 0.1
INTENT_CATEGORY_BOOST = 0.25
INTENT_STYLE_BOOST = 0.25
INTENT_BRAND_BOOST = 0.15
BUDGET_BOOST = 0.15

SEMANTIC_WEIGHT = 1.0
ASSOCIATION_WEIGHT = 1.0

PORT = int(os.getenv("PORT", "5050"))
