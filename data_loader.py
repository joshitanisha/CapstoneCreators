from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    MIN_LIFT,
    MIN_SUPPORT,
    PRODUCTS_PATH,
    PURCHASES_PATH,
)


@dataclass
class RecommenderState:
    products: pd.DataFrame
    purchases: pd.DataFrame
    rules: pd.DataFrame
    embeddings: np.ndarray
    model: SentenceTransformer
    id_to_idx: dict[Any, int]
    popular_products: list[Any]
    category_values: set[str]
    style_values: set[str]
    brand_values: set[str]


def load_products(path: str = PRODUCTS_PATH) -> pd.DataFrame:
    if path.endswith(".json"):
        products = pd.read_json(path)
    else:
        products = pd.read_csv(path)

    required = {"product_id", "category", "style", "brand", "description"}
    missing = required - set(products.columns)
    if missing:
        raise ValueError(f"Missing required product columns: {sorted(missing)}")

    if "price" not in products.columns:
        products["price"] = 0.0

    products["product_id"] = pd.to_numeric(
        products["product_id"], errors="coerce"
    ).astype("Int64")
    products = products.dropna(subset=["product_id"]).copy()
    products["product_id"] = products["product_id"].astype(int)

    products = products.drop_duplicates(subset=["product_id"]).copy()
    products = products.fillna("")

    for col in ["category", "style", "brand", "description"]:
        products[col] = products[col].astype(str).str.strip().str.lower()

    products["price"] = pd.to_numeric(products["price"], errors="coerce").fillna(0.0)
    return products.reset_index(drop=True)


def load_purchases(path: str = PURCHASES_PATH) -> pd.DataFrame:
    purchases = pd.read_csv(path)

    required = {"order_id", "product_id"}
    missing = required - set(purchases.columns)
    if missing:
        raise ValueError(f"Missing required purchase columns: {sorted(missing)}")

    purchases["product_id"] = pd.to_numeric(
        purchases["product_id"], errors="coerce"
    ).astype("Int64")
    purchases = purchases.dropna(subset=["order_id", "product_id"]).copy()
    purchases["product_id"] = purchases["product_id"].astype(int)

    return purchases.reset_index(drop=True)


def build_rules(purchases: pd.DataFrame) -> pd.DataFrame:
    basket = (
        purchases.assign(value=1)
        .pivot_table(
            index="order_id",
            columns="product_id",
            values="value",
            aggfunc="max",
            fill_value=0,
        )
        .astype(bool)
    )

    if basket.empty:
        return pd.DataFrame()

    freq = apriori(basket, min_support=MIN_SUPPORT, use_colnames=True)
    if freq.empty:
        return pd.DataFrame()

    rules = association_rules(freq, metric="lift", min_threshold=MIN_LIFT)
    if rules.empty:
        return pd.DataFrame()

    return rules.sort_values(
        by=["lift", "confidence", "support"],
        ascending=False,
    ).reset_index(drop=True)


def build_semantic_model(
    products: pd.DataFrame,
) -> tuple[SentenceTransformer, np.ndarray, dict[Any, int]]:
    descriptions = products["description"].fillna("").astype(str).tolist()

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        descriptions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    id_to_idx = {
        pid: idx for idx, pid in enumerate(products["product_id"].tolist())
    }
    return model, embeddings, id_to_idx


def build_popular_products(purchases: pd.DataFrame) -> list[Any]:
    return purchases["product_id"].value_counts().index.tolist()


def initialize_state() -> RecommenderState:
    products = load_products()
    purchases = load_purchases()

    product_ids = set(products["product_id"].tolist())
    purchase_ids = set(purchases["product_id"].tolist())
    unmatched = sorted(purchase_ids - product_ids)
    if unmatched:
        raise ValueError(
            f"purchases.csv contains product_ids not found in products file: {unmatched[:10]}"
        )

    rules = build_rules(purchases)
    model, embeddings, id_to_idx = build_semantic_model(products)

    return RecommenderState(
        products=products,
        purchases=purchases,
        rules=rules,
        embeddings=embeddings,
        model=model,
        id_to_idx=id_to_idx,
        popular_products=build_popular_products(purchases),
        category_values=set(products["category"].dropna().astype(str)),
        style_values=set(products["style"].dropna().astype(str)),
        brand_values=set(products["brand"].dropna().astype(str)),
    )
