import argparse
from pathlib import Path

from utils.search_utils import (
    EmbeddingConfig,
    Embedder,
    build_faiss_index,
    flatten_policy,
    load_json,
)


def _join_parts(parts):
    return " ".join(str(part).strip() for part in parts if part)


def build_catalog(products, embedder, out_dir: Path):
    texts = []
    items = []
    for product in products:
        text = _join_parts(
            [
                product.get("product_name"),
                product.get("category"),
                product.get("description"),
                (
                    f"Product ID {product.get('product_id')}"
                    if product.get("product_id")
                    else None
                ),
                (
                    f"Price {product.get('price')}"
                    if product.get("price") is not None
                    else None
                ),
                (
                    f"Rating {product.get('rating')}"
                    if product.get("rating") is not None
                    else None
                ),
            ]
        )
        if not text:
            continue
        texts.append(text)
        items.append(product)
    build_faiss_index(
        texts,
        items,
        embedder,
        out_dir / "catalog.faiss",
        out_dir / "catalog_meta.json",
    )


def build_faqs(faqs, embedder, out_dir: Path):
    texts = []
    items = []
    for entry in faqs:
        product_name = entry.get("product_name")
        product_id = entry.get("product_id")
        for faq in entry.get("faqs", []):
            text = _join_parts(
                [
                    product_name,
                    product_id,
                    faq.get("question"),
                    faq.get("answer"),
                ]
            )
            if not text:
                continue
            texts.append(text)
            items.append(
                {
                    "product_id": product_id,
                    "product_name": product_name,
                    "question": faq.get("question", ""),
                    "answer": faq.get("answer", ""),
                }
            )
    build_faiss_index(
        texts,
        items,
        embedder,
        out_dir / "faq.faiss",
        out_dir / "faq_meta.json",
    )


def build_policy(policy_doc, embedder, out_dir: Path):
    texts = []
    items = []
    for entry in flatten_policy(policy_doc):
        text = _join_parts([entry.get("section"), entry.get("text")])
        if not text:
            continue
        texts.append(text)
        items.append(entry)
    build_faiss_index(
        texts,
        items,
        embedder,
        out_dir / "policy.faiss",
        out_dir / "policy_meta.json",
    )


def build_orders(orders, embedder, out_dir: Path):
    texts = []
    items = []
    for order in orders:
        products = ", ".join(
            _join_parts(
                [
                    p.get("product_name"),
                    f"({p.get('product_id')})" if p.get("product_id") else None,
                ]
            )
            for p in order.get("products", [])
        )
        text = _join_parts(
            [
                f"Order {order.get('order_id')}" if order.get("order_id") else None,
                (
                    f"Customer {order.get('customer_id')}"
                    if order.get("customer_id")
                    else None
                ),
                (
                    f"Status {order.get('order_status')}"
                    if order.get("order_status")
                    else None
                ),
                f"Date {order.get('order_date')}" if order.get("order_date") else None,
                f"Products {products}" if products else None,
            ]
        )
        if not text:
            continue
        texts.append(text)
        items.append(order)
    build_faiss_index(
        texts,
        items,
        embedder,
        out_dir / "orders.faiss",
        out_dir / "orders_meta.json",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS indexes from data JSON files."
    )
    parser.add_argument("--data-dir", default=None, help="Path to data directory")
    parser.add_argument(
        "--out-dir", default=None, help="Path to output indexes directory"
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name",
    )
    parser.add_argument("--device", default=None, help="Embedding device override")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir / "data"
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "indexes"

    embedder = Embedder(EmbeddingConfig(model_name=args.model, device=args.device))

    catalog = load_json(str(data_dir / "product_catalog.json"), [])
    faqs = load_json(str(data_dir / "product_faqs.json"), [])
    policy = load_json(str(data_dir / "company_policy.json"), {}).get(
        "policy_document", {}
    )
    orders = load_json(str(data_dir / "order_database.json"), [])

    build_catalog(catalog, embedder, out_dir)
    build_faqs(faqs, embedder, out_dir)
    build_policy(policy, embedder, out_dir)
    build_orders(orders, embedder, out_dir)

    print(f"Indexes written to {out_dir}")


if __name__ == "__main__":
    main()
