import json
import os
import re
import secrets
from datetime import datetime, timezone

from utils.search_utils import (
    flatten_policy,
    load_json,
    normalize_query,
    parse_query,
    rank_results,
    score_field,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
CATALOG_PATH = os.path.join(DATA_DIR, "product_catalog.json")
FAQ_PATH = os.path.join(DATA_DIR, "product_faqs.json")
POLICY_PATH = os.path.join(DATA_DIR, "company_policy.json")
ORDERS_PATH = os.path.join(DATA_DIR, "order_database.json")


class ProductCatalogTool:
    def __init__(self, path: str = CATALOG_PATH):
        self.products = load_json(path, [])

    def search(self, query: str, k: int = 5):
        parsed = parse_query(query)
        cleaned_query = normalize_query(parsed.get("cleaned_query", query))
        if not cleaned_query:
            return []

        scored = []
        for product in self.products:
            score = 0
            score += score_field(cleaned_query, product.get("product_name", ""), 3)
            score += score_field(cleaned_query, product.get("product_id", ""), 2)
            score += score_field(cleaned_query, product.get("category", ""), 2)
            score += score_field(cleaned_query, product.get("description", ""), 1)
            if score > 0:
                scored.append({"score": score, "item": product})

        results = rank_results(scored, k)
        max_price = parsed.get("max_price")
        min_price = parsed.get("min_price")
        min_rating = parsed.get("min_rating")
        category = parsed.get("category")

        if max_price is not None:
            results = [r for r in results if r.get("price") is not None and r["price"] <= max_price]
        if min_price is not None:
            results = [r for r in results if r.get("price") is not None and r["price"] >= min_price]
        if min_rating is not None:
            results = [r for r in results if r.get("rating") is not None and r["rating"] >= min_rating]
        if category:
            results = [
                r
                for r in results
                if r.get("category") and r["category"].lower() == category
            ]

        sort_hint = parsed.get("sort")
        if sort_hint == "price_asc":
            results = sorted(results, key=lambda r: r.get("price", 0))
        elif sort_hint == "delivery_time_days_asc":
            results = sorted(results, key=lambda r: r.get("delivery_time_days", 0))

        return results

    def retrieve(self, query: str, k: int = 5):
        results = self.search(query=query, k=k)
        if not results:
            return "No matching products found."
        return json.dumps(results, indent=2)


class ProductFAQTool:
    def __init__(self, path: str = FAQ_PATH):
        self.faqs = load_json(path, [])

    def search(self, query: str, k: int = 5):
        query = normalize_query(query)
        if not query:
            return []

        scored = []
        for product in self.faqs:
            product_name = product.get("product_name", "")
            product_id = product.get("product_id", "")
            for faq in product.get("faqs", []):
                score = 0
                score += score_field(query, product_name, 2)
                score += score_field(query, product_id, 2)
                score += score_field(query, faq.get("question", ""), 3)
                score += score_field(query, faq.get("answer", ""), 1)
                if score > 0:
                    scored.append(
                        {
                            "score": score,
                            "item": {
                                "product_id": product_id,
                                "product_name": product_name,
                                "question": faq.get("question", ""),
                                "answer": faq.get("answer", ""),
                            },
                        }
                    )

        return rank_results(scored, k)

    def retrieve(self, query: str, k: int = 5):
        results = self.search(query=query, k=k)
        if not results:
            return "No matching FAQs found."
        return json.dumps(results, indent=2)


class CompanyPolicyTool:
    def __init__(self, path: str = POLICY_PATH):
        self.policies = load_json(path, {}).get("policy_document", {})
        self.entries = flatten_policy(self.policies)

    def search(self, query: str, k: int = 5):
        query = normalize_query(query)
        if not query:
            return []

        scored = []
        for entry in self.entries:
            score = 0
            score += score_field(query, entry.get("section", ""), 2)
            score += score_field(query, entry.get("text", ""), 3)
            if score > 0:
                scored.append({"score": score, "item": entry})

        return rank_results(scored, k)

    def retrieve(self, query: str, k: int = 5):
        results = self.search(query=query, k=k)
        if not results:
            return "No matching policy entries found."
        return json.dumps(results, indent=2)


class OrderDatabaseTool:
    def __init__(self, path: str = ORDERS_PATH):
        self.orders = load_json(path, [])

    def search(self, query: str, k: int = 5):
        query = normalize_query(query)
        if not query:
            return []

        if re.fullmatch(r"c\d{4}", query):
            exact = [
                order
                for order in self.orders
                if normalize_query(order.get("customer_id", "")) == query
            ]
            return exact[: max(1, int(k))]

        scored = []
        for order in self.orders:
            score = 0
            score += score_field(query, order.get("order_id", ""), 3)
            score += score_field(query, order.get("customer_id", ""), 2)
            score += score_field(query, order.get("order_status", ""), 2)
            score += score_field(query, order.get("order_date", ""), 1)
            for product in order.get("products", []):
                score += score_field(query, product.get("product_name", ""), 2)
                score += score_field(query, product.get("product_id", ""), 2)
            if score > 0:
                scored.append({"score": score, "item": order})

        return rank_results(scored, k)

    def retrieve(self, query: str, k: int = 5):
        results = self.search(query=query, k=k)
        if not results:
            return "No matching orders found."
        return json.dumps(results, indent=2)


class KnowledgeBaseTools:
    def __init__(self, customer_id: str | None = None):
        self.catalog = ProductCatalogTool()
        self.faq = ProductFAQTool()
        self.policy = CompanyPolicyTool()
        self.orders = OrderDatabaseTool()
        self.customer_id = customer_id

    def _resolve_customer_id(self, customer_id: str | None):
        return customer_id or self.customer_id

    def _log_action(
        self,
        action: str,
        customer_id: str | None,
        order_id: str,
        product_id: str | None,
        result: str,
        message: str,
        ticket_id: str | None,
    ):
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "customer_id": customer_id,
            "order_id": order_id,
            "product_id": product_id,
            "result": result,
            "reason": message,
            "ticket_id": ticket_id,
        }
        log_path = os.path.join(DATA_DIR, "action_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(payload) + "\n")

    def _find_order(self, order_id: str):
        for order in self.orders.orders:
            if order.get("order_id") == order_id:
                return order
        return None

    def cancel_order(self, order_id: str, customer_id: str | None = None) -> str:
        customer_id = self._resolve_customer_id(customer_id)
        order = self._find_order(order_id)
        if not order:
            message = "Order not found."
            self._log_action(
                action="cancel_order",
                customer_id=customer_id,
                order_id=order_id,
                product_id=None,
                result="rejected",
                message=message,
                ticket_id=None,
            )
            return json.dumps(
                {"status": "rejected", "message": message, "ticket_id": None},
                indent=2,
            )

        allowed = {"Placed", "Shipped"}
        if order.get("order_status") not in allowed:
            message = "Order is not eligible for cancellation."
            self._log_action(
                action="cancel_order",
                customer_id=customer_id,
                order_id=order_id,
                product_id=None,
                result="rejected",
                message=message,
                ticket_id=None,
            )
            return json.dumps(
                {"status": "rejected", "message": message, "ticket_id": None},
                indent=2,
            )

        ticket_id = f"CAN-{secrets.token_hex(3)}"
        message = "Cancellation request submitted."
        self._log_action(
            action="cancel_order",
            customer_id=customer_id,
            order_id=order_id,
            product_id=None,
            result="approved",
            message=message,
            ticket_id=ticket_id,
        )
        return json.dumps(
            {"status": "approved", "message": message, "ticket_id": ticket_id},
            indent=2,
        )

    def initiate_return(
        self, order_id: str, product_id: str, customer_id: str | None = None
    ) -> str:
        customer_id = self._resolve_customer_id(customer_id)
        order = self._find_order(order_id)
        if not order:
            message = "Order not found."
            self._log_action(
                action="initiate_return",
                customer_id=customer_id,
                order_id=order_id,
                product_id=product_id,
                result="rejected",
                message=message,
                ticket_id=None,
            )
            return json.dumps(
                {"status": "rejected", "message": message, "ticket_id": None},
                indent=2,
            )

        if order.get("order_status") != "Delivered":
            message = "Order is not eligible for return."
            self._log_action(
                action="initiate_return",
                customer_id=customer_id,
                order_id=order_id,
                product_id=product_id,
                result="rejected",
                message=message,
                ticket_id=None,
            )
            return json.dumps(
                {"status": "rejected", "message": message, "ticket_id": None},
                indent=2,
            )

        products = order.get("products", [])
        if not any(p.get("product_id") == product_id for p in products):
            message = "Product not found in order."
            self._log_action(
                action="initiate_return",
                customer_id=customer_id,
                order_id=order_id,
                product_id=product_id,
                result="rejected",
                message=message,
                ticket_id=None,
            )
            return json.dumps(
                {"status": "rejected", "message": message, "ticket_id": None},
                indent=2,
            )

        ticket_id = f"RET-{secrets.token_hex(3)}"
        message = "Return request submitted."
        self._log_action(
            action="initiate_return",
            customer_id=customer_id,
            order_id=order_id,
            product_id=product_id,
            result="approved",
            message=message,
            ticket_id=ticket_id,
        )
        return json.dumps(
            {"status": "approved", "message": message, "ticket_id": ticket_id},
            indent=2,
        )

    def retrieve(self, query: str, mode: str = "catalog", k: int = 5):
        mode = (mode or "catalog").lower().strip()
        if mode in {"catalog", "products"}:
            return self.catalog.retrieve(query=query, k=k)
        if mode in {"faq", "faqs"}:
            return self.faq.retrieve(query=query, k=k)
        if mode in {"policy", "company_policy"}:
            return self.policy.retrieve(query=query, k=k)
        if mode in {"orders", "order"}:
            normalized = normalize_query(query)
            if normalized in {"orders", "order", "my orders", "order history", "previous orders"}:
                query = self.customer_id or query
            return self.orders.retrieve(query=query, k=k)
        if mode in {"catalog+faq"}:
            combined = {
                "catalog": self.catalog.search(query=query, k=k),
                "faq": self.faq.search(query=query, k=k),
            }
            return json.dumps(combined, indent=2)
        return f"Unknown mode '{mode}'."


class CatalogSearch(ProductCatalogTool):
    pass
