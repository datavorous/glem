import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

from utils.search_utils import (
    EmbeddingConfig,
    Embedder,
    VectorIndex,
    load_json,
    normalize_query,
    parse_query,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".." / "data"
INDEX_DIR = DATA_DIR / "indexes"

CATALOG_PATH = DATA_DIR / "product_catalog.json"
FAQ_PATH = DATA_DIR / "product_faqs.json"
POLICY_PATH = DATA_DIR / "company_policy.json"
ORDERS_PATH = DATA_DIR / "order_database.json"

CATALOG_INDEX = INDEX_DIR / "catalog.faiss"
CATALOG_META = INDEX_DIR / "catalog_meta.json"
FAQ_INDEX = INDEX_DIR / "faq.faiss"
FAQ_META = INDEX_DIR / "faq_meta.json"
POLICY_INDEX = INDEX_DIR / "policy.faiss"
POLICY_META = INDEX_DIR / "policy_meta.json"
ORDERS_INDEX = INDEX_DIR / "orders.faiss"
ORDERS_META = INDEX_DIR / "orders_meta.json"


def _clamp_k(value, default: int = 5) -> int:
    try:
        k = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(20, k))


def _extract_order_id(text: str | None):
    if not text:
        return None
    match = re.search(r"\bo\d{4}\b", text.lower())
    if match:
        return match.group(0).upper()
    return None


def _extract_product_id(text: str | None):
    if not text:
        return None
    match = re.search(r"\bp\d{4}\b", text.lower())
    if match:
        return match.group(0).upper()
    return None


def _extract_customer_id(text: str | None):
    if not text:
        return None
    match = re.search(r"\bc\d{4}\b", text.lower())
    if match:
        return match.group(0).upper()
    return None


class VectorSearchTool:
    def __init__(
        self, index_path: Path, meta_path: Path, embedder: Embedder, empty: str
    ):
        self.index = VectorIndex(
            index_path=index_path, meta_path=meta_path, embedder=embedder
        )
        self.empty_message = empty

    def search(self, query: str, k: int = 5) -> list[dict]:
        return self.index.search(query, k=k)

    def retrieve(self, query: str, k: int = 5) -> str:
        try:
            results = self.search(query=query, k=k)
        except FileNotFoundError as exc:
            return str(exc)
        if not results:
            return self.empty_message
        return json.dumps(results, indent=2)


class CatalogSearchTool(VectorSearchTool):
    def __init__(self, index_path: Path, meta_path: Path, embedder: Embedder):
        super().__init__(index_path, meta_path, embedder, "No matching products found.")

    def search(self, query: str, k: int = 5) -> list[dict]:
        parsed = parse_query(query)
        cleaned_query = normalize_query(parsed.get("cleaned_query") or query)
        has_numeric = any(
            parsed.get(key) is not None
            for key in ("max_price", "min_price", "min_rating")
        )
        if not cleaned_query:
            try:
                results = list(self.index.items)
            except FileNotFoundError:
                return []
        else:
            candidate_k = max(_clamp_k(k) * 4, 20)
            results = self.index.search(cleaned_query, k=candidate_k)

        max_price = parsed.get("max_price")
        min_price = parsed.get("min_price")
        min_rating = parsed.get("min_rating")
        category = parsed.get("category")

        if max_price is not None:
            results = [
                r
                for r in results
                if r.get("price") is not None and r["price"] <= max_price
            ]
        if min_price is not None:
            results = [
                r
                for r in results
                if r.get("price") is not None and r["price"] >= min_price
            ]
        if min_rating is not None:
            results = [
                r
                for r in results
                if r.get("rating") is not None and r["rating"] >= min_rating
            ]
        if category:
            results = [
                r
                for r in results
                if r.get("category") and r["category"].lower() == category
            ]

        sort_hint = parsed.get("sort")
        if sort_hint == "price_asc":
            results = sorted(results, key=lambda r: r.get("price") or 0)
        elif sort_hint == "delivery_time_days_asc":
            results = sorted(results, key=lambda r: r.get("delivery_time_days") or 0)

        if not results and cleaned_query and has_numeric:
            try:
                results = list(self.index.items)
            except FileNotFoundError:
                results = []
            if max_price is not None:
                results = [
                    r
                    for r in results
                    if r.get("price") is not None and r["price"] <= max_price
                ]
            if min_price is not None:
                results = [
                    r
                    for r in results
                    if r.get("price") is not None and r["price"] >= min_price
                ]
            if min_rating is not None:
                results = [
                    r
                    for r in results
                    if r.get("rating") is not None and r["rating"] >= min_rating
                ]
            if category:
                results = [
                    r
                    for r in results
                    if r.get("category") and r["category"].lower() == category
                ]
            if sort_hint == "price_asc":
                results = sorted(results, key=lambda r: r.get("price") or 0)
            elif sort_hint == "delivery_time_days_asc":
                results = sorted(
                    results, key=lambda r: r.get("delivery_time_days") or 0
                )

        return results[: _clamp_k(k)]


class ProductFAQTool(VectorSearchTool):
    def __init__(self, index_path: Path, meta_path: Path, embedder: Embedder):
        super().__init__(index_path, meta_path, embedder, "No matching FAQs found.")


class CompanyPolicyTool(VectorSearchTool):
    def __init__(self, index_path: Path, meta_path: Path, embedder: Embedder):
        super().__init__(
            index_path, meta_path, embedder, "No matching policy entries found."
        )


class OrderDatabaseTool(VectorSearchTool):
    def __init__(self, index_path: Path, meta_path: Path, embedder: Embedder):
        super().__init__(index_path, meta_path, embedder, "No matching orders found.")

    def search(self, query: str, k: int = 5) -> list[dict]:
        normalized = normalize_query(query)
        try:
            if re.fullmatch(r"c\d{4}", normalized):
                return [
                    order
                    for order in self.index.items
                    if normalize_query(order.get("customer_id", "")) == normalized
                ][: _clamp_k(k)]
            if re.fullmatch(r"o\d{4}", normalized):
                return [
                    order
                    for order in self.index.items
                    if normalize_query(order.get("order_id", "")) == normalized
                ][: _clamp_k(k)]
        except FileNotFoundError:
            return []
        return super().search(query, k=k)


class GuardedOrderDatabaseTool(OrderDatabaseTool):
    def __init__(
        self,
        index_path: Path,
        meta_path: Path,
        embedder: Embedder,
        customer_id: str | None = None,
    ):
        super().__init__(index_path, meta_path, embedder)
        self.customer_id = customer_id

    def _normalized_customer_id(self, value: str | None) -> str | None:
        if not value:
            return None
        return normalize_query(value)

    def _matches_current_customer(self, value: str | None) -> bool:
        if not self.customer_id:
            return True
        if not value:
            return False
        return self._normalized_customer_id(value) == self._normalized_customer_id(
            self.customer_id
        )

    def search(self, query: str, k: int = 5) -> list[dict]:
        normalized = normalize_query(query)
        requested_customer = _extract_customer_id(query)
        if requested_customer and not self._matches_current_customer(
            requested_customer
        ):
            return []

        try:
            if re.fullmatch(r"c\d{4}", normalized):
                if not self._matches_current_customer(normalized):
                    return []
                return [
                    order
                    for order in self.index.items
                    if normalize_query(order.get("customer_id", "")) == normalized
                ][: _clamp_k(k)]
            if re.fullmatch(r"o\d{4}", normalized):
                results = [
                    order
                    for order in self.index.items
                    if normalize_query(order.get("order_id", "")) == normalized
                ]
                if self.customer_id:
                    results = [
                        order
                        for order in results
                        if self._matches_current_customer(order.get("customer_id"))
                    ]
                return results[: _clamp_k(k)]
        except FileNotFoundError:
            return []

        results = VectorSearchTool.search(self, query=query, k=k)
        if self.customer_id:
            results = [
                order
                for order in results
                if self._matches_current_customer(order.get("customer_id"))
            ]
        return results


class KnowledgeBaseTools:
    def __init__(
        self,
        customer_id: str | None = None,
        embed_model: str | None = None,
        index_dir: Path | None = None,
    ):
        embed_config = EmbeddingConfig(
            model_name=embed_model or EmbeddingConfig().model_name
        )
        embedder = Embedder(embed_config)
        index_root = index_dir or INDEX_DIR

        self.catalog = CatalogSearchTool(
            index_root / "catalog.faiss", index_root / "catalog_meta.json", embedder
        )
        self.faq = ProductFAQTool(
            index_root / "faq.faiss", index_root / "faq_meta.json", embedder
        )
        self.policy = CompanyPolicyTool(
            index_root / "policy.faiss", index_root / "policy_meta.json", embedder
        )
        self.orders = GuardedOrderDatabaseTool(
            index_root / "orders.faiss",
            index_root / "orders_meta.json",
            embedder,
            customer_id=customer_id,
        )

        self.customer_id = customer_id

    def _resolve_customer_id(self, customer_id: str | None):
        return customer_id or self.customer_id

    def _is_other_customer_query(self, query: str | None) -> bool:
        if not query or not self.customer_id:
            return False
        requested = _extract_customer_id(query)
        if not requested:
            return False
        return normalize_query(requested) != normalize_query(self.customer_id)

    def _order_belongs_to_customer(self, order: dict | None, customer_id: str | None):
        if not order or not customer_id:
            return False
        return normalize_query(order.get("customer_id", "")) == normalize_query(
            customer_id
        )

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
        log_path = DATA_DIR / "action_log.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _find_order(self, order_id: str):
        orders = load_json(str(ORDERS_PATH), [])
        for order in orders:
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

        if customer_id and not self._order_belongs_to_customer(order, customer_id):
            message = "Order does not belong to this customer."
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
            status = order.get("order_status") or "Unknown"
            reason = (
                f"Order status is {status}, only Placed or Shipped can be cancelled."
            )
            message = "Order is not eligible for cancellation."
            self._log_action(
                action="cancel_order",
                customer_id=customer_id,
                order_id=order_id,
                product_id=None,
                result="rejected",
                message=reason,
                ticket_id=None,
            )
            return json.dumps(
                {
                    "status": "rejected",
                    "message": message,
                    "reason": reason,
                    "ticket_id": None,
                },
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

        if customer_id and not self._order_belongs_to_customer(order, customer_id):
            message = "Order does not belong to this customer."
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
            status = order.get("order_status") or "Unknown"
            reason = f"Order status is {status}, only Delivered orders can be returned."
            message = "Order is not eligible for return."
            self._log_action(
                action="initiate_return",
                customer_id=customer_id,
                order_id=order_id,
                product_id=product_id,
                result="rejected",
                message=reason,
                ticket_id=None,
            )
            return json.dumps(
                {
                    "status": "rejected",
                    "message": message,
                    "reason": reason,
                    "ticket_id": None,
                },
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
            if self._is_other_customer_query(query):
                return "Access denied. You can only access your own orders."
            normalized = normalize_query(query)
            generic_orders = {
                "orders",
                "order",
                "my orders",
                "order history",
                "previous orders",
                "recent orders",
                "recent order",
                "my recent orders",
                "order list",
                "my order list",
            }
            if normalized in generic_orders:
                query = self.customer_id or query
            elif (
                self.customer_id
                and "order" in normalized
                and not _extract_order_id(query)
                and not _extract_customer_id(query)
            ):
                query = self.customer_id
            return self.orders.retrieve(query=query, k=k)
        if mode in {"catalog+faq", "catalog_faq", "catalog+faqs"}:
            combined = {
                "catalog": self.catalog.search(query=query, k=k),
                "faq": self.faq.search(query=query, k=k),
            }
            return json.dumps(combined, indent=2)
        return f"Unknown mode '{mode}'."

    def execute_tool_call(
        self, tool_name: str | None, args: dict | None, default_query: str = ""
    ) -> str:
        tool_name = (tool_name or "").strip()
        args = args or {}

        def _confirmation_required(action: str) -> str:
            return json.dumps(
                {
                    "status": "rejected",
                    "message": "Confirmation required before processing this request.",
                    "reason": f"confirmation_required:{action}",
                    "ticket_id": None,
                },
                indent=2,
            )

        if tool_name == "retrieve":
            query = args.get("query") or default_query
            mode = args.get("mode") or "catalog"
            k = _clamp_k(args.get("k", 5))
            if mode in {"orders", "order"}:
                if self._is_other_customer_query(query):
                    return "Access denied. You can only access your own orders."
                normalized = normalize_query(query)
                if (
                    self.customer_id
                    and "order" in normalized
                    and not _extract_order_id(query)
                    and not _extract_customer_id(query)
                ):
                    query = self.customer_id
            return self.retrieve(query=query, mode=mode, k=k)

        if tool_name == "cancel_order":
            if args.get("confirm") is not True:
                return _confirmation_required("cancel_order")
            order_id = args.get("order_id") or _extract_order_id(default_query)
            if not order_id:
                return "Missing order_id for cancellation."
            return self.cancel_order(order_id=order_id)

        if tool_name == "initiate_return":
            if args.get("confirm") is not True:
                return _confirmation_required("initiate_return")
            order_id = args.get("order_id") or _extract_order_id(default_query)
            product_id = args.get("product_id") or _extract_product_id(default_query)
            if not order_id or not product_id:
                return "Missing order_id or product_id for return."
            return self.initiate_return(order_id=order_id, product_id=product_id)

        return f"Unknown tool '{tool_name}'."


class CatalogSearch(CatalogSearchTool):
    pass
