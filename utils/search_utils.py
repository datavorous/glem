import json
import re
from typing import Any, Dict, Iterable, List

from fuzzysearch import find_near_matches


def load_json(path: str, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {path} not found.")
        return default


def normalize_query(query: str) -> str:
    return (query or "").lower().strip()


def fuzzy_score(query: str, haystack: str, weight: int) -> int:
    if not query or not haystack:
        return 0
    matches = find_near_matches(query, haystack.lower(), max_l_dist=2)
    if not matches:
        return 0
    best = min(m.dist for m in matches)
    return max(weight - best, 1)


def score_field(query: str, value: str, weight: int) -> int:
    if not value:
        return 0
    value_lower = value.lower()
    if query in value_lower:
        return weight
    return fuzzy_score(query, value, weight)


def rank_results(results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    k = max(1, int(k))
    results.sort(key=lambda r: r["score"], reverse=True)
    return [r["item"] for r in results[:k]]


def flatten_policy(node: Any, path: Iterable[str] = ()):
    entries = []
    if isinstance(node, dict):
        topic = node.get("topic")
        new_path = list(path)
        if topic:
            new_path.append(topic)
        if "clean_text" in node:
            entries.append(
                {
                    "section": " > ".join(new_path) if new_path else "policy",
                    "text": node.get("clean_text", ""),
                }
            )
        if "segments" in node:
            for seg in node.get("segments", []):
                entries.extend(flatten_policy(seg, new_path))
        for key, value in node.items():
            if key in {"topic", "clean_text", "segments", "citation_ids"}:
                continue
            entries.extend(flatten_policy(value, list(path) + [key.replace("_", " ")]))
    elif isinstance(node, list):
        for item in node:
            entries.extend(flatten_policy(item, path))
    return entries


def _parse_number(token: str):
    if not token:
        return None
    token = token.lower().replace(",", "").strip()
    match = re.match(r"^(\d+(?:\.\d+)?)(k)?$", token)
    if not match:
        return None
    value = float(match.group(1))
    if match.group(2):
        value *= 1000
    return int(value)


def parse_query(text: str) -> dict:
    data = {
        "cleaned_query": text or "",
        "max_price": None,
        "min_price": None,
        "min_rating": None,
        "category": None,
        "sort": None,
    }
    if not text:
        return data
    try:
        raw = text.lower()
        raw = raw.replace("$", " ")

        category_match = re.search(
            r"\b(electronics|clothing|home|beauty|sports|toys|kitchen)\b", raw
        )
        if category_match:
            data["category"] = category_match.group(1)

        max_match = re.search(
            r"(?:under|below|less than)\s+(\d{1,3}(?:,\d{3})*|\d+(?:\.\d+)?k?)",
            raw,
        )
        if max_match:
            data["max_price"] = _parse_number(max_match.group(1))

        min_match = re.search(
            r"(?:above|more than|over)\s+(\d{1,3}(?:,\d{3})*|\d+(?:\.\d+)?k?)",
            raw,
        )
        if min_match:
            data["min_price"] = _parse_number(min_match.group(1))

        rating_match = re.search(
            r"\b(top rated|best)\b(?:\s*(?:above|over|>=|at least)\s*(\d+(?:\.\d+)?))?",
            raw,
        )
        if rating_match:
            if rating_match.group(2):
                try:
                    data["min_rating"] = float(rating_match.group(2))
                except ValueError:
                    data["min_rating"] = 4
            else:
                data["min_rating"] = 4

        if "cheapest" in raw:
            data["sort"] = "price_asc"
        elif "fastest" in raw:
            data["sort"] = "delivery_time_days_asc"

        cleaned = re.sub(r"\d{1,3}(?:,\d{3})*|\d+(?:\.\d+)?k?", " ", raw)
        cleaned = re.sub(
            r"\b(under|below|less|than|above|more|over|cheapest|fastest|top|rated|best|electronics|clothing|home|beauty|sports|toys|kitchen)\b",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        data["cleaned_query"] = cleaned or text
    except Exception:
        return data
    return data
