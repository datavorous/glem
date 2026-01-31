import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError(
        "numpy is required for vector search. Install requirements first."
    ) from exc

try:
    import faiss
except ImportError as exc:
    raise RuntimeError(
        "faiss-cpu is required for vector search. Install requirements first."
    ) from exc


def load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {path} not found.")
        return default


def normalize_query(query: str) -> str:
    return (query or "").lower().strip()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | None = None


class Embedder:
    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for embeddings. Install requirements first."
            ) from exc
        if self.config.device:
            self._model = SentenceTransformer(
                self.config.model_name, device=self.config.device
            )
        else:
            self._model = SentenceTransformer(self.config.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        model = self._load_model()
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")
        return vectors


@dataclass
class VectorIndex:
    index_path: Path
    meta_path: Path
    embedder: Embedder
    _index: faiss.Index | None = None
    _metadata: list[dict] | None = None

    def _load(self):
        if self._index is None or self._metadata is None:
            if not self.index_path.exists() or not self.meta_path.exists():
                raise FileNotFoundError(
                    f"Missing index files for {self.index_path.stem}. "
                    "Run scripts/build_faiss_indexes.py first."
                )
            self._index = faiss.read_index(str(self.index_path))
            self._metadata = _load_metadata(self.meta_path)
        return self._index, self._metadata

    @property
    def items(self) -> list[dict]:
        self._load()
        return self._metadata or []

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not query:
            return []
        index, metadata = self._load()
        k = max(1, int(k))
        vectors = self.embedder.embed_texts([query])
        scores, idx = index.search(vectors, k)
        results = []
        for item_idx, score in zip(idx[0], scores[0]):
            if item_idx < 0 or item_idx >= len(metadata):
                continue
            item = dict(metadata[item_idx])
            item["_score"] = float(score)
            results.append(item)
        return results


def _load_metadata(path: Path) -> list[dict]:
    data = load_json(str(path), [])
    if isinstance(data, dict) and "items" in data:
        return data.get("items") or []
    if isinstance(data, list):
        return data
    return []


def save_metadata(path: Path, items: list[dict], model_name: str | None = None) -> None:
    payload = {
        "model": model_name,
        "count": len(items),
        "items": items,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def build_faiss_index(
    texts: list[str],
    items: list[dict],
    embedder: Embedder,
    index_path: Path,
    meta_path: Path,
) -> None:
    if not texts:
        raise ValueError("No texts provided for indexing.")
    if len(texts) != len(items):
        raise ValueError("texts and items must have the same length.")
    vectors = embedder.embed_texts(texts)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    ensure_dir(index_path.parent)
    faiss.write_index(index, str(index_path))
    save_metadata(meta_path, items, model_name=embedder.config.model_name)


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
