from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


load_dotenv()


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _point_id(source: str, chunk_idx: int) -> str:
    return _md5(f"{source}::{chunk_idx}")


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    word_tokens = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)
    return word_tokens if word_tokens else list(text)


@dataclass
class Settings:
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_chat_model: str = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "data/chroma")
    vector_collection: str = os.getenv("VECTOR_COLLECTION", "books_rag")
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    vector_recall_k: int = int(os.getenv("VECTOR_RECALL_K", "30"))
    bm25_recall_k: int = int(os.getenv("BM25_RECALL_K", "30"))


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._validate_required_env()

        self.embedder = SentenceTransformer(self.settings.embed_model)
        self.reranker = CrossEncoder(self.settings.rerank_model)
        self.vector_size = self.embedder.get_sentence_embedding_dimension()

        self.vector_client = chromadb.PersistentClient(
            path=self.settings.vector_db_path,
        )
        self.collection = self.vector_client.get_or_create_collection(
            name=self.settings.vector_collection,
            metadata={"distance": "cosine"},
        )

        self.llm_client = OpenAI(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
        )

        self._bm25_cache: list[dict] = []
        self._bm25_index: BM25Okapi | None = None
        self._last_query: str | None = None
        self._last_query_vector: list[float] | None = None

    def _validate_required_env(self) -> None:
        missing = []
        if not self.settings.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")
        if missing:
            raise ValueError(f"缺少必要环境变量: {', '.join(missing)}")

    @staticmethod
    def read_file(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        raise ValueError(f"不支持的文件格式: {path}")

    def chunk_text(self, text: str) -> list[str]:
        text = _normalize(text)
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.settings.chunk_size)
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - self.settings.chunk_overlap
        return chunks

    def _existing_source_hash(self, source: str) -> str | None:
        result = self.collection.get(
            where={"source": source},
            include=["metadatas"],
            limit=1,
        )
        metadatas = result.get("metadatas") or []
        if not metadatas:
            return None
        return metadatas[0].get("source_hash")

    def _delete_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def _invalidate_retrieval_cache(self) -> None:
        self._bm25_cache = []
        self._bm25_index = None
        self._last_query = None
        self._last_query_vector = None

    def ingest_file(self, file_path: Path) -> str:
        source = file_path.name
        content = self.read_file(file_path)
        source_hash = _md5(content)
        old_hash = self._existing_source_hash(source)

        if old_hash == source_hash:
            return f"跳过 {source}（内容未变化）"

        if old_hash is not None:
            self._delete_source(source)

        chunks = self.chunk_text(content)
        if not chunks:
            return f"跳过 {source}（空内容）"

        vectors = self.embedder.encode(chunks, normalize_embeddings=True)
        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict] = []
        documents: list[str] = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            ids.append(_point_id(source, i))
            embeddings.append(vector.tolist())
            documents.append(chunk)
            metadatas.append(
                {
                    "source": source,
                    "chunk_idx": i,
                    "source_hash": source_hash,
                }
            )

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        self._invalidate_retrieval_cache()
        return f"已写入 {source}，共 {len(chunks)} 个分块"

    def ingest_dir(self, directory: Path) -> list[str]:
        supported = {".txt", ".md", ".pdf"}
        files = sorted(p for p in directory.iterdir() if p.suffix.lower() in supported)
        return [self.ingest_file(p) for p in files]

    def _query_vector(self, question: str) -> list[float]:
        if question == self._last_query and self._last_query_vector is not None:
            return self._last_query_vector
        vector = self.embedder.encode(question, normalize_embeddings=True).tolist()
        self._last_query = question
        self._last_query_vector = vector
        return vector

    def _load_bm25_index(self) -> None:
        if self._bm25_index is not None:
            return

        records: list[dict] = []
        points = self.collection.get(include=["documents", "metadatas"])
        ids = points.get("ids") or []
        docs = points.get("documents") or []
        metadatas = points.get("metadatas") or []
        for pid, text, metadata in zip(ids, docs, metadatas):
            records.append(
                {
                    "id": str(pid),
                    "source": metadata.get("source"),
                    "chunk_idx": metadata.get("chunk_idx"),
                    "text": text or "",
                }
            )

        self._bm25_cache = records
        corpus = [_tokenize(r["text"]) for r in records]
        self._bm25_index = BM25Okapi(corpus) if corpus else None

    def retrieve_hybrid(self, question: str, top_k: int | None = None) -> list[dict]:
        self._load_bm25_index()
        final_k = top_k or self.settings.top_k

        query_limit = max(final_k, self.settings.vector_recall_k)
        vector_hits = self.collection.query(
            query_embeddings=[self._query_vector(question)],
            n_results=query_limit,
            include=["documents", "metadatas", "distances"],
        )
        vector_results = {}
        ids = vector_hits.get("ids", [[]])[0]
        docs = vector_hits.get("documents", [[]])[0]
        metadatas = vector_hits.get("metadatas", [[]])[0]
        distances = vector_hits.get("distances", [[]])[0]
        for pid, text, metadata, distance in zip(ids, docs, metadatas, distances):
            vector_results[str(pid)] = {
                "id": str(pid),
                "vector_score": 1.0 - float(distance),
                "bm25_score": 0.0,
                "source": metadata.get("source"),
                "chunk_idx": metadata.get("chunk_idx"),
                "text": text or "",
            }

        bm25_results: dict[str, dict] = {}
        if self._bm25_index is not None and self._bm25_cache:
            bm25_scores = self._bm25_index.get_scores(_tokenize(question))
            ranked = sorted(enumerate(bm25_scores), key=lambda item: item[1], reverse=True)
            for idx, score in ranked[: self.settings.bm25_recall_k]:
                rec = self._bm25_cache[idx]
                bm25_results[rec["id"]] = {
                    "id": rec["id"],
                    "vector_score": 0.0,
                    "bm25_score": float(score),
                    "source": rec["source"],
                    "chunk_idx": rec["chunk_idx"],
                    "text": rec["text"],
                }

        merged = dict(vector_results)
        for pid, item in bm25_results.items():
            if pid in merged:
                merged[pid]["bm25_score"] = item["bm25_score"]
            else:
                merged[pid] = item

        max_v = max((x["vector_score"] for x in merged.values()), default=1.0) or 1.0
        max_b = max((x["bm25_score"] for x in merged.values()), default=1.0) or 1.0
        merged_items = list(merged.values())
        for item in merged_items:
            item["hybrid_score"] = 0.55 * (item["vector_score"] / max_v) + 0.45 * (
                item["bm25_score"] / max_b
            )

        merged_items.sort(key=lambda x: x["hybrid_score"], reverse=True)

        seen_text: set[str] = set()
        deduped: list[dict] = []
        for item in merged_items:
            text_hash = _md5(_normalize(item["text"]))
            if text_hash in seen_text:
                continue
            seen_text.add(text_hash)
            deduped.append(item)
            if len(deduped) >= max(final_k * 3, final_k):
                break

        if not deduped:
            return []

        rerank_pairs = [(question, item["text"]) for item in deduped]
        rerank_scores = self.reranker.predict(rerank_pairs)
        for item, rr_score in zip(deduped, rerank_scores):
            item["rerank_score"] = float(rr_score)

        deduped.sort(key=lambda x: x["rerank_score"], reverse=True)
        return deduped[:final_k]


    def answer(self, question: str) -> str:
        contexts = self.retrieve_hybrid(question)
        context_text = "\n\n".join(
            f"[来源:{c['source']}#{c['chunk_idx']}]\n{c['text']}" for c in contexts
        )

        system_prompt = (
            "你是一个严谨的读书助理。仅依据提供的上下文回答，"
            "如果上下文不足，请明确说“根据当前知识库无法确定”。"
        )
        user_prompt = f"上下文:\n{context_text}\n\n问题: {question}"

        resp = self.llm_client.chat.completions.create(
            model=self.settings.deepseek_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""
