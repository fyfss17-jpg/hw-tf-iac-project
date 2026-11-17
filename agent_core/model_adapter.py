from __future__ import annotations

from typing import List

import dashscope
from openai import OpenAI

from agent_core.config import require_env


class ModelAdapter:
    """统一封装 DashScope/OpenAI 兼容模型调用能力。"""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url or require_env("DASHSCOPE_BASE_URL")
        self.api_key = api_key or require_env("DASHSCOPE_API_KEY")
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def embed_text(self, text: str, *, text_type: str = "query") -> list[float]:
        response = self._client.embeddings.create(
            model="qwen3.7-text-embedding-flash",
            input=text,
            extra_body={"text_type": text_type},
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str], *, text_type: str = "document") -> List[List[float]]:
        response = dashscope.TextEmbedding.call(
            model="qwen3.7-text-embedding-flash",
            input=texts,
            text_type=text_type,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Embedding API failed: {response.message}")
        return [item["embedding"] for item in response.output["embeddings"]]

    def rerank(self, query: str, documents: List[str], *, top_n: int = 3) -> List[str]:
        if not documents:
            return []

        if len(documents) <= top_n:
            return documents[:top_n]

        response = dashscope.TextReRank.call(
            model="qwen3.7-text-rerank",
            api_key=self.api_key,
            query=query,
            documents=documents,
            top_n=top_n,
        )

        if response.status_code != 200:
            return documents[:top_n]

        results = getattr(response.output, "results", []) or []
        ranked_docs: List[str] = []
        for item in results:
            index = item.get("index") if isinstance(item, dict) else getattr(item, "index", None)
            if index is None:
                continue
            if 0 <= index < len(documents):
                ranked_docs.append(documents[index])

        if ranked_docs:
            return ranked_docs
        return documents[:top_n]


model_adapter = ModelAdapter()
