# 检索服务：在真正调用时才初始化 Chroma，避免 import-time 副作用。

from functools import lru_cache
from typing import List

import chromadb

from agent_core.config import require_env
from agent_core.model_adapter import model_adapter


@lru_cache(maxsize=1)
def get_collection():
    """懒加载创建 Chroma collection，避免模块导入时强制要求环境和数据库可用。"""
    require_env("DASHSCOPE_BASE_URL")
    require_env("DASHSCOPE_API_KEY")
    chroma_client = chromadb.PersistentClient(path="./chroma_db_storage")
    return chroma_client.get_or_create_collection(name="huaweicloud_terraform_rag")


# 底层提供 Embedding 与 Rerank 封装函数，支持向量化检索及结果精排过滤
def get_embedding(text: str) -> list[float]:
    """调用云端 qwen3.7-text-embedding-flash 获取单条文本的向量"""
    return model_adapter.embed_text(text, text_type="query")


def qwen_rerank(query: str, documents: list[str], top_n: int = 3):
    """调用阿里云百炼官方 SDK 的 qwen3.7-text-rerank 模型进行精准重排"""
    return model_adapter.rerank(query, documents, top_n=top_n)


def retrieve_tf_knowledge(user_query: str, target_resources: List[str]) -> str:
    """
    双路混合召回主入口：
    1. 宏观骨架路：用 user_query 检索 best_practice，获取多资源联动依赖模板。
    2. 微观字段路：遍历 target_resources 逐个检索 resource 文档并精排。
    3. 附加路：针对 backend/obs 等关键词补充 Provider/Backend 基础配置。
    """
    collection = get_collection()
    collected_snippets = []
    seen_prefixes = set()

    def add_unique_docs(docs: list[str], source_label: str):
        for doc in docs:
            if isinstance(doc, dict):
                doc_text = str(doc.get("text", doc))
            elif isinstance(doc, list):
                doc_text = str(doc[0]) if doc else ""
            else:
                doc_text = str(doc)

            if not doc_text:
                continue

            fingerprint = doc_text[:100].strip()
            if fingerprint not in seen_prefixes:
                seen_prefixes.add(fingerprint)
                collected_snippets.append(f"--- [来源: {source_label}] ---\n{doc_text}")

    query_vector = get_embedding(user_query)
    bp_results = collection.query(
        query_embeddings=[query_vector],
        where={"doc_type": "best_practice"},
        n_results=4,
    )
    if bp_results and bp_results["documents"] and bp_results["documents"][0]:
        top_bp = qwen_rerank(user_query, bp_results["documents"][0], top_n=1)
        add_unique_docs(top_bp, "最佳实践架构模板")

    search_targets = target_resources if target_resources else [user_query]
    per_resource_top_n = 2 if len(search_targets) <= 5 else 1

    for res_name in search_targets:
        res_query = f"Terraform HuaweiCloud {res_name} arguments and example usage"
        res_vec = get_embedding(res_query)

        res_results = collection.query(
            query_embeddings=[res_vec],
            where={"doc_type": "resource"},
            n_results=5,
        )
        if res_results and res_results["documents"] and res_results["documents"][0]:
            top_res = qwen_rerank(
                res_query, res_results["documents"][0], top_n=per_resource_top_n
            )
            add_unique_docs(top_res, f"资源文档：{res_name}")

    global_keywords = ["backend", "obs", "状态", "认证", "ak", "sk", "provider", "region"]
    if any(kw in user_query.lower() for kw in global_keywords):
        global_results = collection.query(
            query_embeddings=[query_vector],
            where={"doc_type": {"$in": ["provider", "backend"]}},
            n_results=4,
        )
        if global_results and global_results["documents"] and global_results["documents"][0]:
            top_global = qwen_rerank(user_query, global_results["documents"][0], top_n=2)
            add_unique_docs(top_global, "全局 Provider / Backend 配置")

    if not collected_snippets:
        return "未在本地知识库中检索到相关参考文档，请严格依据标准 Terraform 华为云规范编写。"
    return "\n\n".join(collected_snippets)


def simple_retrieve(user_query: str, top_n: int = 3) -> list[str]:
    collection = get_collection()
    query_vector = get_embedding(user_query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_n,
    )
    docs = results.get("documents", [[]])
    return docs


