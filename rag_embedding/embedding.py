import json
import logging
import os
import time

import chromadb

from agent_core.config import require_env
from agent_core.model_adapter import model_adapter

# 初始化日志配置：同时把日志打向控制台和本地文件，且支持 UTF-8 防止中文乱码
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("ingest_run.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

require_env("DASHSCOPE_BASE_URL")
require_env("DASHSCOPE_API_KEY")


def get_qwen_embeddings_batch(texts, text_type='document'):
    try:
        return model_adapter.embed_batch(texts, text_type=text_type)
    except Exception as e:
        logging.error(f"请求失败: {e}")
        return None

def ingest_cached_chunks_with_checkpoint():
    cache_folder = "../chunks_cache"
    chroma_db_path = "../chroma_db_storage"
    collection_name = "huaweicloud_terraform_rag"
    
    if not os.path.exists(cache_folder):
        logging.warning("❌ 缓存文件夹不存在")
        return

    client = chromadb.PersistentClient(path=chroma_db_path)
    collection = client.get_or_create_collection(name=collection_name)
    
    json_files = [os.path.join(cache_folder, f) for f in os.listdir(cache_folder) if f.endswith('.json')]
    logging.info(f"📂 开始断点检查并向量化共 {len(json_files)} 个缓存文件...") # 🌟 替换 print 为 logging.info

    API_BATCH_SIZE = 20
    current_ids, current_documents, current_metadatas = [], [], []
    total_inserted = 0
    total_skipped = 0

    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                chunks_data = json.load(f)
            except Exception:
                continue
                
        chunk_ids_in_file = [item["id"] for item in chunks_data]
        existing_results = collection.get(ids=chunk_ids_in_file)
        existing_ids = set(existing_results['ids']) if existing_results else set()

        for item in chunks_data:
            if item["id"] in existing_ids:
                total_skipped += 1
                continue
                
            current_ids.append(item["id"])
            current_documents.append(item["text"])
            clean_metadata = {k: (str(v) if isinstance(v, (dict, list)) else v) for k, v in item["metadata"].items()}
            current_metadatas.append(clean_metadata)
            
            if len(current_ids) >= API_BATCH_SIZE:
                vectors = get_qwen_embeddings_batch(current_documents, text_type='document')
                if vectors:
                    collection.add(ids=current_ids, documents=current_documents, metadatas=current_metadatas, embeddings=vectors)
                    total_inserted += len(current_ids)
                    logging.info(f"   ⚡ [同步成功] 已追加落库 {total_inserted} 个 Chunks... (已累计跳过历史数据 {total_skipped} 个)")
                
                current_ids, current_documents, current_metadatas = [], [], []
                time.sleep(0.1)

    if current_ids:
        vectors = get_qwen_embeddings_batch(current_documents, text_type='document')
        if vectors:
            collection.add(ids=current_ids, documents=current_documents, metadatas=current_metadatas, embeddings=vectors)
            total_inserted += len(current_ids)

    logging.info(f"\n🎉 同步结束！本次新向量化入库 {total_inserted} 个 Chunk，安全检测并直接跳过 {total_skipped} 个已有 Chunk。")

if __name__ == "__main__":
    ingest_cached_chunks_with_checkpoint()
