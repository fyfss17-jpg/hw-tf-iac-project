import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import json
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

def calculate_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))

def parse_and_chunk_md(file_path):
    import os
import re

def parse_and_chunk_md(file_path, file_content=None):
    """
    华为云切片器：标准 Resource、index.md (Provider 凭证)、remote_state_backend.md
    """
    # 支持直接传入内容或读取文件
    if file_content:
        content = file_content
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
    base_name = os.path.splitext(os.path.basename(file_path))[0].lower()
    
    # 1. 动态识别文档类型
    doc_type = "resource"
    if "index" in base_name or "provider" in base_name:
        doc_type = "provider"
    elif "backend" in base_name or "remote_state" in base_name:
        doc_type = "backend"

    front_matter = {
        "resource_name": base_name,
        "doc_type": doc_type
    }
    
    # 2. 提取 Front Matter
    fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if fm_match:
        fm_text = fm_match.group(1)
        content = content[fm_match.end():]
        for line in fm_text.split('\n'):
            if ':' in line:
                k, v = line.split(':', 1)
                front_matter[k.strip()] = v.strip().strip('"').strip("'")
    
    # 3. 提取全局 H1 标题
    h1_match = re.search(r'^#\s+(.+)\$', content, re.MULTILINE)
    resource_title = h1_match.group(1).strip() if h1_match else base_name
    front_matter["resource_title"] = resource_title
    
    # 4. 核心变化：针对特殊文档，同时将二级(##)和三级(###)标题作为物理切分点
    if doc_type in ["provider", "backend"]:
        # 匹配 ## 或 ###
        blocks = re.split(r'(?=^###?\s+)', content, flags=re.MULTILINE)
    else:
        # 标准 Resource 仅按二级标题物理拆分
        blocks = re.split(r'(?=^##\s+)', content, flags=re.MULTILINE)
        
    chunks = []
    
    # 处理 H1 下的导言区
    first_block = blocks[0].strip()
    if first_block:
        chunks.append({
            "text": f"Context: {doc_type.upper()} ({resource_title})\nDescription:\n{first_block}",
            "metadata": front_matter.copy()
        })
        
    # 上层上下文追踪（用于把二级标题作为上下文带入到三级标题的 Chunk 中）
    current_h2_context = "General"
        
    # 5. 循环处理各个子块
    for block in blocks[1:]:
        block = block.strip()
        if not block:
            continue
            
        # 提取当前块的标题等级和名称
        header_match = re.match(r'^(###?)\s+(.+)\$', block, re.MULTILINE)
        if header_match:
            header_level = header_match.group(1)   # '##' 或 '###'
            header_name = header_match.group(2).strip()
            if header_level == "##":
                current_h2_context = header_name   # 刷新二级标题上下文
        else:
            header_name = "General"
            
        current_metadata = front_matter.copy()
        current_metadata["section"] = header_name
        current_metadata["parent_section"] = current_h2_context
        
        # 构造增强型的检索前缀路径
        if doc_type in ["provider", "backend"] and current_h2_context != header_name:
            chunk_prefix = f"Context: {doc_type.upper()} ({resource_title}) -> {current_h2_context} -> {header_name}\n"
        else:
            chunk_prefix = f"Context: {doc_type.upper()} ({resource_title}) -> {header_name}\n"
            
        # 识别是否为高价值的代码、凭证或示例段落
        is_example_or_auth = any(x in header_name.lower() for x in ["example", "usage", "authentication", "credentials"])
        has_code_block = any(lang in block for lang in ["```hcl", "```terraform", "```tf", "```sh", "```bash"])
        
        # 策略 A: 保护高价值示例/配置的绝对完整
        if is_example_or_auth or has_code_block:
            chunks.append({
                "text": f"{chunk_prefix}{block}",
                "metadata": current_metadata
            })
        else:
            # 策略 B: 参数说明（Argument Reference）过长时的动态截断
            if len(block) > 1000:
                # 兼容 * `param` 或 - `param` 形式的参数列表
                sub_sections = re.split(r'\n(?=(?:\*|-)\s+`)', block)
                intro = sub_sections[0]
                current_chunk_text = f"{chunk_prefix}{intro}"
                
                for sub in sub_sections[1:]:
                    if len(current_chunk_text) + len(sub) > 1000:
                        chunks.append({"text": current_chunk_text.strip(), "metadata": current_metadata})
                        current_chunk_text = f"{chunk_prefix}(Continued):\n* {sub.strip()}"
                    else:
                        current_chunk_text += f"\n* {sub.strip()}"
                        
                if current_chunk_text:
                    chunks.append({"text": current_chunk_text.strip(), "metadata": current_metadata})
            else:
                chunks.append({
                    "text": f"{chunk_prefix}{block}",
                    "metadata": current_metadata
                })
                
    return chunks

def main():
    input_folder = "../raw_terraform_docs"
    cache_folder = "../chunks_cache" 
    os.makedirs(cache_folder, exist_ok=True)
    
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        return

    all_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.md')]
    print(f"🚀 开始安全模式切片，共 {len(all_files)} 个文档...")

    total_chunks_count = 0
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        resource_key = os.path.splitext(file_name)
        
        file_chunks = parse_and_chunk_md(file_path) # 调用切片
        if not file_chunks:
            continue
            
        file_serialized_data = []
        for idx, chunk in enumerate(file_chunks):
            t_count = calculate_tokens(chunk["text"])
            chunk["metadata"]["token_count"] = t_count
            
            file_serialized_data.append({
                "id": f"{resource_key}_chunk_{idx}",
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            })
        
        output_cache_path = os.path.join(cache_folder, f"{resource_key}.json")
        with open(output_cache_path, 'w', encoding='utf-8') as f:
            json.dump(file_serialized_data, f, ensure_ascii=False, indent=2)
            
        total_chunks_count += len(file_serialized_data)
        print(f"   ✅ [{file_name}] -> 独立导出至 [{resource_key}.json]（含 {len(file_chunks)} 个切片）")

    print(f"\n🎉 纯本地流式切片结束！总共生成 {total_chunks_count} 个 Chunks。")
    print(f"🗂️ 所有切片已被安全分散保存在：{os.path.abspath(cache_folder)} 文件夹中，可以用 VS Code 逐个秒开查看。")

if __name__ == "__main__":
    main()
