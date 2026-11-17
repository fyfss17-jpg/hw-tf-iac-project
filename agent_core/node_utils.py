import json
import os
import re


def save_hcl_files_to_workspace(llm_output: str, workspace_dir: str) -> list[str]:
    """解析大模型输出的多文件代码块并写入工作区目录。"""
    os.makedirs(workspace_dir, exist_ok=True)
    saved_files = []

    pattern = r"#{2,4}\s*(?:FILE:\s*)?[\*`]*([a-zA-Z0-9_.-]+\.(?:tf|tfvars))[*`]*\s*\n+```(?:hcl|terraform|tf)?\s*\n(.*?)```"
    matches = re.findall(pattern, llm_output, flags=re.DOTALL | re.IGNORECASE)

    if matches:
        for filename, code_content in matches:
            filename = filename.strip()
            file_path = os.path.join(workspace_dir, filename)
            mode = "a" if filename in saved_files else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                prefix = "\n\n" if mode == "a" else ""
                f.write(prefix + code_content.strip() + "\n")
            if filename not in saved_files:
                saved_files.append(filename)
    else:
        fallback_blocks = re.findall(r"```(?:hcl|terraform|tf)?\n(.*?)```", llm_output, flags=re.DOTALL | re.IGNORECASE)
        combined_code = "\n\n".join(b.strip() for b in fallback_blocks) if fallback_blocks else llm_output.strip()
        file_path = os.path.join(workspace_dir, "main.tf")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(combined_code + "\n")
        saved_files.append("main.tf")

    return saved_files


def extract_terraform_diff(plan_json_path: str) -> list:
    """提取 Terraform plan.json 中的精简 diff，供语义解释使用。"""
    if not os.path.exists(plan_json_path):
        return []

    with open(plan_json_path, "r", encoding="utf-8") as f:
        plan_data = json.load(f)

    summarized_changes = []
    for resource in plan_data.get("resource_changes", []):
        address = resource.get("address")
        res_type = resource.get("type")
        change = resource.get("change", {})
        actions = change.get("actions", [])

        if actions == ["no-op"] or not actions:
            continue

        before = change.get("before", {})
        after = change.get("after", {})
        summarized_changes.append({
            "resource_address": address,
            "resource_type": res_type,
            "action": actions,
            "key_attributes_changed": {"before": before, "after": after},
        })

    return summarized_changes


def archive_previous_version(workspace_dir: str, prev_version: int) -> None:
    """归档上一版本工程文件，保留历史版本。"""
    if not os.path.exists(workspace_dir):
        return
    history_dir = os.path.join(workspace_dir, ".history", f"v{prev_version}")
    os.makedirs(history_dir, exist_ok=True)
    for fname in os.listdir(workspace_dir):
        if fname.endswith((".tf", ".tfvars")):
            src = os.path.join(workspace_dir, fname)
            if os.path.isfile(src):
                dest = os.path.join(history_dir, fname)
                with open(src, "rb") as src_f, open(dest, "wb") as dst_f:
                    dst_f.write(src_f.read())
