# 编写五大核心节点 (路由、检索、生成、Docker沙箱校验、反思)
# • router_node（路由员）：调用轻量模型（预留 qwen3.8-flash），判断用户是要“写 TF 代码”还是“日常闲聊/问概念”。
# • retrieve_node（图书管理员）：调用 rag_retriever.py，把最精准的 Top-3 文档填进记事本。
# • generate_node（主程工程师）：调用编码模型（预留 qwen3.7-plus），根据文档写出 HCL 代码；如果是重试轮次，还会参考反思节点的修改意见进行重写。
# • sandbox_validate_node（铁面测试员）：纯 Python 代码。在 Windows 临时目录生成 main.tf，拉起你的 tf-sandbox:huawei 离线容器执行校验，把成功或报错信息写回记事本。
# • reflect_node（架构师复盘）：调用最强推理模型（预留 qwen-max）。一旦沙箱报错，它负责对比报错日志和原始文档，找出错误根因并输出修改指令。

# generate node 注意 加提示词“所有存在上下游关联的资源（如子网归属 VPC、网卡绑定安全组），必须严格使用 HCL 隐式属性引用语法（例如 resource_type.local_name.id），严禁硬编码假 ID。”


import json
import logging
import os
import re
import shutil
import subprocess
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from agent_core.node_utils import archive_previous_version, extract_terraform_diff, save_hcl_files_to_workspace
from agent_core.rag_retriever import retrieve_tf_knowledge, simple_retrieve
from agent_core.services.llm_service import LLMService
from agent_core.state import TerraformAgentState

logger = logging.getLogger(__name__)

# ==================== 1. 初始化百炼多模型矩阵 ====================
llm_service = LLMService()
llm_router = llm_service.router
llm_coder = llm_service.coder
llm_reflector = llm_service.reflector

def _read_workspace_tf_files(workspace_dir: str) -> str:
  """读取工作区当前所有的 .tf 和 .tfvars 文件并格式化为 Prompt 上下文"""
  existing_code_summary = []
  if os.path.exists(workspace_dir):
    for fname in sorted(os.listdir(workspace_dir)):
      if fname.endswith((".tf", ".tfvars")):
        fpath = os.path.join(workspace_dir, fname)
        if os.path.isfile(fpath):
          with open(fpath, "r", encoding="utf-8") as f:
            existing_code_summary.append(
                f"### FILE: {fname}\n```hcl\n{f.read().strip()}\n```"
            )
  return "\n\n".join(existing_code_summary)

# ==================== 3. 节点 1：路由与需求拆解节点 ====================
def router_node(state: TerraformAgentState) -> dict:
    """
    判断用户意图，
    并将模糊的中文架构需求翻译、拆解为标准的华为云 Terraform 资源列表。
    """
    user_query = state["user_query"]
    logger.info("[Router 路由节点] 正在分析用户意图并拆解华为云资源清单...")

    system_prompt = """你是一个资深的华为云架构师与 Terraform 专家。请分析用户的输入，完成以下两项任务：

1. 判断意图 (intent):
   - "generate_tf": 仅当用户明确要求编写、生成、修改具体的 Terraform HCL 代码块，或要求构建具体的云资源实例（如“创建一个ECS”、“写一个VPC的tf文件”）时返回此值。
   - "general_qa": 当用户询问概念解释、最佳实践、架构设计方案、配置方法原理、合规性建议或操作流程（如“标签如何统一绑定”、“State文件怎么存到OBS”、“什么是Remote Backend”、“推荐的安全组规则”）时，返回此值。注意：即使问题涉及技术实现细节，若未明确要求输出完整可执行的HCL代码，均归为此类。

2. 拆解目标资源清单 (target_resources):
   - 当 intent 为 "generate_tf" 时：将用户的自然语言需求精准翻译并拆解为华为云 Terraform Provider 支持的标准资源类型名称列表（例如：用户说“配一个带子网和安全组的虚拟机”，拆解为 ["huaweicloud_vpc", "huaweicloud_vpc_subnet", "huaweicloud_networking_secgroup", "huaweicloud_compute_instance"]）。
   - 当 intent 为 "general_qa" 时：返回空列表 []。

必须严格返回纯 JSON 格式，不要包含任何 Markdown 代码块标记（如 ```json）或多余解释文字。

JSON 格式示例:
{
  "intent": "generate_tf",
  "target_resources": ["huaweicloud_vpc", "huaweicloud_vpc_subnet"]
}
"""


    response = llm_router.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"用户输入：{user_query}")
    ])

    raw_content = response.content.strip()
    # 清理可能意外带出的 markdown json 标记
    raw_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw_content)
        intent = parsed.get("intent", "generate_tf")
        target_resources = parsed.get("target_resources", [])
    except Exception as e:
        logger.warning("[Router 警告] JSON 解析失败 (%s)，自动降级为默认代码生成模式。", e)
        intent = "generate_tf"
        target_resources = []

    logger.info("意图判定: %s | 拆解资源清单: %s", intent, target_resources)

    return {
        "intent": intent,
        "target_resources": target_resources,
        "retry_count": 0,
        "validation_passed": False,
        "security_passed": False,
    }


# ==================== 4. 节点 2：知识检索节点 ====================
def retrieve_node(state: TerraformAgentState) -> dict:
    """
    调用双路混合检索引擎 (Chroma 向量粗筛 + Qwen Rerank 精排)，
    根据路由节点拆解出的 target_resources 精确制导捞取参考文档。
    """
    user_query = state["user_query"]
    target_resources = state.get("target_resources", [])
    
    logger.info("[Retrieve 检索节点] 正在从本地 ChromaDB 检索相关文档与最佳实践...")
    
    # 调用你在 rag_retriever.py 中封装好的双路召回引擎
    rag_context = retrieve_tf_knowledge(
        user_query=user_query,
        target_resources=target_resources
    )
    
    logger.info("知识检索与精排完成，组装参考上下文总长度: %s 字符。", len(rag_context))
    with open("temptest.txt", "w", encoding="utf-8") as f:
        f.write(rag_context)
    return {
        "rag_context": rag_context
    }
# ==================== 5. 节点 3：代码生成与多文件落盘节点 ====================
def generate_node(state: TerraformAgentState) -> dict:
    """
    使用高编码能力模型生成/修复 Terraform 代码并直接写入本地 workspace_dir。
    具体源码结构与提示词约束请参考完整实现代码。
    支持三种工作模式：
    1. 首次初始生成模式 (retry_count == 0 且无 human_feedback)
    2. 人工协同增量修改模式 (存在 human_feedback，自动备份旧版本并增量迭代)
    3. 沙箱报错自愈修复模式 (retry_count > 0，根据反思建议修复报错)
    """
    user_query = state["user_query"]
    rag_context = state.get("rag_context", "")
    workspace_dir = state["workspace_dir"]
    retry_count = state.get("retry_count", 0)
    error_message = state.get("error_message")
    reflection_advice = state.get("reflection_advice")
    
    # 读取版本与人类修改指令
    version = state.get("version", 1)
    human_feedback = state.get("human_feedback")
    
    system_prompt = """你是一个顶级的华为云 Terraform 基础设施即代码（IaC）主程工程师。
请根据提供的华为云官方文档与最佳实践参考信息，所有属性名均来自于参考文档，为用户编写生产级、可直接通过 `terraform validate` 校验的 HCL 代码。
【最高优先级：白名单防幻觉强制规则】
1. 所有你输出的华为云资源，&zwnj;**每一个字段必须100%来自 RAG 参考文档里明确列出的必选/可选属性清单**&zwnj;。
2. 写任何资源前，先从参考文档里找出该资源的完整属性表，不在表里的字段一律禁止写入。
3. 如果你不确定某个字段是否存在，宁可省略不写，也绝对不能自行编造、脑补任何属性名。
4. 禁止使用任何参考文档里没有出现过的资源类型，哪怕你“知道”它在华为云 Provider 里存在，也必须严格以本次提供的 RAG 内容为准。

【工程与语法硬性约束】
1. 严格防幻觉：只能使用“参考文档”中明确声明的资源类型和参数字段，切勿凭空捏造。
2. 跨资源隐式引用：关联资源必须使用 HCL 隐式属性引用语法（如 `vpc_id = huaweicloud_vpc.main.id`），严禁硬编码 UUID。
3. Provider 与版本声明：必须在 `providers.tf` 中正确声明 `required_providers` 与 `provider "huaweicloud"` 块,严禁在 `required_providers` 中写死 `version`（不要出现 `version = "..."` 字段）.
4. 多语言规范：代码中的所有资源标识符、默认名称和参数值必须保持纯英文字符。
5. 标准多文件输出格式：按模块拆分为 `providers.tf`、`variables.tf`、`main.tf`、`outputs.tf`、`terraform.tfvars`。
   每个文件前独占一行使用 `### FILE: 文件名.tf` 作为标记，紧接着输出 ```hcl 代码块。
"""
    # =====================================================================
    # 模式 A：人工协同增量修改模式（从节点 8 带着修改意见回流）
    # =====================================================================
    if human_feedback:
        logger.info(
            "[Generate 版本迭代节点] 正在根据人工审阅意见生成 v%s 增量修改代码...",
            version,
        )
        # 1. 先归档上一版本代码，防止覆盖丢失
        if version > 1:
            archive_previous_version(workspace_dir, version - 1)

        # 2. 读取当前磁盘上的已有完整代码
        previous_files_text = _read_workspace_tf_files(workspace_dir)

        human_prompt = f"""用户审阅了上一版（v{version - 1}）华为云 Terraform 架构代码，并提出了明确的修改需求。
    请你作为资深的华为云 IaC 主程工程师，在**现有代码基础上进行精准的增量修改**（目标版本：v{version}）。

    ### 1. 用户的最新修改指令 (Human Review Feedback)
    👉 **{human_feedback}**

    ### 2. 原始业务需求与参考官方文档 (Context)
    - **原始需求**：{user_query}
    - **参考文档**：
    {rag_context}

    ### 3. 上一版本完整代码 (v{version - 1} Code)
    {previous_files_text}

    ### 4. 增量修改与输出要求 (Strict Constraints)
    1. **局部精准修改**：严格落实用户的修改指令，**保留所有未被提及且已通过校验的现有资源配置、变量定义与隐式引用关系**，切勿无故重写或删减无关架构。
    2. **完整性输出**：必须输出修改后的**完整**工程文件集合（涵盖所有 `.tf` 和 `.tfvars` 文件），严禁使用 `// ... 其余代码省略` 等占位符。
    3. **格式规范**：保持与上一版完全一致的多文件输出标记（`### FILE: xxx.tf` 独占一行，紧跟 ```hcl 代码块），确保程序可直接解析落盘。"""

    # =====================================================================
    # 模式 B：沙箱报错自愈重写模式（从节点 5 带着反思建议回流）
    # =====================================================================
    elif retry_count > 0:
        logger.info(
            "[Generate 修复节点] 正在根据反思建议进行第 %s 次代码重写与修复...",
            retry_count,
        )
        previous_files_text = _read_workspace_tf_files(workspace_dir)

        human_prompt = f"""上一轮 Terraform 校验失败，请你作为资深的华为云（HuaweiCloud）IaC 工程师，根据报错日志、反思建议及上一轮代码进行精准修复。

    ### 1. 沙盒校验报错日志 (Error Log)
    ```text
    {error_message}
    ```

    ### 2. 根因分析与反思指令 (Reflection)
    {reflection_advice}

    ### 3. 上一轮生成的代码 (Previous Code)
    {previous_files_text}

    ### 4. 修复与输出要求 (Strict Constraints)
    1. **针对性修复**：严格对照报错日志中的行号、资源类型、参数名及类型要求进行修正，不要引入无关的改动，保留原本的架构意图。
    2. **沙盒环境约束（极其重要）**：
    - 运行环境为离线沙盒，**严禁**修改 `required_providers` 中的 `source = "huaweicloud/huaweicloud"`。
    - **严禁**随意更改或降级 `version` 约束（如不得改为 `~> 1.47` 等旧版本），必须保持与当前沙盒预装版本一致（例如 `>= 1.100.0`），否则会导致 `terraform init` 失败。
    - 保持 `provider "huaweicloud"` 配置块干净，不要硬编码真实的 `access_key` 或 `secret_key`。
    3. **完整性输出**：输出修复后的**完整** `.tf` 文件内容，不要使用 `// ... 其余代码省略` 等占位符。
    4. **格式规范**：保持与上一轮相同的多文件输出结构（例如使用 `### FILE: xxx.tf` 标记文件名，紧跟 ```hcl 代码块），确保程序可以直接解析并覆盖写入磁盘。"""

    # =====================================================================
    # 模式 C：首次初始从零生成模式
    # =====================================================================
    else:
        logger.info(
            "[Generate 生成节点] 正在基于参考文档编写初始 Terraform 多文件架构..."
        )
        human_prompt = (
            f"""【参考官方文档与最佳实践片段】\n{rag_context}\n\n【用户架构需求】\n{user_query}"""
        )

    # 调用大模型生成代码
    response = llm_coder.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )

    # 物理落盘写入工作区
    saved_files = save_hcl_files_to_workspace(response.content, workspace_dir)
    logger.info(
        "代码 (v%s) 已物理落盘至工作区 [%s]，生成文件列表: %s",
        version,
        workspace_dir,
        saved_files,
    )

    return {
        "generated_files": saved_files,
        "human_feedback": None,  # 🌟 关键：人工指令已消费完毕，立即清空，避免污染后续自愈循环
    }

# ==================== 6. 节点 4：Docker 离线沙箱校验节点 ====================
def sandbox_validate_node(state: TerraformAgentState) -> dict:
    """
    拉起预装华为云插件缓存的 Docker 离线沙箱 (tf-sandbox:huawei)，
    挂载本地 workspace_dir 执行 terraform init 与 terraform validate。
    """
    workspace_dir = state["workspace_dir"]
    retry_count = state.get("retry_count", 0)
    
    logger.info("[Sandbox 沙箱节点] 正在启动离线容器校验工作区: %s...", workspace_dir)

    # Windows 路径兼容处理：将相对路径转为绝对路径，并把反斜杠 '\' 替换为正斜杠 '/'
    abs_workspace = os.path.abspath(workspace_dir).replace("\\", "/")
    ak=os.environ.get("HW_ACCESS_KEY")
    sk=os.environ.get("HW_SECRET_KEY")
    region=os.environ.get("HW_REGION_NAME")

    # 构造 Docker 阅后即焚断网校验命令
    # 注意：由于官方镜像默认 entrypoint 是 terraform，这里用 --entrypoint sh 覆盖，以便连续执行两步操作
    cmd = [
        "docker", "run", "--rm",
        # "--network", "none",
        "-v", f"{abs_workspace}:/workspace",
        "-e", f"HW_ACCESS_KEY={ak}",
        "-e", f"HW_SECRET_KEY={sk}",
        "-e", f"HW_REGION_NAME={region}",
        "--entrypoint", "sh",
        "tf-sandbox:huawei",
        "-c", (
            "terraform init -backend=false -input=false -no-color -plugin-dir=/root/.terraform.d/plugin-cache > /dev/null && "
            "terraform validate -no-color > /dev/null && "
            "terraform plan -input=false -lock=false -no-color -out=tfplan  > /dev/null &&"
            "terraform show -json tfplan > /workspace/plan.json && "
            "opa eval --data /app/policies/security.rego --input /workspace/plan.json 'data.terraform.security.deny' --format json"
        )
    ]

    try:
        # 设置 30 秒超时保护，防止极端异常导致挂起
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=90
        )
        # 获取完整的标准输出与错误输出
        raw_output = process.stdout.strip()
        raw_error = process.stderr.strip()
        # 1. 检查 Terraform 基础设施命令（init/validate/plan）本身是否崩了
        if process.returncode != 0:
            # 发生 Terraform 语法错误、凭证不对或参数传错，走反思节点自我修正
            full_error_msg = f"{raw_error}\n{raw_output}".strip()
            logger.error("[Terraform 执行失败] 沙箱检测到配置错误，准备交由反思节点分析...")
            return {
                "validation_passed": False,
                "security_passed": False,
                "error_message": full_error_msg,
                "retry_count": retry_count + 1,
            }
        else:
            # 2. Terraform 执行成功，接着解析最后一步 OPA 的返回结果
            logger.info("[Terraform 基础校验] 初始化、语法及 Plan 执行全部通过！")

            try:
                # 尝试将标准输出解析为 OPA 的 JSON 结果
                opa_output = json.loads(raw_output)
                violations = []

                # OPA eval 返回值的标准嵌套结构解析
                if "result" in opa_output and len(opa_output["result"]) > 0:
                    expressions = opa_output["result"][0].get("expressions", [])
                    if expressions:
                        violations = expressions[0].get("value", [])

                    if violations:
                    # OPA 扫描到了合规违规项（安全卡点拦截）
                        logger.warning(
                            "[OPA 安全合规拦截] 发现 %s 处违规风险，准备交由反思/提示节点处理...",
                            len(violations),
                        )
                    formatted_violations = "\n".join([f"- {v}" for v in violations])
                    return {
                        "validation_passed": True,  # Terraform 语法没问题，但是安全没过
                        "security_passed": False,
                        "error_message": (
                            f"【OPA 安全合规检查未通过】:\n{formatted_violations}"
                        ),
                        "retry_count": retry_count + 1,
                    }
                else:
                    # Terraform 合法，且 OPA 检查 0 违规
                    logger.info("[安全全通] Terraform 语法合法 且 OPA 华为云合规硬校验完美通过！")
                    return {
                        "validation_passed": True,
                        "security_passed": True,
                        "error_message": None,
                    }

            except json.JSONDecodeError:
                # 如果输出不是合法的 JSON（理论上不该发生，除非 shell 脚本其他地方打印了杂音）
                logger.warning("[警告] 无法解析 OPA 的 JSON 输出，原始输出内容：%s", raw_output[:500])
                return {
                    "validation_passed": True,
                    "security_passed": False,
                    "error_message": (
                        f"OPA 扫描结果解析异常，原始输出: {raw_output[-300:]}"
                    ),
                    "retry_count": retry_count + 1,
                }

    except subprocess.TimeoutExpired:
        logger.error("[沙箱超时] Docker 容器执行超过 90 秒。")
        return {
            "validation_passed": False,
            "security_passed": False,
            "error_message": "Docker 沙箱执行超时，请检查是否存在死循环或异常结构。",
            "retry_count": retry_count + 1
        }
    except Exception as e:
        logger.error("[系统异常] 调用 Docker 失败: %s", e)
        return {
            "validation_passed": False,
            "security_passed": False,
            "error_message": f"Docker 调用异常: {str(e)}",
            "retry_count": retry_count + 1
        }


# ==================== 7. 节点 5：架构师反思与纠错节点 ====================
class ReflectDiagnosis(BaseModel):
    error_category: Literal["code_or_opa_issue", "env_or_network_issue"] = Field(
        description=(
            "错误归因分类："
            "'code_or_opa_issue' 表示 HCL语法错误、参数写错、资源引用不对、资源配置不符合云端规则或 OPA 安全策略违规，可以通过修改 .tf 代码修复；"
            "'env_or_network_issue' 表示沙箱断网、下载 Provider 依赖包超时/失败、State 锁冲突、云账号 AK/SK 鉴权失败或配额/库存不足，无法通过修改代码解决。"
        )
    )
    is_fixable_by_agent: bool = Field(
        description="如果 error_category 为 'code_or_opa_issue' 则为 True；如果是 'env_or_network_issue' 则为 False。"
    )
    reflection_advice: str = Field(
        description=(
            "如果 is_fixable_by_agent 为 True，请输出精准、分条的 .tf 代码修改指令（指明哪个文件/资源块删掉什么、改成什么）；"
            "如果为 False，请简要说明环境/网络异常原因，供人工运维排查。"
        )
    )
def reflect_node(state: TerraformAgentState) -> dict:
    """
    对沙箱报错进行深度根因分析 (Root Cause Analysis)，
    读取工作区报错代码、结合报错堆栈与原始文档，输出可执行的精准修复建议。
    由 AI 自主对沙箱报错进行归因分类 (Root Cause Classification & Analysis)：
    - 若属于代码/逻辑/OPA合规问题 -> 生成修复指令，流转回 generate_code 节点
    - 若属于沙箱网络/依赖包/账号权限等环境问题 -> 标记需人工介入
    """
    workspace_dir = state["workspace_dir"]
    error_message = state.get("error_message", "未知错误")
    rag_context = state.get("rag_context", "")
    retry_count = state.get("retry_count", 1)

    logger.info("[Reflect 反思节点] 正在诊断第 %s 次校验报错原因...", retry_count)

    # 按需从本地工作区硬盘读取当前的 .tf 源码文件（不长期占用内存）
    current_files_content = []
    if os.path.exists(workspace_dir):
        for fname in sorted(os.listdir(workspace_dir)):
            if fname.endswith(".tf"):
                fpath = os.path.join(workspace_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    current_files_content.append(f"### FILE: {fname}\n```hcl\n{f.read().strip()}\n```")
    
    files_text = "\n\n".join(current_files_content)

    system_prompt = """你是一个顶级的华为云 Terraform 排错专家与首席架构师。
初级工程师编写的 Terraform 代码在沙箱执行 `terraform validate` 时报错了。
请你严格对比【沙箱原始报错日志】、【当前工作区源码】以及【华为云官方参考文档】，找出错误根因并给出精准的修复指令。
你的首要任务是【精准归因分类】：
1. **判断是否可以通过修改 `.tf` 代码解决 (`is_fixable_by_agent`)**：
   - **可以解决 (`True`)**：HCL 语法错误、华为云资源字段名幻觉/过时、跨资源引用路径错误、缺少条件必填项、CIDR/命名规范违规、以及 OPA 安全合规策略拒绝（如端口暴露、未加密、缺标签）。
   - **无法通过改代码解决 (`False`)**：沙箱网络不通/超时、Provider 依赖包下载失败、Terraform State 锁冲突、磁盘空间不足、华为云账号 AK/SK 鉴权失败（403）、账号配额不足或可用区库存售罄。
2. **输出对应指导 (`reflection_advice`)**：
   - 若为 `True`：严格对比【沙箱原始报错日志】、【当前工作区源码】和【华为云官方参考文档】，直接给出清晰、分条的修改指令，明确指出“哪个文件的哪个资源块”应该“删掉什么、改成什么”（注意修复 OPA 策略时不要破坏原有合法字段），无需重写全部代码，把修改指令精准传达给主程工程师即可。
   - 若为 `False`：用一两句话概括环境或网络故障原因，明确提示需要人工介入排查沙箱或云账号。
"""
    human_prompt = f"""【沙箱原始报错日志】
{error_message}

【当前工作区代码】
{files_text}

【华为云官方参考文档】
{rag_context}

请给出精准的错误根因分析与修改指令："""
    structured_llm = llm_reflector.with_structured_output(ReflectDiagnosis)
    diagnosis: ReflectDiagnosis = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    if not diagnosis.is_fixable_by_agent:
       logger.warning("[AI 判定：环境/网络异常] 无法通过修改代码修复，终止任务转交人工。")
       return {
            "needs_manual_intervention": True,
            "reflection_advice": f"【人工介入通知】{diagnosis.reflection_advice}"
        }
    logger.info("[反思完成] 已生成精准修复指令，准备交回生成节点重写。")
    return {
        "needs_manual_intervention": False,
        "reflection_advice": diagnosis.reflection_advice
    }
    
# ==================== 8. 节点 6：普通概念问答兜底节点 ====================
def general_qa_node(state: TerraformAgentState) -> dict:
    """
    当路由节点判定用户只是在咨询纯理论概念、名词解释，不需要生成实际工程代码时，
    给出专业解答，支持轻量级RAG参考。
    """
    user_query = state["user_query"]
    logger.info("[General QA 问答节点] 正在检索知识库并解答概念咨询...")

    # 1 轻量级知识库检索，只取top3避免噪声
    retrieved_docs = ""
    try:
        # 替换为你自己的向量库检索函数，例如 vector_store.similarity_search_with_score(user_query, k=3)
        docs = simple_retrieve(user_query,1)
        if docs and docs[0]:
            retrieved_docs = "\n\n".join(docs[0])
            logger.info("已命中 %s 条相关知识库片段，将结合官方文档作答。", len(docs[0]))
    except Exception as e:
        logger.warning("知识库检索跳过，降级为纯模型解答: %s", e)

    # 2 组装Prompt
    if retrieved_docs:
        system_prompt = """你是一个资深的华为云与 Terraform 基础设施架构师。
    请优先参考提供的【华为云官方知识库片段】来解答用户的概念或架构咨询。
    要求：
    1. 事实准确：涉及华为云服务特性、限制或参数说明时，严格以参考文档为准，切勿编造。
    2. 结构清晰：用通俗易懂的中文分点解答，必要时可用简短的 HCL 代码块辅助说明。
    3. 如果参考文档未能覆盖用户的部分通用问题，可结合你的专业知识进行补充。"""

        human_prompt = f"""【华为云官方知识库片段】 {retrieved_docs}
    【用户咨询问题】{user_query}"""
    else:
        system_prompt = """你是一个资深的华为云与 Terraform 基础设施架构师。
    用户正在向你咨询关于 Terraform 或华为云的理论概念、最佳实践原则或名词解释。
    请用清晰、结构化、通俗易懂的中文直接回答用户的问题。如果涉及简单的概念性 HCL 语法片段示例，可以用 Markdown 代码块展示，但无需生成完整的工程项目目录结构。"""
        human_prompt = user_query

    response = llm_router.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    return {
       "rag_context": retrieved_docs,
       "final_response": response.content.strip()
    }

# ==================== 9. 节点 7：LLM 语义风险解读节点 ====================
def risk_analyze_node(state: TerraformAgentState)->dict:
  """读取沙箱生成的 plan.json，提取核心 Diff 并交由大模型生成中文高危变更语义风险解读报告。"""
  logger.info("[Risk Analyze 语义风险节点] 正在提取 Plan Diff 并生成中文风险解读报告...")

  workspace_dir = state.get("workspace_dir")
  plan_json_path = os.path.join(workspace_dir, "plan.json")
  validation_passed = state.get("validation_passed", False)
  security_passed = state.get("security_passed", True)
  error_message = state.get("error_message", "")

  # 1. 前置检查：如果是重试超限触发熔断进入本节点的，直接生成降级告警报告
  if not validation_passed:
    logger.warning("检测到沙箱基础校验未通过（已触发熔断），生成高危人工接管提示。")
    fallback_report = (
        "### 🔴 沙箱校验未通过（已触发安全熔断）\n\n"
        "- **风险等级**：`HIGH`（高危 - 代码存在语法或合规阻断项）\n"
        "- **OPA 安全合规检测**："
        f"{'✅ 通过' if security_passed else '❌ 未通过'}\n"
        f"- **沙箱拦截/报错日志**：\n```text\n{error_message}\n```\n"
        "请人工介入检查工作区代码，或输入修改指令让系统重新生成。"
    )
    return {"risk_report": fallback_report, "risk_level": "HIGH"}

  if not os.path.exists(plan_json_path):
    return {"risk_report": "⚠️ 未检测到有效的 plan.json 文件，跳过语义风险分析。"}

  # 2. 提取精简 Diff 供大模型阅读
  diffs = extract_terraform_diff(plan_json_path)
  if not diffs:
    return {"risk_report": "✅ 本次变更不涉及任何实际资源改动（No Changes）。"}

  # 3. 组装 Prompt
  diff_str = json.dumps(diffs, indent=2, ensure_ascii=False)
  prompt = f"""
    你是一个资深的华为云 SecOps（安全运维）与云原生架构专家。
现在需要对一份即将应用到华为云生产环境的 Terraform 变更计划（Plan Diff）进行深度语义风险解读。


请从以下几个维度进行专业、严谨的中文化风险评估，并以结构化 Markdown 格式输出：
1. 📊 **变更概览**：统计本次新增 (Create)、修改 (Update)、销毁 (Destroy) 的资源数量及核心资产。
2. 🔴 **高危风险与破坏性影响**：识别是否有高危变更（如生产数据库删除、公网高危端口暴露、安全组大面积放通、云服务器重启等），并指出潜在的业务影响。
3. 🟡 **配置变更盲点**：指出一些虽然没有违规但值得注意的架构配置风险（如未开启删除保护、规格变更带来的停机风险等）。
4. 💡 **优化建议**：给出针对性的安全加固或修改建议。

语气要求：专业、客观、严厉指出安全隐患。
    """
  human_prompt = f"""
以下是本次变更的核心摘要信息：
{diff_str}"""

  # 3. 调用 LLM
  try:
    response = llm_reflector.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=human_prompt),
    ])
    
    # 兼容 LangChain AIMessage (.content) 与原生 OpenAI SDK (.choices)
    raw_content = (
        response.content
        if hasattr(response, "content")
        else response.choices[0].message.content
    )

    # 提取第一行的风险等级标签
    risk_level = "MEDIUM"  # 默认兜底等级
    match = re.search(
        r"\[RISK_LEVEL:\s*(LOW|MEDIUM|HIGH)\]", raw_content, re.IGNORECASE
    )
    if match:
      risk_level = match.group(1).upper()
      # 将标签行从最终展示报告中剔除，保持报告排版整洁
      risk_report = re.sub(
          r"\[RISK_LEVEL:\s*(LOW|MEDIUM|HIGH)\]\s*",
          "",
          raw_content,
          count=1,
          flags=re.IGNORECASE,
      ).strip()
    else:
      risk_report = raw_content.strip()

    # 如果 OPA 策略本身未通过，强制将最终风险等级兜底提升为 HIGH
    if not security_passed:
      risk_level = "HIGH"

    logger.info("语义风险报告生成完毕 [评估风险等级: %s]，即将进入人工审阅节点。", risk_level)
    return {"risk_report": risk_report, "risk_level": risk_level}

  except Exception as e:
    logger.warning("风险报告生成失败: %s", e)
    return {
        "risk_report": f"⚠️ 风险报告生成异常: {str(e)}",
        "risk_level": "MEDIUM",
    }

# ==================== 10. 节点 8：人机协同与版本修改节点 ====================
def human_review_node(state: TerraformAgentState) -> dict:
  """暂停工作流，挂起流程，向用户展示代码、OPA结果与LLM风险报告，等待审批或修改指令"""
  current_version = state.get("version", 1)
  # 触发 LangGraph 中断，将当前关键信息抛给前端/控制台
  user_feedback = interrupt({
      "message": (
          f"请审阅生成的 Terraform 代码 (v{current_version}) 与风险分析报告"
      ),
      "version": current_version,
      "risk_level": state.get("risk_level"),
      "validation_passed": state.get("validation_passed"),
      "security_passed": state.get("security_passed"),
      "risk_report": state.get("risk_report"),
      "workspace_dir": state.get("workspace_dir"),
      "generated_files": state.get("generated_files", []),
  })

  # 当外部通过 Command(resume=...) 恢复执行时，user_feedback 会接收到用户输入
  # 假设输入格式为 {"action": "approve"} 或 {"action": "modify", "feedback": "把带宽改成10M"}
  action = user_feedback.get("action", "approve")
  feedback_text = user_feedback.get("feedback", "")

  if action == "modify":
    return {
        "human_action": "modify",
        "human_feedback": feedback_text,
        "version": current_version + 1,  # 版本号递增，进入下一版迭代
        "retry_count": 0,  # 重置重试计数器，给新一轮修改完整的自愈机会
        "reflection_advice": None,  # 清空上一版本的反思意见
        "error_message": None,  # 清空上一版本的报错信息
    }

  return {
      "human_action": "approve",
      "human_feedback": None,
  }