# 定义LangGraph 全局状态图
# 定义一个 TypedDict 字典，里面装着 user_query（用户需求）、rag_context（查到的文档）、generated_hcl（当前生成的代码）、error_message（Docker 报错日志）和 retry_count（重试计数器）。所有节点都围绕读写这个记事本工作。


# 问题：用户要求一个涵盖非常多资源的云架构怎么办？
# 答：1、前置资源清单拆解，target_resources:List[str]; 2、状态覆写而非历史堆积，仅保留最近一版代码和最新一次报错；3、精细化反思把全量文档和修改建议分离。
from typing import TypedDict, List, Optional

class TerraformAgentState(TypedDict):
    """
    LangGraph 全局共享状态机
    代码实体和plan.json直接落地到 workspace_dir 硬盘目录，内存中仅传递路径与元数据，彻底杜绝内存膨胀
    """
    # 1. 任务与工作区物理路径
    task_id: str                         # 唯一任务编号，用于隔离并发工作区
    workspace_dir: str                   # 🌟 核心内存优化：本地物理工作区路径 (如 "./workspaces/task_001")
    user_query: str                      # 用户的原始自然语言需求
    
    # 2. 路由与任务拆解
    intent: str                          # 意图路由标识: "generate_tf" 或 "general_qa"
    target_resources: List[str]          # 拆解出的目标华为云资源清单，用于按需检索与依赖统筹
    
    # 3. 知识检索层
    rag_context: str                     # 按资源清单精准检索并 Rerank 后的精简上下文
    
    # 4. 沙箱验证与状态指针
    generated_files: List[str]           # 🌟 当前工作区内已生成的文件名列表 (如 ["providers.tf", "main.tf", "variables.tf"])
    validation_passed: bool              # Docker 沙箱 terraform validate 是否通过
    security_passed: bool                # OPA检测
    error_message: Optional[str]         # 最近一次沙箱校验失败的原始报错日志
    
    # 5. 反思修正与熔断控制
    reflection_advice: Optional[str]     # 反思节点 (qwen-max) 针对报错给出的精准修改指令
    retry_count: int                     # 当前自动修正重试次数（防死循环）
    needs_manual_intervention: bool         # 【新增】是否遇到环境/网络/依赖包等非代码问题，需跳出交由人工处理

    # 6. LLM语义解读
    risk_report: Optional[str] # 节点 7 读取硬盘上的 plan.json 后，生成的语义风险与资源变更解读报告
    risk_level: Optional[str]  # 风险等级标识: "LOW" | "MEDIUM" | "HIGH"

    # 7. HITL 
    version: int  # 当前代码版本号（初始为 1，每次人工打回修改后 +1）
    human_action: Optional[
        str
    ]  # 人类审批动作: "approve"（确认交付）或 "modify"（提出修改意见）
    human_feedback: Optional[str]  # 人类在节点 8 输入的具体修改意见或补充需求

    # 8. 终态
    final_response: str                  # 最终输出给用户的交付总结说明
