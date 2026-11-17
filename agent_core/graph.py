# 组装节点、连接条件边 (Edges)，编译出最终的 Agent 图
# 使用 LangGraph 的 StateGraph 把上面 5 个节点连起来，并设置两个关键分支判断（Conditional Edges）：
# 	• 路由后：是写代码 ➔ 进检索；只是问概念 ➔ 直接回答结束。
# 	• 沙箱校验后：校验通过 ➔ 完美输出结束；校验失败且重试 < 3 次 ➔ 进反思节点再回炉重造；重试 >= 3 次 ➔ 熔断结束并提示人工介入。

# router ➔ retrieve ➔ generate ➔ sandbox_validate ➔ (若报错) reflect ➔ generate ➔ sandbox_validate ➔ (若通过) summarize ➔ END

import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent_core.state import TerraformAgentState
from agent_core.nodes import (
    router_node,
    retrieve_node,
    generate_node,
    sandbox_validate_node,
    reflect_node,
    general_qa_node,
    risk_analyze_node,
    human_review_node,
)

# ==================== 1. 新增收尾总结节点 ====================
def summarize_node(state: TerraformAgentState) -> dict:
    """
    当 Docker 沙箱校验彻底通过，或者达到最大重试上限触发熔断时，
    生成最终的交付物总结说明，告知用户代码存放的本地物理路径及校验状态。
    """
    workspace_dir = state["workspace_dir"]
    generated_files = state.get("generated_files", [])
    validation_passed = state.get("validation_passed", False)
    retry_count = state.get("retry_count", 0)
    error_message = state.get("error_message", "")

    if validation_passed:
        summary = (
            f"🎉 **华为云 Terraform 架构代码已生成并通过离线沙箱严格校验！**\n\n"
            f"- **本地交付目录**：`{os.path.abspath(workspace_dir)}`\n"
            f"- **包含工程文件**：`{', '.join(generated_files)}`\n"
            f"- **自动修正轮次**：历经 {retry_count} 次沙箱自愈迭代。\n\n"
            f"你可以直接进入该目录，配置好华为云 AK/SK 环境变量后执行 `terraform plan` 和 `terraform apply` 进行部署。"
        )
    else:
        summary = (
            f"⚠️ **代码已生成，但在连续 {retry_count} 次自动修正后仍未完全通过沙箱校验（已触发安全熔断）。**\n\n"
            f"- **本地代码目录**：`{os.path.abspath(workspace_dir)}`\n"
            f"- **包含工程文件**：`{', '.join(generated_files)}`\n"
            f"- **最后一次报错日志**：\n```text\n{error_message}\n```\n"
            f"建议你打开上述目录中的 `.tf` 文件，结合报错信息进行微调。"
        )

    return {"final_response": summary}


# ==================== 2. 定义动态条件路由函数 ====================
def route_after_intent(state: TerraformAgentState) -> str:
    """第一道分流：判断是走代码工程流，还是走纯概念问答流"""
    if state.get("intent") == "general_qa":
        return "general_qa"
    return "retrieve"


def route_after_validation(state: TerraformAgentState) -> str:
    """第二道分流：判断沙箱校验结果，决定是交付、自我修正还是熔断"""
    if state.get("validation_passed"):
        return "human_review"
    
    # 设置最大重试上限为 3 次，防止陷入死循环烧 Token
    MAX_RETRIES = 3
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "reflect"
    else:
        print(f"\n🛑 [熔断触发] 已达到最大自动修正次数 ({MAX_RETRIES} 次)，停止重试并输出报告。")
        return "risk_analyze"
def router_after_reflect(state: TerraformAgentState) -> str:
    """第三道分流：判断是否需要重新生成代码"""
    if state.get("needs_manual_intervention"):
        return "human_review"
    return "generate"

def route_after_human(state: TerraformAgentState) -> str:
  """第四道分流：人机协同决策 -> 结束交付 或 带着修改意见重新生成"""
  human_action = state.get("human_action")  # 例如: "approve" | "modify"
  if human_action == "modify":
    # 带着用户的修改意见回到代码生成节点（若涉及新资源需求，也可路由回 retrieve）
    return "generate"
  return 'summarize'
# ==================== 3. 组装并编译 LangGraph 状态图 ====================
def build_terraform_agent_graph():
    """构建并返回编译好的 LangGraph 可执行应用实例"""
    workflow = StateGraph(TerraformAgentState)

    # 1. 注册所有工作节点
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("sandbox_validate", sandbox_validate_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("general_qa", general_qa_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("risk_analyze", risk_analyze_node)
    workflow.add_node("human_review", human_review_node)

    # 2. 设置入口节点
    workflow.set_entry_point("router")

    # 3. 添加第一道条件边（意图分流）
    workflow.add_conditional_edges(
        "router",
        route_after_intent,
        {
            "general_qa": "general_qa",
            "retrieve": "retrieve"
        }
    )

    # 4. 连接标准工程主干线
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "sandbox_validate")

    # 5. 添加第二道条件边（沙箱校验 -> 自愈闭环或交付）
    workflow.add_conditional_edges(
        "sandbox_validate",
        route_after_validation,
        {
            "risk_analyze": "risk_analyze",
            "reflect": "reflect",
            "human_review": "human_review"
        }
    )

    # 6. 连接自我修正回流线（反思 -> 重新生成）
    workflow.add_conditional_edges(
        "reflect",
        router_after_reflect,
        {"generate": "generate", "human_review": "human_review"},
    )
    # 7. 风险解读完成后，进入人机协同节点
    workflow.add_edge("risk_analyze", "human_review")
    workflow.add_conditional_edges(
        "human_review",
        route_after_human,
        {"generate": "generate", "summarize": "summarize"},
    )

    # 8. 设置终点
    workflow.add_edge("general_qa", END)
    workflow.add_edge("summarize", END)


    # 编译生成最终的可调用图应用
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
