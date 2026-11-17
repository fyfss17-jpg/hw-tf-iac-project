import os
import time

from langgraph.types import Command

from agent_core.graph import build_terraform_agent_graph
from agent_core.services.workspace_service import WorkspaceService

workspace_service = WorkspaceService()


def write_task_log(workspace_dir: str, message: str):
    """Write a plain-text progress log into the task workspace and print it to console."""
    log_path = os.path.join(workspace_dir, "task_logic.log")
    print(message)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")
    return log_path


def run_agent_task(user_query: str):
    """Create a task workspace and run the LangGraph workflow."""
    task_id, workspace_dir = workspace_service.create_task_workspace()
    log_path = os.path.join(workspace_dir, "task_logic.log")

    write_task_log(workspace_dir, "=" * 60)
    write_task_log(workspace_dir, f"🚀 启动华为云 Terraform 智能体 | 任务 ID: {task_id}")
    write_task_log(workspace_dir, f"📁 独立工作区已就绪: {workspace_dir}")
    write_task_log(workspace_dir, f"📄 逻辑日志文件: {log_path}")
    write_task_log(workspace_dir, "=" * 60)

    agent_app = build_terraform_agent_graph()

    initial_state = {
        "task_id": task_id,
        "workspace_dir": workspace_dir,
        "user_query": user_query,
        "intent": "",
        "target_resources": [],
        "rag_context": "",
        "generated_files": [],
        "validation_passed": False,
        "security_passed": True,
        "error_message": None,
        "reflection_advice": None,
        "retry_count": 0,
        "risk_report": None,
        "risk_level": None,
        "version": 1,
        "human_action": None,
        "human_feedback": None,
        "final_response": "",
    }

    config = {"configurable": {"thread_id": task_id}}
    current_state = agent_app.invoke(initial_state, config=config)

    while current_state.get("__interrupt__"):
        interrupt_payload = current_state["__interrupt__"]
        interrupt_info = interrupt_payload[0].value if isinstance(interrupt_payload, list) else interrupt_payload

        write_task_log(workspace_dir, "\n" + "⏸️ " * 20)
        write_task_log(workspace_dir, f"【人机协同审批】{interrupt_info.get('message')}")
        write_task_log(
            workspace_dir,
            f"📦 代码版本: v{interrupt_info.get('version')} | 文件列表: {interrupt_info.get('generated_files')}",
        )
        write_task_log(
            workspace_dir,
            f"🛡️  沙箱校验: {interrupt_info.get('validation_passed')} | OPA合规: {interrupt_info.get('security_passed')} | 风险等级: {interrupt_info.get('risk_level')}",
        )
        write_task_log(workspace_dir, f"📂 本地路径: {os.path.abspath(workspace_dir)}")
        write_task_log(workspace_dir, "-" * 60)
        write_task_log(workspace_dir, f"📊 LLM 语义风险解读报告:\n{interrupt_info.get('risk_report')}")
        write_task_log(workspace_dir, "⏸️ " * 20)

        review_input = input(
            "\n👉 请输入审批指令（直接回车或输入 ok 确认交付；输入具体意见则打回修改代码）：\n> "
        ).strip()

        if review_input.lower() in ["", "ok", "yes", "y", "approve", "通过", "确认"]:
            resume_payload = {"action": "approve"}
        else:
            resume_payload = {"action": "modify", "feedback": review_input}

        current_state = agent_app.invoke(Command(resume=resume_payload), config=config)

    write_task_log(workspace_dir, "\n" + "=" * 60)
    write_task_log(workspace_dir, "🏁 任务执行完毕！最终交付报告：")
    write_task_log(workspace_dir, "=" * 60)
    write_task_log(workspace_dir, current_state.get("final_response", "未生成有效回复"))


def run_cli():
    """Interactive CLI entrypoint."""
    if "DASHSCOPE_API_KEY" not in os.environ:
        print("❌ 错误：未检测到 DASHSCOPE_API_KEY 环境变量，请先配置 API Key！")
        raise SystemExit(1)

    print("华为云 Terraform 架构生成 Agent 已启动（输入 'exit' 退出）")

    default_test_query = (
        "帮我搭一套华为云基础网络和计算环境：需要一个自定义网段的VPC，下面挂一个子网，"
        "再配一个允许SSH(22端口)和HTTP(80端口)访问的安全组，最后创建一台按需计费的ECS云服务器。"
    )

    while True:
        user_input = input("\n请输入你的架构需求（直接回车将使用默认复杂场景测试）：\n> ").strip()
        start_time = time.perf_counter()
        if user_input.lower() in ["exit", "quit"]:
            break

        query_to_run = user_input if user_input else default_test_query
        run_agent_task(query_to_run)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"执行耗时: {duration:.6f} 秒")
