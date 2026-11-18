from types import SimpleNamespace
from unittest.mock import Mock, patch
import uuid

from langchain_core.messages import AIMessage
import os
import sys
# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取隔壁文件夹（folderB）的路径
sibling_dir = os.path.join(current_dir, '..')

# 将隔壁文件夹加入到环境变量 sys.path 中
sys.path.append(os.path.abspath(sibling_dir))
from agent_core import graph as graph_module
from agent_core import nodes


def test_full_script_generation_workflow(tmp_path, monkeypatch):
    """
    端到端测试说明：
    - Mock LLM 避免真实 API 调用，并固定路由与代码输出。
    - Mock RAG 验证检索节点确实被调用，即使知识库返回空也不能跳过调用。
    - Mock Sandbox 覆盖 Docker 外部依赖，只验证节点状态变化。
    - Mock Human Review 模拟低风险人工批准，避免 LangGraph interrupt 阻塞测试。
    """
    user_query = (
        "编写一个 Python 脚本，读取当前目录下的 data.csv "
        "并计算平均值。"
    )

    router_llm = Mock(
        invoke=Mock(
            return_value=AIMessage(
                content=(
                    '{"intent": "generate_tf", '
                    '"target_resources": ["huaweicloud_vpc"]}'
                )
            )
        )
    )

    python_script = (
        "import csv\n"
        "\n"
        "with open('data.csv', newline='', encoding='utf-8') as f:\n"
        "    rows = csv.DictReader(f)\n"
        "    values = [float(row['value']) for row in rows]\n"
        "\n"
        "print(sum(values) / len(values) if values else 0)\n"
    )

    coder_output = (
        "### FILE: main.tf\n"
        "```hcl\n"
        'resource "huaweicloud_vpc" "main" { cidr = "192.168.0.0/16" }\n'
        "```\n"
        "\n"
        "```text\n"
        f"Python script test fixture:\n{python_script}"
        "```\n"
    )

    coder_llm = Mock(
        invoke=Mock(
            return_value=AIMessage(content=coder_output)
        )
    )

    rag_mock = Mock(
        return_value="",
        side_effect=lambda user_query, target_resources: "",
    )

    sandbox_mock = Mock(
        return_value={
            "validation_passed": True,
            "security_passed": True,
            "error_message": None,
            "retry_count": 0,
        }
    )

    human_review_mock = Mock(
        return_value={
            "human_action": "approve",
            "human_feedback": None,
            "risk_level": "LOW",
        }
    )

    def mock_summarize(state):
        """测试替身：用于验证最终交付内容同时包含代码及执行结果。"""
        code_block = f"```python\n{python_script}```"
        execution_result = "执行结果：平均值为 42.0"
        return {
            "final_response": (
                "低风险模式已通过自动审批。\n\n"
                f"{code_block}\n\n"
                f"{execution_result}"
            )
        }

    monkeypatch.setattr(nodes, "llm_router", router_llm)
    monkeypatch.setattr(nodes, "llm_coder", coder_llm)
    monkeypatch.setattr(nodes, "retrieve_tf_knowledge", rag_mock)
    monkeypatch.setattr(nodes, "sandbox_validate_node", sandbox_mock)
    monkeypatch.setattr(nodes, "human_review_node", human_review_mock)

    # build_terraform_agent_graph 使用模块中注册的函数对象，
    # 因此在编译前替换 graph_module 中的节点与总结函数。
    monkeypatch.setattr(graph_module, "router_node", nodes.router_node)
    monkeypatch.setattr(graph_module, "retrieve_node", nodes.retrieve_node)
    monkeypatch.setattr(graph_module, "generate_node", nodes.generate_node)
    monkeypatch.setattr(graph_module, "sandbox_validate_node", sandbox_mock)
    monkeypatch.setattr(graph_module, "human_review_node", human_review_mock)
    monkeypatch.setattr(graph_module, "summarize_node", mock_summarize)
    monkeypatch.setattr(graph_module, "risk_analyze_node", nodes.risk_analyze_node)
    monkeypatch.setattr(graph_module, "reflect_node", nodes.reflect_node)
    monkeypatch.setattr(graph_module, "general_qa_node", nodes.general_qa_node)

    app = graph_module.build_terraform_agent_graph()

    initial_state = {
        "task_id": f"e2e-{uuid.uuid4()}",
        "workspace_dir": str(tmp_path),
        "user_query": user_query,
        "intent": "",
        "target_resources": [],
        "rag_context": "",
        "generated_files": [],
        "validation_passed": False,
        "security_passed": False,
        "error_message": None,
        "reflection_advice": None,
        "retry_count": 0,
        "needs_manual_intervention": False,
        "risk_report": None,
        "risk_level": "LOW",
        "version": 1,
        "human_action": None,
        "human_feedback": None,
        "final_response": "",
    }

    final_state = app.invoke(
        initial_state,
        {"configurable": {"thread_id": str(uuid.uuid4())}},
    )

    assert final_state["intent"] == "generate_tf"
    assert final_state["target_resources"] == ["huaweicloud_vpc"]

    rag_mock.assert_called()
    rag_mock.assert_called_once_with(
        user_query=user_query,
        target_resources=["huaweicloud_vpc"],
    )

    sandbox_mock.assert_called_once()
    human_review_mock.assert_called_once()

    assert final_state["human_action"] == "approve"
    assert final_state["validation_passed"] is True
    assert final_state["security_passed"] is True

    final_response = final_state["final_response"]
    assert "```python" in final_response
    assert "读取当前目录下的 data.csv" not in final_response
    assert "sum(values) / len(values)" in final_response
    assert "平均值为 42.0" in final_response
    assert "低风险模式已通过自动审批" in final_response