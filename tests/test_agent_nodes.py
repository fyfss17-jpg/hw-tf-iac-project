import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage
import os
import sys
# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取隔壁文件夹（folderB）的路径
sibling_dir = os.path.join(current_dir, '..')

# 将隔壁文件夹加入到环境变量 sys.path 中
sys.path.append(os.path.abspath(sibling_dir))
from agent_core import nodes
from agent_core import rag_retriever
from agent_core.graph import (
    route_after_human,
    route_after_intent,
    route_after_validation,
    router_after_reflect,
)


class StructuredDiagnosis:
    def __init__(self, *, fixable=True, advice=""):
        self.is_fixable_by_agent = fixable
        self.reflection_advice = advice


# ==================== Router 单元测试 ====================

@pytest.mark.parametrize(
    ("user_query", "mock_json", "expected_intent"),
    [
        (
            "写个爬虫",
            '{"intent": "generate_tf", "target_resources": []}',
            "generate_tf",
        ),
        (
            "你好",
            '{"intent": "general_qa", "target_resources": []}',
            "general_qa",
        ),
    ],
)
def test_router_intent(user_query, mock_json, expected_intent, base_state):
    base_state["user_query"] = user_query
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=mock_json)

    with patch.object(nodes, "llm_router", llm):
        result = nodes.router_node(base_state)

    assert result["intent"] == expected_intent
    llm.invoke.assert_called_once()

    route = (
        "general_qa"
        if result["intent"] == "general_qa"
        else "retrieve"
    )
    assert route == (
        "general_qa"
        if expected_intent == "general_qa"
        else "retrieve"
    )


def test_router_json_parse_failure_defaults_to_generate(base_state):
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="not-json")

    with patch.object(nodes, "llm_router", llm):
        result = nodes.router_node(base_state)

    assert result["intent"] == "generate_tf"
    assert result["target_resources"] == []


# ==================== RAG 单元测试 ====================

def test_retrieve_node_calls_rag_and_preserves_empty_result(
    base_state,
    monkeypatch,
):
    rag_mock = Mock(return_value="")
    monkeypatch.setattr(nodes, "retrieve_tf_knowledge", rag_mock)

    result = nodes.retrieve_node(base_state)

    rag_mock.assert_called_once_with(
        user_query="创建一个华为云 VPC",
        target_resources=["huaweicloud_vpc"],
    )
    assert result["rag_context"] == ""


def test_retrieve_node_filters_and_formats_ranked_context(
    base_state,
    monkeypatch,
):
    base_state["target_resources"] = ["huaweicloud_vpc"]

    rag_mock = Mock(
        return_value=(
            "--- [来源: 资源文档：huaweicloud_vpc] ---\n"
            "resource \"huaweicloud_vpc\" \"main\" { cidr = \"192.168.0.0/16\" }"
        )
    )
    monkeypatch.setattr(nodes, "retrieve_tf_knowledge", rag_mock)

    result = nodes.retrieve_node(base_state)

    assert "huaweicloud_vpc" in result["rag_context"]
    rag_mock.assert_called_once()


def test_rag_retriever_mock_collection_and_rerank(
    mock_vector_collection,
    monkeypatch,
):
    monkeypatch.setattr(
        rag_retriever,
        "get_collection",
        lambda: mock_vector_collection,
    )
    monkeypatch.setattr(
        rag_retriever,
        "get_embedding",
        lambda text: [0.1, 0.2, 0.3],
    )
    monkeypatch.setattr(
        rag_retriever,
        "qwen_rerank",
        lambda query, documents, top_n=3: documents[:top_n],
    )

    text = rag_retriever.retrieve_tf_knowledge(
        user_query="创建一个华为云 VPC",
        target_resources=["huaweicloud_vpc"],
    )

    assert "华为云 VPC 资源配置示例" in text
    assert mock_vector_collection.query.called


def test_rag_retriever_returns_fallback_when_no_documents(
    mock_vector_collection,
    monkeypatch,
):
    mock_vector_collection.query.return_value = {
        "documents": [[]],
    }
    monkeypatch.setattr(
        rag_retriever,
        "get_collection",
        lambda: mock_vector_collection,
    )
    monkeypatch.setattr(
        rag_retriever,
        "get_embedding",
        lambda text: [0.1],
    )

    text = rag_retriever.retrieve_tf_knowledge(
        user_query="不相关的问题",
        target_resources=[],
    )

    assert "未在本地知识库中检索到相关参考文档" in text


# ==================== Generate 节点测试 ====================

def test_generate_node_saves_mocked_hcl_files(base_state, monkeypatch):
    base_state["rag_context"] = "VPC 官方文档片段"

    mock_output = (
        "### FILE: providers.tf\n"
        "```hcl\n"
        'terraform { required_providers { huaweicloud = { source = "huaweicloud/huaweicloud" } } }\n'
        "```\n"
        "### FILE: main.tf\n"
        "```hcl\n"
        'resource "huaweicloud_vpc" "main" { cidr = "192.168.0.0/16" }\n'
        "```"
    )
    monkeypatch.setattr(
        nodes,
        "llm_coder",
        Mock(invoke=Mock(return_value=AIMessage(content=mock_output))),
    )

    result = nodes.generate_node(base_state)

    assert set(result["generated_files"]) == {"providers.tf", "main.tf"}
    assert (base_state["workspace_dir"] + "/main.tf").replace(
        "//", "/"
    )


# ==================== Sandbox 节点测试 ====================

def test_sandbox_success(base_state, mock_sandbox_execution, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        mock_sandbox_execution,
    )

    result = nodes.sandbox_validate_node(base_state)

    assert result["validation_passed"] is True
    assert result["security_passed"] is True
    assert result["error_message"] is None
    mock_sandbox_execution.assert_called_once()


def test_sandbox_return_code_failure(base_state, monkeypatch):
    failed_process = SimpleNamespace(
        returncode=1,
        stdout="Error: unsupported argument",
        stderr="Syntax Error",
    )
    run_mock = Mock(return_value=failed_process)

    monkeypatch.setattr(subprocess, "run", run_mock)

    result = nodes.sandbox_validate_node(base_state)

    assert result["validation_passed"] is False
    assert result["retry_count"] == 1
    assert "Syntax Error" in result["error_message"]
    assert "unsupported argument" in result["error_message"]


def test_sandbox_timeout(base_state, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=90,
        )

    monkeypatch.setattr(subprocess, "run", timeout)

    result = nodes.sandbox_validate_node(base_state)

    assert result["validation_passed"] is False
    assert result["security_passed"] is False
    assert "超时" in result["error_message"]
    assert result["retry_count"] == 1


def test_sandbox_permission_denied(base_state, monkeypatch):
    def permission_error(*args, **kwargs):
        raise PermissionError("Docker engine access denied")

    monkeypatch.setattr(subprocess, "run", permission_error)

    result = nodes.sandbox_validate_node(base_state)

    assert result["validation_passed"] is False
    assert "Docker 调用异常" in result["error_message"]
    assert "access denied" in result["error_message"]


# ==================== Reflection 节点测试 ====================

def test_reflection_extracts_syntax_error(base_state, monkeypatch):
    base_state["retry_count"] = 1
    base_state["error_message"] = (
        "main.tf:12,3-15: Argument or block definition required; "
        "Unsupported argument: an argument named \"fake_field\" is not expected."
    )
    base_state["rag_context"] = "huaweicloud_vpc 官方字段中不存在 fake_field"

    with open(
        base_state["workspace_dir"] + "/main.tf",
        "w",
        encoding="utf-8",
    ) as tf_file:
        tf_file.write(
            'resource "huaweicloud_vpc" "main" {\n'
            '  fake_field = "invalid"\n'
            "}\n"
        )

    diagnosis = StructuredDiagnosis(
        fixable=True,
        advice=(
            "删除 main.tf 中 huaweicloud_vpc.main 的 fake_field 参数；"
            "保留 cidr 参数。"
        ),
    )
    structured_llm = Mock()
    structured_llm.invoke.return_value = diagnosis
    reflector = Mock()
    reflector.with_structured_output.return_value = structured_llm
    monkeypatch.setattr(nodes, "llm_reflector", reflector)

    result = nodes.reflect_node(base_state)

    assert result["needs_manual_intervention"] is False
    assert "fake_field" in result["reflection_advice"]
    assert "main.tf" in result["reflection_advice"]
    structured_llm.invoke.assert_called_once()


def test_reflection_environment_issue_requires_human(
    base_state,
    monkeypatch,
):
    base_state["error_message"] = "Provider download timeout"

    diagnosis = StructuredDiagnosis(
        fixable=False,
        advice="沙箱无法下载 Provider，请人工检查网络和代理。",
    )
    structured_llm = Mock()
    structured_llm.invoke.return_value = diagnosis
    reflector = Mock()
    reflector.with_structured_output.return_value = structured_llm
    monkeypatch.setattr(nodes, "llm_reflector", reflector)

    result = nodes.reflect_node(base_state)

    assert result["needs_manual_intervention"] is True
    assert "人工介入" in result["reflection_advice"]


# ==================== General QA、Risk、Human 节点测试 ====================

def test_general_qa_node(base_state, monkeypatch):
    base_state["user_query"] = "什么是 Terraform Remote Backend？"

    monkeypatch.setattr(
        nodes,
        "simple_retrieve",
        lambda query, top_n=3: [["Remote Backend 可用于集中存储状态文件"]],
    )
    monkeypatch.setattr(
        nodes,
        "llm_router",
        Mock(invoke=Mock(
            return_value=AIMessage(
                content="Remote Backend 用于集中存储 Terraform 状态。",
            )
        )),
    )

    result = nodes.general_qa_node(base_state)

    assert "Remote Backend" in result["final_response"]
    assert "状态" in result["rag_context"]


def test_risk_analyze_failed_validation_fallback(base_state):
    base_state["validation_passed"] = False
    base_state["security_passed"] = False
    base_state["error_message"] = "Syntax Error"

    result = nodes.risk_analyze_node(base_state)

    assert result["risk_level"] == "HIGH"
    assert "安全熔断" in result["risk_report"]


def test_human_review_auto_approve(base_state, monkeypatch):
    monkeypatch.setattr(
        nodes,
        "interrupt",
        lambda payload: {"action": "approve"},
    )

    result = nodes.human_review_node(base_state)

    assert result["human_action"] == "approve"
    assert result["human_feedback"] is None


def test_human_review_modify_increments_version(base_state, monkeypatch):
    monkeypatch.setattr(
        nodes,
        "interrupt",
        lambda payload: {
            "action": "modify",
            "feedback": "把带宽改成10M",
        },
    )

    result = nodes.human_review_node(base_state)

    assert result["human_action"] == "modify"
    assert result["human_feedback"] == "把带宽改成10M"
    assert result["version"] == 2
    assert result["retry_count"] == 0


# ==================== Conditional Edge 测试 ====================

@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"intent": "general_qa"}, "general_qa"),
        ({"intent": "generate_tf"}, "retrieve"),
        ({}, "retrieve"),
    ],
)
def test_route_after_intent(state, expected):
    assert route_after_intent(state) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {"validation_passed": True, "retry_count": 0},
            "human_review",
        ),
        (
            {"validation_passed": False, "retry_count": 1},
            "reflect",
        ),
        (
            {"validation_passed": False, "retry_count": 3},
            "risk_analyze",
        ),
    ],
)
def test_route_after_validation(state, expected):
    assert route_after_validation(state) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"needs_manual_intervention": True}, "human_review"),
        ({"needs_manual_intervention": False}, "generate"),
    ],
)
def test_router_after_reflect(state, expected):
    assert router_after_reflect(state) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"human_action": "approve"}, "summarize"),
        ({"human_action": "modify"}, "generate"),
    ],
)
def test_route_after_human(state, expected):
    assert route_after_human(state) == expected