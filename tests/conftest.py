import os
from types import SimpleNamespace
from unittest.mock import Mock
from collections.abc import Iterator

import pytest


# 在导入 Agent 模块前设置环境变量，避免配置类在测试收集阶段要求真实密钥。
os.environ.setdefault("DASHSCOPE_BASE_URL", "https://mock.dashscope.example")
os.environ.setdefault("DASHSCOPE_API_KEY", "mock-dashscope-key")
os.environ.setdefault("OPENAI_BASE_URL", "https://mock.openai.example/v1")
os.environ.setdefault("OPENAI_API_KEY", "mock-openai-key")


class MockLLM:
    """可按调用顺序或固定内容返回 LangChain 格式响应的自定义 Mock。"""

    def __init__(self, responses: list[str] | str):
        if isinstance(responses, str):
            responses = [responses]
        self.responses = list(responses)
        self.invoke = Mock(side_effect=self._invoke)

    def _invoke(self, messages):
        if not self.responses:
            raise AssertionError("Mock LLM 收到了未预期的额外调用")
        content = self.responses.pop(0)
        return SimpleNamespace(content=content)


class MockStructuredLLM:
    """Mock with_structured_output 返回的结构化 LLM。"""

    def __init__(self, result):
        self.result = result
        self.invoke = Mock(return_value=result)


@pytest.fixture
def mock_llm_response() -> Iterator[MockLLM]:
    """提供固定的 Router/Coder 响应，并暴露 Mock 调用记录。"""
    llm = MockLLM(
        [
            '{"intent": "generate_tf", '
            '"target_resources": ["huaweicloud_vpc"]}',
        ]
    )
    yield llm


@pytest.fixture
def mock_sandbox_execution() -> Iterator[Mock]:
    """模拟 Terraform 与 OPA 均成功的 Docker 沙箱进程。"""
    process = Mock(
        returncode=0,
        stdout=(
            '{"result":[{"expressions":[{"value":[]}]}]}'
        ),
        stderr="",
    )
    yield Mock(return_value=process)


@pytest.fixture
def mock_vector_collection() -> Mock:
    """模拟支持 where、n_results 和 query_embeddings 的 Chroma collection。"""
    collection = Mock()

    collection.query.return_value = {
        "documents": [[
            "华为云 VPC 资源配置示例",
            "huaweicloud_vpc 支持 cidr 参数",
        ]],
        "metadatas": [[
            {"doc_type": "resource"},
            {"doc_type": "resource"},
        ]],
        "distances": [[0.05, 0.20]],
    }
    return collection


@pytest.fixture
def base_state(tmp_path) -> dict:
    """创建所有节点都可安全读取的完整基础状态。"""
    return {
        "task_id": "test-task",
        "workspace_dir": str(tmp_path),
        "user_query": "创建一个华为云 VPC",
        "intent": "generate_tf",
        "target_resources": ["huaweicloud_vpc"],
        "rag_context": "",
        "generated_files": [],
        "validation_passed": False,
        "security_passed": False,
        "error_message": None,
        "reflection_advice": None,
        "retry_count": 0,
        "needs_manual_intervention": False,
        "risk_report": None,
        "risk_level": None,
        "version": 1,
        "human_action": None,
        "human_feedback": None,
        "final_response": "",
    }