from langchain_openai import ChatOpenAI

from agent_core.config import get_dashscope_settings, get_openai_compatible_settings


class LLMService:
    def __init__(self) -> None:
        dashscope = get_dashscope_settings()
        openai = get_openai_compatible_settings()

        self.base_url = dashscope["base_url"]
        self.api_key = dashscope["api_key"]
        self.openai_api_key = openai["api_key"]
        self.openai_base_url = openai["base_url"]

    @property
    def router(self) -> ChatOpenAI:
        return ChatOpenAI(
            model="qwen3.8-flash",
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=0.0,
            timeout=60.0,
        )

    @property
    def coder(self) -> ChatOpenAI:
        return ChatOpenAI(
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            base_url=self.openai_base_url,
            api_key=self.openai_api_key,
            temperature=0.1,
            timeout=120.0,
        )

    @property
    def reflector(self) -> ChatOpenAI:
        return ChatOpenAI(
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            base_url=self.openai_base_url,
            api_key=self.openai_api_key,
            temperature=0.1,
            timeout=120.0,
        )
