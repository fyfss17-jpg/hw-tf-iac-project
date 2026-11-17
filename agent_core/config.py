import os
from typing import Dict, Optional


def require_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_dashscope_settings() -> Dict[str, str]:
    return {
        "base_url": require_env("DASHSCOPE_BASE_URL"),
        "api_key": require_env("DASHSCOPE_API_KEY"),
    }


def get_openai_compatible_settings() -> Dict[str, str]:
    return {
        "base_url": require_env("OPENAI_BASE_URL"),
        "api_key": require_env("OPENAI_API_KEY"),
    }


def get_optional_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else None
