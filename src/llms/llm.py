"""
LLM provider management.
Uses ChatQwen for native Qwen model support.
"""
import logging
from langchain_qwq import ChatQwen
from src.config.settings import settings

logger = logging.getLogger(__name__)


def _create_chat_qwen(model_name: str, **kwargs) -> ChatQwen:
    """
    Helper to instantiate ChatQwen.

    The DashScope API key should be provided via the DASHSCOPE_API_KEY
    environment variable (see README_CHATQWEN.md). Passing it as a direct
    keyword argument causes incompatibilities with newer langchain-openai
    releases, so we avoid that entirely and rely on env config.
    """
    base_url = kwargs.pop("base_url", settings.DASHSCOPE_API_BASE)
    return ChatQwen(model=model_name, base_url=base_url, **kwargs)


def get_llm_by_type(llm_type: str):
    """
    Get LLM instance by type.
    """
    if llm_type == "qwen":
        return _create_chat_qwen(
            settings.QWEN_MODEL,
            temperature=0.7,
            max_tokens=2000,
        )
    elif llm_type == "qwen_vl":
        return _create_chat_qwen(
            settings.QWEN_VL_MODEL,
            temperature=0.7,
            max_tokens=2000,
        )
    else:
        logger.warning(f"Unknown LLM type: {llm_type}, falling back to default Qwen")
        return _create_chat_qwen(
            settings.QWEN_MODEL,
            temperature=0.7,
            max_tokens=2000,
        )


def get_llm_with_config(llm_type: str, **kwargs):
    """
    Get LLM with custom configuration.
    """
    if llm_type == "qwen":
        return _create_chat_qwen(settings.QWEN_MODEL, **kwargs)
    elif llm_type == "qwen_vl":
        return _create_chat_qwen(settings.QWEN_VL_MODEL, **kwargs)
    else:
        return _create_chat_qwen(settings.QWEN_MODEL, **kwargs)
