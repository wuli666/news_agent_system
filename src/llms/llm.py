"""
LLM provider management.
Supports both DashScope (Qwen) and Ollama.
"""
import logging
from typing import Any
from langchain_qwq import ChatQwen
from src.config.settings import settings

logger = logging.getLogger(__name__)


def _create_chat_qwen(model_name: str, **kwargs) -> ChatQwen:
    """
    Helper to instantiate ChatQwen for DashScope API.

    The DashScope API key should be provided via the DASHSCOPE_API_KEY
    environment variable. Passing it as a direct keyword argument causes
    incompatibilities with newer langchain-openai releases, so we avoid
    that entirely and rely on env config.
    """
    base_url = kwargs.pop("base_url", settings.DASHSCOPE_API_BASE)
    return ChatQwen(model=model_name, base_url=base_url, **kwargs)


def _create_chat_ollama(model_name: str, **kwargs) -> Any:
    """
    Helper to instantiate ChatOllama for local Ollama deployment.

    Requires Ollama to be running locally at OLLAMA_BASE_URL.
    Install with: pip install langchain-ollama
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "langchain-ollama is not installed. "
            "Install it with: pip install langchain-ollama"
        )

    base_url = kwargs.pop("base_url", settings.OLLAMA_BASE_URL)
    return ChatOllama(
        model=model_name,
        base_url=base_url,
        **kwargs
    )


def get_llm_by_type(llm_type: str):
    """
    Get LLM instance by type.

    Args:
        llm_type: Type of LLM to use ("qwen" or "qwen_vl")

    Returns:
        LLM instance based on configured provider (Qwen or Ollama)
    """
    provider = settings.LLM_PROVIDER

    if provider == "ollama":
        # Use Ollama for local deployment
        if llm_type == "qwen_vl":
            # Use vision model for visual tasks
            return _create_chat_ollama(
                settings.OLLAMA_VL_MODEL,
                temperature=0.7,
                num_predict=10000,  # Ollama uses num_predict instead of max_tokens
            )
        else:
            # Use regular model for text tasks
            return _create_chat_ollama(
                settings.OLLAMA_MODEL,
                temperature=0.7,
                num_predict=10000,
            )

    elif provider == "qwen":
        # Use DashScope (Qwen) API
        if llm_type == "qwen":
            return _create_chat_qwen(
                settings.QWEN_MODEL,
                temperature=0.7,
                max_tokens=10000,
            )
        elif llm_type == "qwen_vl":
            return _create_chat_qwen(
                settings.QWEN_VL_MODEL,
                temperature=0.7,
                max_tokens=10000,
            )
        else:
            logger.warning(f"Unknown LLM type: {llm_type}, falling back to default Qwen")
            return _create_chat_qwen(
                settings.QWEN_MODEL,
                temperature=0.7,
                max_tokens=2000,
            )

    else:
        logger.error(f"Unknown LLM provider: {provider}, falling back to Qwen")
        return _create_chat_qwen(
            settings.QWEN_MODEL,
            temperature=0.7,
            max_tokens=2000,
        )


def get_llm_with_config(llm_type: str, **kwargs):
    """
    Get LLM with custom configuration.

    Args:
        llm_type: Type of LLM to use ("qwen" or "qwen_vl")
        **kwargs: Custom configuration parameters

    Returns:
        LLM instance with custom config based on configured provider
    """
    provider = settings.LLM_PROVIDER

    if provider == "ollama":
        # Convert max_tokens to num_predict for Ollama
        if "max_tokens" in kwargs:
            kwargs["num_predict"] = kwargs.pop("max_tokens")

        if llm_type == "qwen_vl":
            return _create_chat_ollama(settings.OLLAMA_VL_MODEL, **kwargs)
        else:
            return _create_chat_ollama(settings.OLLAMA_MODEL, **kwargs)

    elif provider == "qwen":
        if llm_type == "qwen":
            return _create_chat_qwen(settings.QWEN_MODEL, **kwargs)
        elif llm_type == "qwen_vl":
            return _create_chat_qwen(settings.QWEN_VL_MODEL, **kwargs)
        else:
            return _create_chat_qwen(settings.QWEN_MODEL, **kwargs)

    else:
        logger.error(f"Unknown LLM provider: {provider}, falling back to Qwen")
        return _create_chat_qwen(settings.QWEN_MODEL, **kwargs)
