"""
System configuration settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """System configuration."""

    # LLM Provider Selection
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "qwen")  # Options: qwen, ollama

    # DashScope (Qwen) Configuration
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    DASHSCOPE_API_BASE = os.getenv(
        "DASHSCOPE_API_BASE",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")  # or qwen-turbo, qwen-max
    QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen-vl-max-latest")  # for image/video

    # Ollama Configuration (for local deployment)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")  # or llama3.1, mistral, etc.
    OLLAMA_VL_MODEL = os.getenv("OLLAMA_VL_MODEL", "llava:13b")  # for vision tasks

    # Agent Configuration
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
    QUALITY_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", "0.7"))
    MIN_TEXT_NEWS = int(os.getenv("MIN_TEXT_NEWS", "3"))
    MIN_VIDEO_NEWS = int(os.getenv("MIN_VIDEO_NEWS", "3"))
    SUPERVISOR_BATCH_SIZE = int(os.getenv("SUPERVISOR_BATCH_SIZE", "0"))  # 0 表示不限制，处理全部
    ANALYSIS_MAX_CYCLES = int(os.getenv("ANALYSIS_MAX_CYCLES", "3"))  # 分析团队内层循环上限

    # Tavily search
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    TAVILY_ENDPOINT = os.getenv("TAVILY_ENDPOINT", "https://api.tavily.com/search")

    # interactive crawler
    ENABLE_INTERACTIVE_CRAWLER: bool = os.getenv("ENABLE_INTERACTIVE_CRAWLER", "false").lower() == "true"
    # Video Platform Recognition
    VIDEO_PLATFORMS = os.getenv(
        "VIDEO_PLATFORMS",
        "bilibili.com,b23.tv,douyin.com,tiktok.com,youtube.com,youtu.be"
    ).split(",")

    # Email notification (optional)
    EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
    EMAIL_SMTP_HOST = os.getenv("EMAIL_SMTP_HOST", "")
    EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "465"))
    EMAIL_SMTP_USER = os.getenv("EMAIL_SMTP_USER", "")
    EMAIL_SMTP_PASSWORD = os.getenv("EMAIL_SMTP_PASSWORD", "")
    EMAIL_FROM = os.getenv("EMAIL_FROM", "")
    EMAIL_TO = os.getenv("EMAIL_TO", "")  # comma-separated
    EMAIL_USE_SSL = os.getenv("EMAIL_USE_SSL", "true").lower() == "true"
    EMAIL_USE_TLS = os.getenv("EMAIL_USE_TLS", "false").lower() == "true"

    # Timezone display (for report header)
    TIMEZONE = os.getenv("TIMEZONE", "Asia/Shanghai (UTC+08:00)")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    @classmethod
    def display(cls):
        """Display current configuration."""
        print("\n=== System Configuration ===")
        print(f"LLM Provider: {cls.LLM_PROVIDER}")

        if cls.LLM_PROVIDER == "qwen":
            print(f"Qwen Model: {cls.QWEN_MODEL}")
            print(f"Qwen-VL Model: {cls.QWEN_VL_MODEL}")
            print(f"DashScope API Key: {'***' + cls.DASHSCOPE_API_KEY[-4:] if cls.DASHSCOPE_API_KEY else 'Not Set'}")
            print(f"DashScope API Base: {cls.DASHSCOPE_API_BASE}")
        elif cls.LLM_PROVIDER == "ollama":
            print(f"Ollama Base URL: {cls.OLLAMA_BASE_URL}")
            print(f"Ollama Model: {cls.OLLAMA_MODEL}")
            print(f"Ollama VL Model: {cls.OLLAMA_VL_MODEL}")

        print(f"Max Iterations: {cls.MAX_ITERATIONS}")
        print(f"Quality Threshold: {cls.QUALITY_THRESHOLD}")
        print(f"Min Text News: {cls.MIN_TEXT_NEWS}")
        print(f"Min Video News: {cls.MIN_VIDEO_NEWS}")
        print(f"Tavily: {'已配置' if cls.TAVILY_API_KEY else '未配置'}")
        print(f"Email Notify: {'已开启' if cls.EMAIL_ENABLED else '未开启'}")
        print(f"Timezone: {cls.TIMEZONE}")
        print(f"Debug Mode: {cls.DEBUG}")
        print("===========================\n")


settings = Settings()
