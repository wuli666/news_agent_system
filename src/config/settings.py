"""
System configuration settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """System configuration."""

    # LLM Configuration - Using DashScope (Alibaba Cloud)
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    DASHSCOPE_API_BASE = os.getenv(
        "DASHSCOPE_API_BASE",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")  # or qwen-turbo, qwen-max
    QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen-vl-max-latest")  # for image/video

    # Agent Configuration
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
    QUALITY_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", "0.7"))
    MIN_TEXT_NEWS = int(os.getenv("MIN_TEXT_NEWS", "3"))
    MIN_VIDEO_NEWS = int(os.getenv("MIN_VIDEO_NEWS", "3"))

    # Video Platform Recognition
    VIDEO_PLATFORMS = os.getenv(
        "VIDEO_PLATFORMS",
        "bilibili.com,b23.tv,douyin.com,tiktok.com,youtube.com,youtu.be"
    ).split(",")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    @classmethod
    def display(cls):
        """Display current configuration."""
        print("\n=== System Configuration ===")
        print(f"Qwen Model: {cls.QWEN_MODEL}")
        print(f"Qwen-VL Model: {cls.QWEN_VL_MODEL}")
        print(f"DashScope API Key: {'***' + cls.DASHSCOPE_API_KEY[-4:] if cls.DASHSCOPE_API_KEY else 'Not Set'}")
        print(f"DashScope API Base: {cls.DASHSCOPE_API_BASE}")
        print(f"Max Iterations: {cls.MAX_ITERATIONS}")
        print(f"Quality Threshold: {cls.QUALITY_THRESHOLD}")
        print(f"Min Text News: {cls.MIN_TEXT_NEWS}")
        print(f"Min Video News: {cls.MIN_VIDEO_NEWS}")
        print(f"Debug Mode: {cls.DEBUG}")
        print("===========================\n")


settings = Settings()
