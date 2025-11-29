"""
News classification utilities.
Classify news items into text or video based on URL patterns.
"""
import logging
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse
from src.config.settings import settings

logger = logging.getLogger(__name__)


def is_video_url(url: str) -> bool:
    """
    Check if a URL is from a video platform.

    Args:
        url: URL to check

    Returns:
        True if URL is from a known video platform
    """
    if not url:
        return False

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove 'www.' prefix
        if domain.startswith('www.'):
            domain = domain[4:]

        # Check if domain matches any video platform
        for platform in settings.VIDEO_PLATFORMS:
            if platform.strip().lower() in domain:
                logger.debug(f"Identified video URL: {url} (platform: {platform})")
                return True

        return False
    except Exception as e:
        logger.warning(f"Error parsing URL {url}: {e}")
        return False


def classify_news_items(news_items: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Classify news items into text news and video news.

    Args:
        news_items: List of news items collected by tools

    Returns:
        Tuple of (text_news, video_news)
    """
    text_news = []
    video_news = []

    for item in news_items:
        url = item.get('url', '') or item.get('link', '')

        if is_video_url(url):
            video_news.append(item)
            logger.info(f"Classified as VIDEO: {item.get('title', 'Unknown')} - {url}")
        else:
            text_news.append(item)
            logger.info(f"Classified as TEXT: {item.get('title', 'Unknown')}")

    logger.info(f"Classification complete: {len(text_news)} text, {len(video_news)} video")
    return text_news, video_news


def format_news_for_agent(news_items: List[Dict[str, Any]], category: str) -> str:
    """
    Format news items for agent processing.

    Args:
        news_items: List of news items
        category: "text" or "video"

    Returns:
        Formatted string for agent
    """
    if not news_items:
        return f"No {category} news available."

    formatted = f"\n=== {category.upper()} NEWS ({len(news_items)} items) ===\n\n"

    for i, item in enumerate(news_items, 1):
        title = item.get('title', 'Unknown Title')
        url = item.get('url', '') or item.get('link', '')
        source = item.get('source', 'Unknown Source')
        timestamp = item.get('timestamp', '') or item.get('published_at', '')

        formatted += f"{i}. {title}\n"
        formatted += f"   Source: {source}\n"
        if url:
            formatted += f"   URL: {url}\n"
        if timestamp:
            formatted += f"   Time: {timestamp}\n"
        formatted += "\n"

    return formatted
