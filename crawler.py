"""
Periodic news crawler - saves data to historical storage.

Similar to TrendRadar, this script fetches news periodically
and saves them to local files for historical analysis.
"""
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.history_storage import get_storage
from src.tools.news_tools import get_latest_news

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def fetch_and_save_news():
    """
    Fetch latest news and save to historical storage.
    """
    logger.info("=" * 60)
    logger.info("Starting news crawl...")
    logger.info("=" * 60)

    try:
        # Fetch news
        logger.info("Fetching latest news via NewsNow service...")
        result = await get_latest_news.ainvoke({
            "limit": 200,  # Get more for historical storage
            "include_url": True,
            "deduplicate": True,
        })

        logger.info(f"Fetched {result.get('total', 0)} news items")

        # Save to historical storage
        storage = get_storage()
        file_path = storage.save_news_data(result)

        logger.info(f"✅ News data saved to: {file_path}")

        # Show stats
        stats = storage.get_stats()
        logger.info(f"Storage stats: {stats}")

    except Exception as e:
        logger.error(f"❌ Crawl failed: {e}", exc_info=True)


async def main():
    """Main entry point."""
    logger.info("News Crawler - Historical Data Storage")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("-" * 60)

    await fetch_and_save_news()

    logger.info("-" * 60)
    logger.info("Crawl completed!")


if __name__ == "__main__":
    asyncio.run(main())
