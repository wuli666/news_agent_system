"""Test script to diagnose arXiv fetching issues."""
import asyncio
import os
from datetime import datetime
from typing import Any, List
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from langchain_community.utilities import ArxivAPIWrapper

# Get the query from environment or use default
ARXIV_DAILY_QUERY = os.getenv(
    "ARXIV_DAILY_QUERY",
    "cat:cs.AI OR cat:cs.CL OR cat:cs.LG OR cat:cs.IR",
).strip()


async def test_arxiv_fetch(limit: int = 6) -> List[dict[str, Any]]:
    """Test fetching papers from arXiv."""
    logger.info(f"Testing arXiv fetch with query: {ARXIV_DAILY_QUERY}")

    try:
        wrapper = ArxivAPIWrapper(
            top_k_results=30,
            load_max_docs=30,
            load_all_available_meta=True,
        )

        logger.info("ArxivAPIWrapper created successfully")

        def _query() -> List[Any]:
            logger.info("Executing query...")
            # Use get_summaries_as_docs instead of load to avoid downloading PDFs
            results = wrapper.get_summaries_as_docs(ARXIV_DAILY_QUERY)
            logger.info(f"Query returned {len(results) if results else 0} results")
            return results

        docs = await asyncio.to_thread(_query)

        if not docs:
            logger.warning("No documents returned from arXiv")
            return []

        logger.info(f"Processing {len(docs)} documents...")

        def _parse_ts(meta: dict) -> float:
            for key in ("Published", "published", "updated", "Updated"):
                val = meta.get(key)
                if not val:
                    continue
                try:
                    return datetime.fromisoformat(str(val).replace("Z", "+00:00")).timestamp()
                except Exception as e:
                    logger.debug(f"Failed to parse timestamp from {key}={val}: {e}")
                    continue
            return 0.0

        papers: List[dict[str, Any]] = []
        for idx, doc in enumerate(docs or []):
            try:
                meta = getattr(doc, "metadata", {}) or {}
                logger.debug(f"Doc {idx} metadata keys: {list(meta.keys())}")

                title = meta.get("Title") or meta.get("title") or ""
                # Entry ID is the arxiv URL
                entry_id = meta.get("Entry ID") or meta.get("entry_id") or ""
                url = entry_id if entry_id else (meta.get("pdf_url") or meta.get("link"))
                summary = getattr(doc, "page_content", "") or meta.get("Summary") or meta.get("summary") or ""

                if not title:
                    logger.warning(f"Doc {idx} has no title, skipping")
                    continue

                categories_raw = meta.get("categories", []) or meta.get("Category", []) or []
                categories = categories_raw if isinstance(categories_raw, list) else [categories_raw]

                paper = {
                    "title": str(title).strip(),
                    "category": ", ".join(categories) if categories else "arXiv",
                    "url": url,
                    "summary": str(summary).strip()[:240],
                    "ts": _parse_ts(meta),
                }
                papers.append(paper)
                logger.info(f"Parsed paper {idx + 1}: {paper['title'][:50]}...")

            except Exception as e:
                logger.error(f"Failed to process doc {idx}: {e}", exc_info=True)
                continue

        papers_sorted = sorted(papers, key=lambda x: x.get("ts", 0), reverse=True)
        logger.info(f"Successfully fetched {len(papers_sorted)} papers")

        return papers_sorted[:limit]

    except Exception as e:
        logger.error(f"Arxiv fetch failed: {e}", exc_info=True)
        return []


async def main():
    """Main test function."""
    print("=" * 80)
    print("Testing arXiv Paper Fetching")
    print("=" * 80)
    print(f"Query: {ARXIV_DAILY_QUERY}")
    print()

    papers = await test_arxiv_fetch(limit=10)

    print()
    print("=" * 80)
    print(f"Results: Found {len(papers)} papers")
    print("=" * 80)

    if papers:
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Category: {paper['category']}")
            print(f"   URL: {paper['url']}")
            print(f"   Summary: {paper['summary'][:100]}...")
    else:
        print("\nNo papers found! Check the error logs above for details.")


if __name__ == "__main__":
    asyncio.run(main())
