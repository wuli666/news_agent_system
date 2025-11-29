"""
News deduplication and merging utility.

Groups similar news items based on title similarity and merges their information.
"""
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_title_similarity(title1: str, title2: str) -> float:
    """
    Compute similarity between two titles using SequenceMatcher.

    Returns:
        float: Similarity score between 0 and 1
    """
    if not title1 or not title2:
        return 0.0

    # Normalize: lowercase and strip
    t1 = title1.lower().strip()
    t2 = title2.lower().strip()

    # Exact match
    if t1 == t2:
        return 1.0

    # Use SequenceMatcher to compute ratio
    return SequenceMatcher(None, t1, t2).ratio()


def group_similar_news(
    news_items: List[Dict[str, Any]],
    similarity_threshold: float = 0.7
) -> List[List[Dict[str, Any]]]:
    """
    Group news items by title similarity.

    Args:
        news_items: List of news items with 'title' field
        similarity_threshold: Minimum similarity to consider items as duplicates

    Returns:
        List of groups, where each group is a list of similar news items
    """
    if not news_items:
        return []

    groups: List[List[Dict[str, Any]]] = []
    used_indices = set()

    for i, item in enumerate(news_items):
        if i in used_indices:
            continue

        # Start a new group with this item
        current_group = [item]
        used_indices.add(i)

        # Find all similar items
        for j, other_item in enumerate(news_items):
            if j <= i or j in used_indices:
                continue

            similarity = compute_title_similarity(
                item.get("title", ""),
                other_item.get("title", "")
            )

            if similarity >= similarity_threshold:
                current_group.append(other_item)
                used_indices.add(j)

        groups.append(current_group)

    return groups


def merge_news_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge a group of similar news items into a single representative item.

    Strategy:
    - Use the first item's title as primary
    - Collect all unique platforms
    - Keep the highest rank
    - Merge sources
    - Collect all URLs
    """
    if not group:
        return {}

    if len(group) == 1:
        return group[0]

    # Use first item as base
    merged = group[0].copy()

    # Collect all platforms
    platforms = set()
    sources = set()
    urls = []
    min_rank = float('inf')

    for item in group:
        if platform_id := item.get("platform_id"):
            platforms.add(platform_id)
        if platform_name := item.get("platform_name"):
            platforms.add(platform_name)
        if source := item.get("source"):
            sources.add(source)
        if url := item.get("url"):
            if url not in urls:
                urls.append(url)
        if rank := item.get("rank"):
            min_rank = min(min_rank, rank)

    # Update merged item
    merged["platforms"] = sorted(list(platforms))
    merged["sources"] = sorted(list(sources))
    merged["duplicate_count"] = len(group)
    merged["rank"] = int(min_rank) if min_rank != float('inf') else merged.get("rank", 0)

    if urls:
        merged["urls"] = urls
        merged["url"] = urls[0]  # Keep first URL as primary

    # Add summary of duplicate titles for transparency
    merged["duplicate_titles"] = [item.get("title", "") for item in group[1:]]

    return merged


def deduplicate_news(
    news_items: List[Dict[str, Any]],
    similarity_threshold: float = 0.7,
    keep_duplicates_info: bool = True
) -> Dict[str, Any]:
    """
    Deduplicate and merge similar news items.

    Args:
        news_items: List of news items from NewsNow API
        similarity_threshold: Similarity threshold (0-1) for grouping
        keep_duplicates_info: Whether to keep information about merged duplicates

    Returns:
        Dict with:
            - items: List of deduplicated news items
            - total: Total count after deduplication
            - original_total: Original count before deduplication
            - removed_duplicates: Number of duplicates removed
    """
    if not news_items:
        return {
            "items": [],
            "total": 0,
            "original_total": 0,
            "removed_duplicates": 0,
        }

    original_total = len(news_items)

    # Group similar news
    groups = group_similar_news(news_items, similarity_threshold)

    # Merge each group
    deduplicated_items = []
    for group in groups:
        merged = merge_news_group(group)

        # Optionally remove duplicate info for cleaner output
        if not keep_duplicates_info and "duplicate_titles" in merged:
            del merged["duplicate_titles"]

        deduplicated_items.append(merged)

    removed_count = original_total - len(deduplicated_items)

    logger.info(
        "Deduplication: %d items -> %d items (removed %d duplicates)",
        original_total,
        len(deduplicated_items),
        removed_count,
    )

    return {
        "items": deduplicated_items,
        "total": len(deduplicated_items),
        "original_total": original_total,
        "removed_duplicates": removed_count,
    }
