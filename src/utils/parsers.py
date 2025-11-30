from bs4 import BeautifulSoup
import asyncio
import httpx
from typing import List
import os
 
def _extract_baidu_hot_search_links(html_content: str, n: int = 1) -> list[dict]:
    """
    从百度热搜等搜索平台的 HTML 中提取前 n 个结果的超链接
    
    Args:
        html_content: 网页 HTML 内容
        n: 需要提取的结果数量，默认 1
        
    Returns:
        list[dict]: 包含标题和链接的字典列表，格式：[{"title": "...", "url": "..."}]
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 找到前 n 个 class 为 "result c-container xpath-log new-pmd" 的块
    result_blocks = soup.find_all('div', class_='result c-container xpath-log new-pmd', limit=n)
    
    links = []
    for block in result_blocks:
        # 在每个块中找到 <a> 标签
        item = {}
        a_tag = block.find('a')
        img = block.find('img')
        if a_tag:
            item["url"] = a_tag['href'] # 提取文章链接
        if img and img.has_attr('src'):
            item["img"] = img['src'] # 提取图片链接
        links.append(item)

    return links


HIT_WEBSITES = {
    "baidu": _extract_baidu_hot_search_links,
}

async def mining_from_serch(platform: str, client: httpx.AsyncClient, header: dict, url: str, limit: int = 1) -> List[dict]:
    """
    从HTML内容中提取纯文本。

    Args:
        html_content: 包含HTML的字符串

    Returns:
        提取的纯文本字符串
    """
    mining_results = []

    if platform not in HIT_WEBSITES:
        mining_results.append({"url": url})  # 如果没有对应的解析函数，返回原始URL
    else:
        # print(f"Mining from {platform}...")
        await asyncio.sleep(0.5)  # 避免请求过快
        response = await client.get(url, headers=header)
        try:
            response.raise_for_status()
        except Exception as e:
            print(f"Error while fetching {url}: {e}")
            mining_results.append({"url": url})
            return mining_results
        html_content = response.text
        extractor = HIT_WEBSITES[platform](html_content, n=limit)
        if extractor and len(extractor) > 0:
            mining_results.extend(extractor)
        else:
            mining_results.append({"url": url})  # 如果没有提取到内容，返回原始URL
    
    return mining_results


async def main():
    """Main entry point for testing."""
    test_url = "https://www.baidu.com/s?wd=%E5%B1%B1%E8%A5%BF%E5%90%95%E6%A2%81%E9%93%81%E8%B7%AF%E5%9D%8D%E5%A1%8C%E8%87%B411%E6%AD%BB%E7%B3%BB%E8%B0%A3%E8%A8%80"

    user_agent = os.getenv(
        "NEWSNOW_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/118.0 Safari/537.36",
    )
    accept_language = os.getenv("NEWSNOW_ACCEPT_LANGUAGE", "zh-CN,zh;q=0.9,en;q=0.8")
    referer = os.getenv("NEWSNOW_BASE_URL", "https://newsnow.busiyi.world").rstrip("/")
    cookie = os.getenv("NEWSNOW_COOKIE", "").strip()
    default_headers = {
        "User-Agent": user_agent,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": accept_language,
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Referer": referer,
    }
    if cookie:
        default_headers["Cookie"] = cookie

    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        results = await mining_from_serch("baidu", client, default_headers, test_url, limit=1)
        for item in results:
            print(item)

if __name__ == "__main__":
    asyncio.run(main())