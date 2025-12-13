from bs4 import BeautifulSoup
import asyncio
import httpx
from typing import List
import os
import random
from playwright.async_api import async_playwright
 
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
    
    result_blocks = soup.find_all('div', class_='result c-container xpath-log new-pmd', limit=n)

    print(f"[DEBUG] Found {len(result_blocks)} result blocks")

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

    print(f"[DEBUG] Extracted {len(links)} valid links")

    return links


HIT_WEBSITES = {
    "baidu": _extract_baidu_hot_search_links,
}

async def mining_from_serch_with_browser(platform: str, url: str, limit: int = 1, interactive: bool = True) -> List[dict]:
    """
    使用 Playwright 浏览器自动化从搜索页面提取链接，可绕过验证码
    
    Args:
        platform: 平台名称（如 'baidu'）
        url: 目标网页 URL
        limit: 需要提取的结果数量，默认 1
        interactive: 是否启用交互式验证码处理，默认 True
        
    Returns:
        list[dict]: 包含提取内容的字典列表
    """
    mining_results = []
    
    if platform not in HIT_WEBSITES:
        mining_results.append({"url": url})
        return mining_results
    
    async with async_playwright() as p:
        # 当需要交互式验证时，使用有头模式
        print(f"🚀 启动浏览器 (headless={not interactive}, interactive={interactive})")
        browser = await p.chromium.launch(
            headless=not interactive,  # 交互模式下显示浏览器窗口
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-dev-shm-usage',
            ],
            # 增加慢速模式，更容易观察（仅调试时使用）
            # slow_mo=100 if interactive else 0,
        )
        
        # 创建浏览器上下文，模拟真实用户
        context = await browser.new_context(
            user_agent=os.getenv(
                "NEWSNOW_USER_AGENT",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/118.0 Safari/537.36",
            ),
            viewport={'width': 1920, 'height': 1080},
            locale='zh-CN',
            timezone_id='Asia/Shanghai',
        )
        
        # 注入 JavaScript 隐藏 webdriver 特征
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        
        page = await context.new_page()
        
        try:
            # 随机延迟，模拟人类行为
            await asyncio.sleep(random.uniform(1, 3))
            
            # 访问页面，等待网络空闲
            print(f"🌐 正在访问: {url}")
            try:
                await page.goto(url, wait_until='networkidle', timeout=30000)
            except Exception as e:
                # 即使超时也继续，可能只是部分资源加载失败
                print(f"⚠️  页面加载警告: {e}")
                await page.wait_for_timeout(2000)
            
            # 额外等待页面完全加载
            await page.wait_for_timeout(random.randint(2000, 4000))
            
            # 检测是否遇到验证码页面（检查多次以确保准确）
            max_retries = 3
            for retry in range(max_retries):
                current_url = page.url
                page_title = await page.title()
                
                print(f"📍 当前页面 URL: {current_url}")
                print(f"📄 当前页面标题: {page_title}")
                
                # 检测百度验证码特征
                is_captcha_page = (
                    'captcha' in current_url.lower() or 
                    'tuxing' in current_url.lower() or
                    'wappass' in current_url.lower() or
                    '安全验证' in page_title or
                    '百度安全验证' in page_title or
                    await page.locator('text=安全验证').count() > 0
                )
                
                if is_captcha_page:
                    if interactive:
                        print("\n" + "="*60)
                        print("⚠️  检测到百度安全验证码！")
                        print(f"📍 验证码页面: {current_url}")
                        print(f"📄 页面标题: {page_title}")
                        print("="*60)
                        print("\n🔔 请在当前浏览器窗口中手动完成验证...")
                        print("💡 提示：")
                        print("   1. 查看打开的 Chromium 浏览器窗口")
                        print("   2. 完成图片验证或滑块验证")
                        print("   3. 等待页面自动跳转到搜索结果")
                        print("   4. 看到正常搜索结果后，回到终端按 Enter")
                        print("="*60 + "\n")
                        
                        # 等待用户输入
                        input("✋ 完成验证后按 Enter 继续 >>> ")
                        
                        # 用户完成验证后，等待页面稳定
                        print("⏳ 等待页面加载完成...")
                        await page.wait_for_timeout(3000)
                        
                        # 再次检查是否成功跳转
                        final_url = page.url
                        final_title = await page.title()
                        
                        print(f"📍 验证后 URL: {final_url}")
                        print(f"📄 验证后标题: {final_title}")
                        
                        # 验证是否成功
                        still_captcha = (
                            'captcha' in final_url.lower() or 
                            'tuxing' in final_url.lower() or
                            'wappass' in final_url.lower()
                        )
                        
                        if not still_captcha:
                            print("✅ 验证成功！继续提取内容...")
                            break
                        else:
                            print("⚠️  似乎还在验证页面...")
                            if retry < max_retries - 1:
                                print(f"🔄 重试第 {retry + 2}/{max_retries} 次...")
                                await page.wait_for_timeout(2000)
                            else:
                                print("❌ 验证失败次数过多，返回原始 URL")
                                mining_results.append({"url": url})
                                return mining_results
                    else:
                        print(f"⚠️  检测到验证码但未启用交互模式: {current_url}")
                        print("💡 提示: 设置 interactive=True 以手动完成验证")
                        mining_results.append({"url": url})
                        return mining_results
                else:
                    # 没有验证码，直接继续
                    print("✅ 未检测到验证码，直接提取内容")
                    break
            
            # 随机滚动页面，模拟真实用户行为
            await page.evaluate('window.scrollBy(0, Math.random() * 500)')
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # 获取页面内容
            html_content = await page.content()
            
            # 调试：保存 HTML 和截图（可选）
            if os.getenv('DEBUG_SAVE_HTML', 'true').lower() == 'true':
                os.makedirs('/tmp/crawler_debug', exist_ok=True)
                
                # 保存 HTML
                debug_html_file = '/tmp/crawler_debug/baidu_debug.html'
                with open(debug_html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                print(f"[DEBUG] Saved HTML to {debug_html_file}")
                
                # 保存截图
                debug_screenshot_file = '/tmp/crawler_debug/baidu_debug.png'
                await page.screenshot(path=debug_screenshot_file, full_page=True)
                print(f"[DEBUG] Saved screenshot to {debug_screenshot_file}")
            
            # 使用对应平台的解析器
            print(f"[DEBUG] Parsing HTML content, length: {len(html_content)} chars")
            extractor = HIT_WEBSITES[platform](html_content, n=limit)
            if extractor and len(extractor) > 0:
                mining_results.extend(extractor)
                print(f"✅ Successfully extracted {len(extractor)} items")
            else:
                print(f"[WARNING] No content extracted from {url}, returning original URL")
                mining_results.append({"url": url})
                
        except Exception as e:
            print(f"❌ Browser automation error for {url}: {e}")
            mining_results.append({"url": url})
        finally:
            await browser.close()
    
    return mining_results


def _is_security_challenge(html_content: str, response_url: str = "") -> bool:
    """
    检测响应是否包含安全验证（验证码、反爬虫检测）
    
    Args:
        html_content: 响应的 HTML 内容
        response_url: 最终响应的 URL（可能被重定向）
        
    Returns:
        bool: 如果检测到安全验证返回 True，否则返回 False
    """
    if not html_content:
        return False
    
    # 检查 URL 特征
    url_lower = response_url.lower()
    url_indicators = [
        'captcha',
        'tuxing',
        'wappass',
        'verify',
        'challenge',
        'security',
    ]
    
    for indicator in url_indicators:
        if indicator in url_lower:
            print(f"🔍 检测到安全验证 URL 特征: {indicator}")
            return True
    
    # 检查 HTML 内容特征
    html_lower = html_content.lower()
    content_indicators = [
        '安全验证',
        '百度安全验证',
        '滑动验证',
        '图形验证',
        '人机验证',
        'robot check',
        'security verification',
        'access denied',
        'captcha',
    ]
    
    for indicator in content_indicators:
        if indicator in html_lower:
            print(f"🔍 检测到安全验证内容特征: {indicator}")
            return True
    
    # 检查是否包含验证码相关的脚本或元素
    if 'id="captcha"' in html_content or 'class="captcha"' in html_content:
        print("🔍 检测到验证码元素")
        return True
    
    return False


async def mining_from_serch(platform: str, client: httpx.AsyncClient, header: dict, url: str, limit: int = 1, use_browser: bool = False, interactive: bool = True) -> List[dict]:
    """
    从搜索页面中提取链接信息
    优化策略：优先使用快速的 HTTP 请求，只有在检测到安全验证时才切换到浏览器模式

    Args:
        platform: 平台名称
        client: HTTP 客户端
        header: 请求头
        url: 目标 URL
        limit: 提取数量限制
        use_browser: 是否强制使用浏览器模式（默认 False，自动检测）
        interactive: 浏览器模式下是否启用交互式验证码处理

    Returns:
        提取的链接信息列表
    """
    mining_results = []

    if platform not in HIT_WEBSITES:
        mining_results.append({"url": url})
        return mining_results
    
    # 如果强制使用浏览器模式，直接调用浏览器方法
    if use_browser:
        print("🌐 强制使用浏览器模式")
        return await mining_from_serch_with_browser(platform, url, limit, interactive=interactive)
    
    # 第一步：尝试使用快速的 HTTP 请求
    print(f"⚡ 尝试快速 HTTP 请求: {url[:80]}...")
    
    # 随机延迟，避免请求过快
    await asyncio.sleep(random.uniform(0.5, 1.5))
    
    # 增强请求头，模拟真实浏览器
    enhanced_headers = {
        **header,
        "sec-ch-ua": '"Google Chrome";v="118", "Chromium";v="118", "Not=A?Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }
    
    try:
        response = await client.get(url, headers=enhanced_headers, timeout=30)
        response.raise_for_status()
        html_content = response.text
        
        # 检查是否触发了安全验证
        final_url = str(response.url)
        if _is_security_challenge(html_content, final_url):
            # 如果 interactive 为 False，则不切换到浏览器模式
            if not interactive:
                print("⚠️  检测到安全验证，但 interactive=False，返回原始 URL")
                mining_results.append({"url": url})
                return mining_results
            
            print("⚠️  检测到安全验证，自动切换到浏览器模式...")
            return await mining_from_serch_with_browser(platform, url, limit, interactive=interactive)
        
        # 未检测到安全验证，继续使用 HTTP 请求提取内容
        print("✅ HTTP 请求成功，未检测到安全验证")
        extractor = HIT_WEBSITES[platform](html_content, n=limit)
        
        if extractor and len(extractor) > 0:
            mining_results.extend(extractor)
            print(f"✅ 成功提取 {len(extractor)} 条内容 (HTTP 模式)")
        else:
            print(f"⚠️  未提取到内容，返回原始 URL")
            mining_results.append({"url": url})
            
    except httpx.HTTPStatusError as e:
        # HTTP 状态错误，可能是反爬虫机制
        print(f"⚠️  HTTP 请求失败 (状态码 {e.response.status_code})，切换到浏览器模式...")
        if interactive is False:
            print("⚠️  interactive=False，返回原始 URL")
            mining_results.append({"url": url})
            return mining_results
        return await mining_from_serch_with_browser(platform, url, limit, interactive=interactive)
        
    except Exception as e:
        # 其他错误（超时、连接失败等），尝试浏览器模式
        print(f"⚠️  HTTP 请求异常: {e}")
        print("🔄 切换到浏览器模式重试...")
        return await mining_from_serch_with_browser(platform, url, limit, interactive=interactive)
    
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
        # print("测试普通 HTTP 请求模式：")
        # results = await mining_from_serch("baidu", client, default_headers, test_url, limit=1, use_browser=False)
        # for item in results:
        #     print(item)
        
        print("\n测试浏览器自动化模式（推荐，可绕过验证码）：")
        print("💡 如果遇到验证码，浏览器窗口会自动打开，请手动完成验证\n")
        results_browser = await mining_from_serch("baidu", client, default_headers, test_url, limit=1, use_browser=False, interactive=False
                                                  )
        for item in results_browser:
            print(item)

if __name__ == "__main__":
    asyncio.run(main())