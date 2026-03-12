"""WSJ newsletter news integration via gog CLI + Playwright scraper.

Fetches WSJ newsletter content from Gmail using the `gog` CLI tool,
extracts article links, and optionally scrapes full article text.
Supplements (does not replace) existing news sources.
"""

import os
import re
import json
import base64
import subprocess
from html.parser import HTMLParser
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlparse, unquote

from .config import get_config

GOG_PATH = "/usr/local/bin/gog"
SCRAPER_PYTHON = "/home/acacia/anaconda3/bin/python3"

# WSJ section URLs for supplemental scraping
WSJ_SECTIONS = {
    "markets": "https://www.wsj.com/news/markets",
    "economy": "https://www.wsj.com/news/economy",
    "politics": "https://www.wsj.com/news/politics",
}

# Company/ticker keyword mapping for filtering
TICKER_KEYWORDS = {
    "BTC": ["bitcoin", "btc", "crypto", "cryptocurrency"],
    "ETH": ["ethereum", "eth", "crypto"],
    "NVDA": ["nvidia", "nvda", "gpu", "chip"],
    "AAPL": ["apple", "aapl", "iphone"],
    "MSFT": ["microsoft", "msft", "azure"],
    "GOOGL": ["google", "alphabet", "googl"],
    "AMZN": ["amazon", "amzn", "aws"],
    "META": ["meta", "facebook", "instagram"],
    "TSLA": ["tesla", "tsla", "ev", "electric vehicle"],
    "VOO": ["s&p 500", "s&p", "vanguard", "index fund"],
    "COIN": ["coinbase", "coin", "crypto exchange"],
    "CRSP": ["crispr", "crsp", "gene editing"],
}


class _NewsletterParser(HTMLParser):
    """Extract article titles, summaries, and links from WSJ newsletter HTML."""

    def __init__(self):
        super().__init__()
        self.articles = []
        self._current_text = []
        self._in_link = False
        self._current_href = None
        self._capture = True

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "a" and attrs_dict.get("href", ""):
            href = attrs_dict["href"]
            if "wsj.com" in href or "trk.wsj.com" in href:
                self._in_link = True
                self._current_href = href

    def handle_endtag(self, tag):
        if tag == "a" and self._in_link:
            self._in_link = False
            self._current_href = None

    def handle_data(self, data):
        text = data.strip()
        if text:
            self._current_text.append(text)
            if self._in_link and self._current_href and len(text) > 15:
                self.articles.append({
                    "title": text,
                    "url": self._current_href,
                    "summary": "",
                })

    def get_articles(self):
        # Deduplicate by title
        seen = set()
        unique = []
        for a in self.articles:
            if a["title"] not in seen:
                seen.add(a["title"])
                unique.append(a)
        return unique


def _decode_wsj_tracking_url(url: str) -> str:
    """Decode a trk.wsj.com tracking URL to the real wsj.com URL.

    Format: https://trk.wsj.com/click/.../BASE64/...
    The base64 segment decodes to the actual article URL.
    """
    if "trk.wsj.com" not in url:
        return url

    # Try to find base64-encoded URL segments in the path
    parts = url.split("/")
    for part in parts:
        # Base64 segments are typically long and contain URL-safe base64 chars
        if len(part) > 30:
            try:
                # Try standard base64
                decoded = base64.b64decode(part + "==", validate=False).decode("utf-8", errors="ignore")
                if "wsj.com" in decoded:
                    # Extract the URL from the decoded string
                    url_match = re.search(r'https?://[^\s"<>]+wsj\.com[^\s"<>]*', decoded)
                    if url_match:
                        return url_match.group(0)
            except Exception:
                pass
            try:
                # Try URL-safe base64
                decoded = base64.urlsafe_b64decode(part + "==").decode("utf-8", errors="ignore")
                if "wsj.com" in decoded:
                    url_match = re.search(r'https?://[^\s"<>]+wsj\.com[^\s"<>]*', decoded)
                    if url_match:
                        return url_match.group(0)
            except Exception:
                pass

    # Fallback: return original URL
    return url


def _decode_wsj_tracking_urls(html: str) -> list[str]:
    """Extract and decode all WSJ tracking URLs from HTML content."""
    urls = re.findall(r'href="(https?://trk\.wsj\.com/[^"]+)"', html)
    decoded = []
    seen = set()
    for url in urls:
        real_url = _decode_wsj_tracking_url(url)
        if real_url not in seen:
            seen.add(real_url)
            decoded.append(real_url)
    return decoded


def _extract_newsletter_articles(html: str) -> list[dict]:
    """Parse newsletter HTML to extract article titles, summaries, and URLs."""
    parser = _NewsletterParser()
    parser.feed(html)
    articles = parser.get_articles()

    # Decode tracking URLs
    for article in articles:
        article["url"] = _decode_wsj_tracking_url(article["url"])

    return articles


def _gog_gmail_search(query: str, max_results: int = 5) -> list[dict]:
    """Search Gmail using gog CLI. Returns list of thread dicts."""
    if not os.path.exists(GOG_PATH):
        return []

    try:
        result = subprocess.run(
            [GOG_PATH, "gmail", "search", query, "--json", "--no-input", f"--max={max_results}", "-a", "toosakarin00@gmail.com"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "GOG_KEYRING_PASSWORD": "acacia"},
        )
        if result.returncode != 0:
            print(f"gog gmail search error: {result.stderr}")
            return []

        output = result.stdout.strip()
        if not output:
            return []

        return json.loads(output)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"gog gmail search failed: {e}")
        return []


def _gog_gmail_get(message_id: str) -> Optional[dict]:
    """Get a specific Gmail message using gog CLI."""
    if not os.path.exists(GOG_PATH):
        return None

    try:
        result = subprocess.run(
            [GOG_PATH, "gmail", "get", message_id, "--json", "--no-input", "-a", "toosakarin00@gmail.com"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "GOG_KEYRING_PASSWORD": "acacia"},
        )
        if result.returncode != 0:
            return None

        output = result.stdout.strip()
        if not output:
            return None

        return json.loads(output)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return None


def _scrape_wsj_article(url: str) -> Optional[dict]:
    """Scrape a WSJ article using the Playwright scraper.

    Uses /home/acacia/anaconda3/bin/python3 (base conda env) since
    Playwright is only installed there.
    Returns None if Chrome CDP is unavailable.
    """
    config = get_config()
    scraper_path = config.get("wsj_scraper_path", "/mnt/acacia_rw/scripts/wsj-scraper.py")

    if not os.path.exists(scraper_path):
        return None

    try:
        result = subprocess.run(
            [SCRAPER_PYTHON, scraper_path, url, "--json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None

        output = result.stdout.strip()
        if not output:
            return None

        return json.loads(output)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return None


def _scrape_wsj_section(section_url: str) -> list[dict]:
    """Scrape a WSJ section page to extract headline links and snippets."""
    scraped = _scrape_wsj_article(section_url)
    if not scraped:
        return []

    # The scraper returns paragraphs from any page.
    # For section pages, "paragraphs" are typically headline/snippet text.
    articles = []
    title = scraped.get("title", "")
    paragraphs = scraped.get("paragraphs", [])

    for p in paragraphs:
        if len(p) > 30:  # Filter out navigation/short text
            articles.append({
                "title": p[:200],
                "url": section_url,
                "summary": "",
                "full_text": "",
                "source": "WSJ Website",
            })

    return articles[:10]  # Limit to avoid noise


def _filter_articles_for_ticker(articles: list[dict], ticker: str) -> list[dict]:
    """Filter articles relevant to a specific ticker."""
    ticker_upper = ticker.upper().replace("-USD", "")
    keywords = TICKER_KEYWORDS.get(ticker_upper, [ticker_upper.lower()])
    # Always include the ticker itself
    keywords = [k.lower() for k in keywords]
    if ticker_upper.lower() not in keywords:
        keywords.append(ticker_upper.lower())

    matching = []
    for article in articles:
        text = (article.get("title", "") + " " + article.get("summary", "")).lower()
        if any(kw in text for kw in keywords):
            matching.append(article)

    return matching


def _load_cache(cache_dir: str, key: str) -> Optional[list]:
    """Load cached data if available."""
    cache_file = os.path.join(cache_dir, f"wsj-cache-{key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def _save_cache(cache_dir: str, key: str, data: list):
    """Save data to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"wsj-cache-{key}.json")
    try:
        with open(cache_file, "w") as f:
            json.dump(data, f, default=str)
    except IOError:
        pass


def _fetch_newsletter_articles(start_date: str, end_date: str) -> list[dict]:
    """Fetch and parse WSJ newsletter articles from Gmail for a date range."""
    # Search for WSJ newsletters
    query = f"from:access@interactive.wsj.com after:{start_date} before:{end_date}"
    threads = _gog_gmail_search(query, max_results=5)

    if not threads:
        return []

    all_articles = []

    for thread in threads:
        # Extract message ID — handle both dict and list formats
        msg_id = None
        if isinstance(thread, dict):
            msg_id = thread.get("id") or thread.get("threadId")
            # If thread has messages, get the first message ID
            messages = thread.get("messages", [])
            if messages and isinstance(messages[0], dict):
                msg_id = messages[0].get("id", msg_id)
        elif isinstance(thread, str):
            msg_id = thread

        if not msg_id:
            continue

        msg = _gog_gmail_get(str(msg_id))
        if not msg:
            continue

        # Extract HTML body from message
        html_body = ""
        if isinstance(msg, dict):
            # Try different structures gog might return
            html_body = msg.get("body", "") or msg.get("html", "") or msg.get("snippet", "")

            # Check payload structure
            payload = msg.get("payload", {})
            if payload:
                parts = payload.get("parts", [])
                for part in parts:
                    if part.get("mimeType") == "text/html":
                        body_data = part.get("body", {}).get("data", "")
                        if body_data:
                            try:
                                html_body = base64.urlsafe_b64decode(body_data).decode("utf-8")
                            except Exception:
                                pass

        if html_body:
            articles = _extract_newsletter_articles(html_body)
            all_articles.extend(articles)

    return all_articles


def _scrape_articles_full_text(articles: list[dict], max_scrape: int = 3) -> list[dict]:
    """Scrape full text for articles. Modifies articles in-place."""
    config = get_config()
    max_scrape = config.get("wsj_max_scrape_articles", max_scrape)

    scraped_count = 0
    for article in articles:
        if scraped_count >= max_scrape:
            break

        url = article.get("url", "")
        if not url or "wsj.com" not in url:
            continue

        scraped = _scrape_wsj_article(url)
        if scraped:
            paragraphs = scraped.get("paragraphs", [])
            if paragraphs:
                article["full_text"] = "\n".join(paragraphs)
                article["source"] = "WSJ (full article)"
                scraped_count += 1
            else:
                article["source"] = "WSJ Newsletter"
        else:
            article["source"] = "WSJ Newsletter"

    return articles


def _format_articles_markdown(articles: list[dict], header: str) -> str:
    """Format articles as markdown matching yfinance output pattern."""
    if not articles:
        return ""

    news_str = ""
    for article in articles:
        title = article.get("title", "No title")
        source = article.get("source", "WSJ")
        news_str += f"### {title} (source: {source})\n"

        summary = article.get("summary", "")
        if summary:
            news_str += f"{summary}\n"

        full_text = article.get("full_text", "")
        if full_text:
            # Truncate to first 500 chars for the merged output
            truncated = full_text[:500]
            if len(full_text) > 500:
                truncated += "..."
            news_str += f"{truncated}\n"

        url = article.get("url", "")
        if url:
            news_str += f"Link: {url}\n"

        news_str += "\n"

    return f"{header}\n\n{news_str}"


def get_news_wsj(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """Get WSJ news for a specific ticker.

    1. Search Gmail for WSJ newsletters in date range
    2. Parse newsletters for articles
    3. Filter articles mentioning the ticker/company
    4. Scrape full text for matching articles
    5. Also scrape relevant WSJ section pages
    6. Deduplicate and merge
    """
    config = get_config()
    cache_dir = config.get("data_cache_dir", "data")

    # Check cache
    cache_key = f"news-{ticker}-{start_date}-{end_date}"
    cached = _load_cache(cache_dir, cache_key)
    if cached is not None:
        return _format_articles_markdown(
            cached,
            f"## WSJ News for {ticker.upper()}, from {start_date} to {end_date}:"
        )

    # Fetch newsletter articles
    # Add one day to end_date for Gmail search (exclusive)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    newsletter_articles = _fetch_newsletter_articles(start_date, end_dt.strftime("%Y-%m-%d"))

    # Filter for ticker relevance
    relevant = _filter_articles_for_ticker(newsletter_articles, ticker)

    # Scrape full text for relevant articles
    relevant = _scrape_articles_full_text(relevant)

    # Also scrape WSJ section pages for additional coverage
    ticker_upper = ticker.upper().replace("-USD", "")
    section_articles = []
    if ticker_upper in ("BTC", "ETH", "SOL", "COIN"):
        # Crypto tickers — check markets section
        section_articles = _scrape_wsj_section(WSJ_SECTIONS["markets"])
    elif ticker_upper in ("VOO",):
        section_articles = _scrape_wsj_section(WSJ_SECTIONS["markets"])
    else:
        section_articles = _scrape_wsj_section(WSJ_SECTIONS["markets"])

    # Filter section articles for relevance
    section_relevant = _filter_articles_for_ticker(section_articles, ticker)

    # Merge and deduplicate by URL
    seen_urls = {a.get("url", "") for a in relevant if a.get("url")}
    for article in section_relevant:
        if article.get("url", "") not in seen_urls:
            seen_urls.add(article["url"])
            relevant.append(article)

    # Cache results
    _save_cache(cache_dir, cache_key, relevant)

    if not relevant:
        return f"No WSJ news found for {ticker} between {start_date} and {end_date}"

    return _format_articles_markdown(
        relevant,
        f"## WSJ News for {ticker.upper()}, from {start_date} to {end_date}:"
    )


def get_global_news_wsj(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 5,
) -> str:
    """Get global WSJ news from newsletters and section pages.

    1. Search Gmail for WSJ newsletters in date range
    2. Parse all newsletters for all articles
    3. Scrape full text for top articles
    4. Also scrape WSJ section pages for additional stories
    5. Deduplicate and merge
    """
    config = get_config()
    cache_dir = config.get("data_cache_dir", "data")

    start_dt = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_dt = datetime.strptime(curr_date, "%Y-%m-%d") + timedelta(days=1)

    # Check cache
    cache_key = f"global-{start_date}-{curr_date}"
    cached = _load_cache(cache_dir, cache_key)
    if cached is not None:
        return _format_articles_markdown(
            cached[:limit],
            f"## WSJ Global News, from {start_date} to {curr_date}:"
        )

    # Fetch newsletter articles
    all_articles = _fetch_newsletter_articles(start_date, end_dt.strftime("%Y-%m-%d"))

    # Scrape full text for top articles
    all_articles = _scrape_articles_full_text(all_articles)

    # Also scrape WSJ section pages
    for section_name, section_url in WSJ_SECTIONS.items():
        section_articles = _scrape_wsj_section(section_url)
        # Deduplicate
        seen_titles = {a.get("title", "") for a in all_articles}
        for article in section_articles:
            if article.get("title", "") not in seen_titles:
                seen_titles.add(article["title"])
                all_articles.append(article)

    # Cache results
    _save_cache(cache_dir, cache_key, all_articles)

    if not all_articles:
        return f"No WSJ global news found for {curr_date} (looking back {look_back_days} days)"

    return _format_articles_markdown(
        all_articles[:limit],
        f"## WSJ Global News, from {start_date} to {curr_date}:"
    )
