"""Free tools: PDF text extraction, DuckDuckGo search, page fetch, date + domain checks."""
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests

MAX_CV_CHARS = 3500
CACHE_FILE = Path(".cache") / "search_cache.json"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml", "Accept-Language": "en-US,en;q=0.8"}


# ---------------------------------------------------------------- PDF
def pdf_extract(file_bytes: bytes) -> str:
    """Extract text from a text-based PDF and truncate to 3,500 chars."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    parts = []
    for page in reader.pages[:12]:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return re.sub(r"\s+", " ", " ".join(parts)).strip()[:MAX_CV_CHARS]


# ---------------------------------------------------------------- cache
def _load_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_FILE.parent.mkdir(exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------- search
def _ddg_html(query: str, n: int) -> list[dict]:
    """Last-resort search: DuckDuckGo's plain HTML endpoint (no package, no key)."""
    try:
        r = requests.post("https://html.duckduckgo.com/html/", data={"q": query}, headers=UA, timeout=10)
        if r.status_code != 200:
            return []
        out = []
        for m in re.finditer(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', r.text, re.S):
            href = m.group(1).replace("&amp;", "&")
            if "uddg=" in href:
                href = unquote(parse_qs(urlparse(href).query).get("uddg", [""])[0])
            if href.startswith("//"):
                href = "https:" + href
            if not href.startswith("http"):
                continue
            out.append({"title": re.sub(r"<[^>]+>", "", m.group(2)).strip(), "url": href, "snippet": ""})
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


def ddg_search(query: str, max_results: int = 4) -> list[dict]:
    """Free search: ddgs (auto, bing backends) -> DuckDuckGo HTML endpoint -> cached results."""
    key = query.lower().strip()
    cache = _load_cache()
    DDGS = None
    try:
        from ddgs import DDGS
    except Exception:
        try:
            from duckduckgo_search import DDGS
        except Exception:
            DDGS = None
    if DDGS is not None:
        new_pkg = DDGS.__module__.startswith("ddgs")
        for backend in ("auto", "bing") if new_pkg else ("auto",):
            try:
                kw = {"max_results": max_results}
                if new_pkg and backend != "auto":
                    kw["backend"] = backend
                hits = list(DDGS().text(query, **kw))
                res = [{"title": h.get("title", ""), "url": h.get("href") or h.get("url", ""),
                        "snippet": h.get("body", "")} for h in hits]
                res = [r for r in res if r["url"]]
                if res:
                    cache[key] = res
                    _save_cache(cache)
                    return res
            except Exception:
                time.sleep(1.5)
    res = _ddg_html(query, max_results)
    if res:
        cache[key] = res
        _save_cache(cache)
        return res
    return cache.get(key, [])


AGGREGATORS = (
    "scholars4dev", "scholarshipportal", "mastersportal", "phdportal", "bachelorsportal", "studyportals",
    "opportunitydesk", "scholarshipsads", "scholarshipdb", "fundsforngos", "afterschoolafrica",
    "scholarshipscorner", "scholarshiproar", "opportunitiescorners", "fastweb", "bigfuture", "unigo",
    "youtube.com", "facebook.com", "linkedin.com", "reddit.com", "quora.com", "medium.com",
    "wikipedia.org", "pinterest.com", "instagram.com", "twitter.com", "x.com", "tiktok.com",
)


def is_aggregator(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    for a in AGGREGATORS:
        if host == a or host.endswith("." + a) or ("." not in a and a in host):
            return True
    return False


def fetch_many(urls: list[str], workers: int = 6) -> dict:
    """Fetch several pages concurrently. Returns {url: (ok, text)}."""
    urls = list(dict.fromkeys(urls))
    if not urls:
        return {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(zip(urls, ex.map(fetch_page, urls)))


def fetch_page(url: str, max_chars: int = 6000) -> tuple[bool, str]:
    """Open a page. Returns (ok, plain_text). ok=False if unreachable or HTTP >= 400."""
    try:
        r = requests.get(url, headers=UA, timeout=8, allow_redirects=True)
        if r.status_code >= 400:
            return False, ""
        if "html" not in r.headers.get("content-type", "text/html").lower():
            return True, ""
        html = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", r.text)
        text = re.sub(r"(?s)<[^>]+>", " ", html)
        text = re.sub(r"&nbsp;|&amp;", " ", text)
        return True, re.sub(r"\s+", " ", text).strip()[:max_chars]
    except Exception:
        return False, ""


# ---------------------------------------------------------------- dates
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
_P_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MON}\.?,?\s+(\d{{4}})\b", re.I)
_P_MDY = re.compile(rf"\b{_MON}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_P_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DEADLINE_WORDS = ("deadline", "apply by", "applications close", "closing date", "close on",
                  "due date", "submit by", "application period", "applications must")


def find_dates(text: str) -> list[tuple[date, int]]:
    found = []
    for m in _P_DMY.finditer(text):
        found.append((m.group(1), m.group(2), m.group(3), m.start()))
    for m in _P_MDY.finditer(text):
        found.append((m.group(2), m.group(1), m.group(3), m.start()))
    out = []
    for d, mon, y, pos in found:
        try:
            out.append((date(int(y), _MONTHS[mon[:3].lower()], int(d)), pos))
        except ValueError:
            pass
    for m in _P_ISO.finditer(text):
        try:
            out.append((date(int(m.group(1)), int(m.group(2)), int(m.group(3))), m.start()))
        except ValueError:
            pass
    return out


def extract_deadline(text: str, today: Optional[date] = None) -> tuple[Optional[date], str]:
    """Find a deadline-like date in page text. Returns (date, 'future'|'expired'|'none')."""
    today = today or date.today()
    cands = []
    for d, pos in find_dates(text):
        ctx = text[max(0, pos - 140): pos + 60].lower()
        if any(w in ctx for w in DEADLINE_WORDS):
            cands.append(d)
    future = sorted(d for d in cands if d >= today)
    if future:
        return future[0], "future"
    if cands:
        return max(cands), "expired"
    return None, "none"


# ---------------------------------------------------------------- domain trust
TRUSTED_SUFFIXES = (
    ".edu", ".gov", ".mil", ".ac.uk", ".gov.uk", ".europa.eu", ".gc.ca", ".gov.au", ".edu.au",
    ".go.jp", ".ac.jp", ".go.kr", ".ac.kr", ".gov.tr", ".edu.tr", ".gov.cn", ".edu.cn",
    ".gov.my", ".edu.my", ".gov.pk", ".edu.pk", ".gouv.fr", ".gov.it", ".gov.hu", ".gov.nl",
)
TRUSTED_DOMAINS = (
    "daad.de", "chevening.org", "fulbrightonline.org", "fulbright.org", "gatescambridge.org",
    "stipendiumhungaricum.hu", "campuschina.org", "csc.edu.cn", "studyinjapan.go.jp", "si.se",
    "studyinsweden.se", "studyinnl.org", "nuffic.nl", "campusfrance.org", "studyinitaly.esteri.it",
    "esteri.it", "ox.ac.uk", "cam.ac.uk", "vanier.gc.ca", "turkiyeburslari.gov.tr",
    "erasmus-plus.ec.europa.eu", "eacea.ec.europa.eu",
    "rhodeshouse.ox.ac.uk", "cscuk.fcdo.gov.uk", "dfat.gov.au", "studyaustralia.gov.au",
)


def is_trusted(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if not host:
        return False
    if any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS):
        return True
    if any(host.endswith(s) for s in TRUSTED_SUFFIXES):
        return True
    # European university domains such as uni-xyz.de, tu-xyz.de, univ-xyz.fr, tum.de
    return bool(re.match(r"^(uni|tu|univ|hu|fu|rwth|kth|lu|uu|su)[-.]", host)) or host.endswith(
        (".uni-heidelberg.de", "tum.de", "kth.se", "lu.se", "uu.se", "su.se", "uva.nl", "tudelft.nl"))


# ---------------------------------------------------------------- JSON
def extract_json(text: str):
    """Tolerant JSON extraction: strips fences, finds first balanced object or array."""
    if not text:
        raise ValueError("empty response")
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):          # first complete object/array wins; trailing junk is ignored
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(text[i:])
                return obj
            except Exception:
                continue
    raise ValueError("no valid JSON found in model output")
