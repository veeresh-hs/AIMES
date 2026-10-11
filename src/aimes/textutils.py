"""Text / URL / domain helpers with pure-python fallbacks.

Optional packages (bs4, tldextract, rapidfuzz) are used when installed but the
module works without them so the project never hard-fails on a fresh laptop.
"""
from __future__ import annotations

import html as html_lib
import re
from urllib.parse import urlsplit

try:  # optional, better HTML handling
    from bs4 import BeautifulSoup  # type: ignore
    _HAS_BS4 = True
except Exception:  # pragma: no cover
    _HAS_BS4 = False

try:  # optional, correct public-suffix handling
    import tldextract  # type: ignore
    _EXTRACT = tldextract.TLDExtract(suffix_list_urls=())  # offline snapshot only
    _HAS_TLD = True
except Exception:  # pragma: no cover
    _HAS_TLD = False

URL_RE = re.compile(
    r"""(?xi)\b(
        (?:https?://|ftp://|www\d{0,3}\.)
        [^\s<>"'\)\]\},]+
    )"""
)
HTML_HREF_RE = re.compile(r"""(?i)<a[^>]+href\s*=\s*["']([^"']+)["']""")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
WS_RE = re.compile(r"[ \t\r\f\v]+")
NL_RE = re.compile(r"\n{3,}")

_MULTI_SUFFIXES = {
    "co.uk", "ac.uk", "gov.uk", "org.uk", "co.in", "net.in", "org.in", "ac.in",
    "gov.in", "com.au", "co.jp", "com.br", "co.za", "com.cn", "co.kr", "com.mx",
}


def strip_html(raw: str) -> str:
    """Turn an HTML body into readable text without executing anything."""
    if not raw:
        return ""
    if _HAS_BS4:
        try:
            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(["script", "style", "head", "meta", "link"]):
                tag.decompose()
            text = soup.get_text(" ")
        except Exception:
            text = re.sub(r"<[^>]+>", " ", raw)
    else:
        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
        text = re.sub(r"<[^>]+>", " ", text)
    return html_lib.unescape(text)


def normalise_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WS_RE.sub(" ", text)
    text = NL_RE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def clean_body(text_part: str, html_part: str = "", limit: int = 20000) -> str:
    """Produce the text the ML layers see.  Evidence-preserving: URLs stay."""
    parts = []
    if text_part:
        parts.append(text_part)
    if html_part:
        parts.append(strip_html(html_part))
    text = normalise_whitespace("\n".join(parts))
    return text[:limit]


def extract_urls(text_part: str, html_part: str = "") -> list[str]:
    """Collect URLs from plain text and from href attributes using DOM parsing if available."""
    found: list[str] = []
    
    # 1. Extract from plain text
    for match in URL_RE.finditer(text_part or ""):
        found.append(match.group(1))
        
    # 2. Extract from HTML
    if html_part:
        if _HAS_BS4:
            try:
                soup = BeautifulSoup(html_part, "html.parser")
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"].strip()
                    if href.lower().startswith(("http://", "https://", "ftp://", "www.")):
                        found.append(href)
            except Exception:
                pass
        else:
            # Fallback to regex if BeautifulSoup is not installed
            for href in HTML_HREF_RE.findall(html_part):
                href = href.strip()
                if href.lower().startswith(("http://", "https://", "ftp://", "www.")):
                    found.append(href)
                    
        # Catch raw URLs floating in the HTML body text
        for match in URL_RE.finditer(strip_html(html_part)):
            found.append(match.group(1))

    # 3. Deduplicate and normalize
    seen, out = set(), []
    for url in found:
        url = url.rstrip(".,;:!?)\"'")
        if not url:
            continue
        if url.lower().startswith("www."):
            url = "http://" + url
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def host_of(url: str) -> str:
    try:
        host = urlsplit(url if "://" in url else "http://" + url).hostname or ""
    except ValueError:
        return ""
    return host.lower().strip(".")


def registered_domain(host_or_url: str) -> str:
    """eTLD+1 of a host or URL. Uses tldextract when available."""
    host = host_of(host_or_url) if ("/" in host_or_url or "://" in host_or_url) else host_or_url.lower().strip(".")
    if not host:
        return ""
    if re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", host):
        return host
    if _HAS_TLD:
        try:
            ext = _EXTRACT(host)
            if ext.domain and ext.suffix:
                return f"{ext.domain}.{ext.suffix}"
        except Exception:
            pass
    labels = host.split(".")
    if len(labels) < 2:
        return host
    last_two = ".".join(labels[-2:])
    if last_two in _MULTI_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last_two


def domain_of_address(address: str) -> str:
    match = EMAIL_RE.search(address or "")
    if not match:
        return ""
    return match.group(0).split("@", 1)[1].lower().strip(".")


def levenshtein(a: str, b: str) -> int:
    """Edit distance.  rapidfuzz when present, otherwise a DP fallback."""
    try:
        from rapidfuzz.distance import Levenshtein  # type: ignore

        return int(Levenshtein.distance(a, b))
    except Exception:
        pass
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return 1.0 - (levenshtein(a, b) / max(len(a), len(b)))


def defang(url: str) -> str:
    """Render a URL inert for display in the dashboard."""
    return (url.replace("http://", "hxxp://")
               .replace("https://", "hxxps://")
               .replace(".", "[.]"))