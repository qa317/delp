# app.py
# Streamlit app: Packaging trends & EU packaging law chatter monitor
# Sources: GDELT (news), RSS (news/blogs), Official pages (EU Commission / EUR-Lex)
# Storage: SQLite (auto-created) packaging_trends.db by default

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse, quote_plus

import pandas as pd
import requests
import streamlit as st

# Optional deps (we handle if missing)
try:
    import feedparser  # type: ignore
except Exception:
    feedparser = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None


# ----------------------------
# Config
# ----------------------------

APP_TITLE = "EU Packaging Trends Monitor"

DEFAULT_DB = "packaging_trends.db"

DEFAULT_KEYWORDS = [
    "packaging waste",
    "PPWR",
    "packaging and packaging waste regulation",
    "extended producer responsibility",
    "EPR packaging",
    "recyclable packaging",
    "reuse targets",
    "PFAS packaging",
    "compostable packaging",
    "packaging labeling",
]

# These are generally reliable RSS feeds because they come from Google News RSS searches.
# You can add/remove feeds in the sidebar.
DEFAULT_RSS_FEEDS = [
    # EU packaging regulation / PPWR / EPR
    "https://news.google.com/rss/search?q=PPWR%20packaging%20waste%20regulation%20EU&hl=en&gl=EU&ceid=EU:en",
    "https://news.google.com/rss/search?q=packaging%20waste%20regulation%20EU&hl=en&gl=EU&ceid=EU:en",
    "https://news.google.com/rss/search?q=extended%20producer%20responsibility%20packaging%20EU&hl=en&gl=EU&ceid=EU:en",
    # Consumer / trends
    "https://news.google.com/rss/search?q=sustainable%20packaging%20Europe&hl=en&gl=EU&ceid=EU:en",
    "https://news.google.com/rss/search?q=reusable%20packaging%20Europe&hl=en&gl=EU&ceid=EU:en",
    "https://news.google.com/rss/search?q=PFAS%20food%20contact%20packaging%20EU&hl=en&gl=EU&ceid=EU:en",
]

# A couple of official pages you can scrape as “reports/official info”.
# (These are HTML pages; we extract text from main content.)
DEFAULT_OFFICIAL_PAGES = [
    "https://environment.ec.europa.eu/topics/waste-and-recycling/packaging-waste_en",
    "https://environment.ec.europa.eu/topics/waste-and-recycling/packaging-waste/packaging-packaging-waste-regulation_en",
    "https://eur-lex.europa.eu/eli/reg/2025/40/oj/eng",
]

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0 Safari/537.36 (edu-project; streamlit)"
)

REQUEST_TIMEOUT = 20


# ----------------------------
# Helpers
# ----------------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def safe_parse_date(s: str) -> Optional[datetime]:
    """
    Tries a few common formats. Returns UTC datetime if possible.
    """
    if not s:
        return None
    s = s.strip()

    # feedparser often returns time.struct_time; this handler is for strings.
    fmts = [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d",
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    return None


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def request_get(url: str) -> requests.Response:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    return requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)


def extract_text_from_html(html: str) -> str:
    """
    Very lightweight extractor using BeautifulSoup (if installed).
    Falls back to stripping tags crudely if BS4 isn't available.
    """
    if not html:
        return ""

    if BeautifulSoup is None:
        # crude fallback: strip tags
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        return normalize_text(text)

    soup = BeautifulSoup(html, "html.parser")

    # remove junk
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # Prefer main/article content if present
    main = soup.find("main") or soup.find("article")
    target = main if main else soup

    chunks = []
    for p in target.find_all(["p", "li", "h1", "h2", "h3"]):
        t = p.get_text(" ", strip=True)
        if t and len(t) > 40:
            chunks.append(t)

    text = " ".join(chunks) if chunks else target.get_text(" ", strip=True)
    return normalize_text(text)


# ----------------------------
# Database
# ----------------------------

def db_connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            domain TEXT,
            title TEXT,
            published_at TEXT,
            collected_at TEXT NOT NULL,
            text TEXT,
            meta_json TEXT
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_published_at ON documents(published_at);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);")
    conn.commit()


def insert_document(
    conn: sqlite3.Connection,
    source: str,
    url: str,
    title: str,
    published_at: Optional[datetime],
    text: str,
    meta_json: str = "",
) -> bool:
    url = (url or "").strip()
    if not url:
        return False

    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO documents
              (source, url, domain, title, published_at, collected_at, text, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                source,
                url,
                domain_of(url),
                (title or "").strip(),
                published_at.astimezone(timezone.utc).isoformat() if published_at else None,
                now_utc().isoformat(),
                (text or "").strip(),
                meta_json or "",
            ),
        )
        conn.commit()
        # rowcount is unreliable for INSERT OR IGNORE in sqlite; re-check existence
        cur = conn.execute("SELECT 1 FROM documents WHERE url = ? LIMIT 1;", (url,))
        return cur.fetchone() is not None
    except Exception:
        return False


def load_documents_df(conn: sqlite3.Connection, days: int = 30) -> pd.DataFrame:
    since = now_utc() - timedelta(days=days)
    cur = conn.execute(
        """
        SELECT source, url, domain, title, published_at, collected_at, text
        FROM documents
        WHERE (published_at IS NULL) OR (published_at >= ?)
        ORDER BY COALESCE(published_at, collected_at) DESC
        LIMIT 5000;
        """,
        (since.isoformat(),),
    )
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["source", "url", "domain", "title", "published_at", "collected_at", "text"])
    # parse dates for plotting
    for col in ["published_at", "collected_at"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


# ----------------------------
# Collectors
# ----------------------------

@dataclass
class FetchStats:
    seen: int = 0
    inserted: int = 0
    extracted_ok: int = 0
    errors: int = 0


def fetch_gdelt_articles(
    keywords: List[str],
    days: int,
    max_records: int,
) -> Tuple[List[Dict], Dict]:
    """
    Returns (articles, debug_info).
    debug_info includes status code, content type, final URL, and a short preview.
    """
    # GDELT DOC API supports only recent windows; we hard-cap to 90 days.
    days = max(1, min(int(days), 90))

    query = " OR ".join([f'"{k.strip()}"' for k in keywords if k.strip()])
    if not query:
        query = '"packaging waste"'

    # Use start/end datetime
    end_dt = now_utc()
    start_dt = end_dt - timedelta(days=days)

    start_str = start_dt.strftime("%Y%m%d%H%M%S")
    end_str = end_dt.strftime("%Y%m%d%H%M%S")

    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    url = (
        f"{base}?query={quote_plus(query)}"
        f"&mode=ArtList"
        f"&format=json"
        f"&startdatetime={start_str}"
        f"&enddatetime={end_str}"
        f"&maxrecords={int(max_records)}"
        f"&sort=HybridRel"
    )

    debug = {"request_url": url}

    try:
        r = request_get(url)
        debug["status_code"] = r.status_code
        debug["content_type"] = r.headers.get("Content-Type", "")
        debug["final_url"] = r.url
        debug["preview"] = (r.text or "")[:300].replace("\n", " ")

        if r.status_code != 200:
            return [], debug

        # Some blocking pages return HTML instead of JSON
        ct = (r.headers.get("Content-Type", "") or "").lower()
        if "json" not in ct and not (r.text.strip().startswith("{") and r.text.strip().endswith("}")):
            return [], debug

        data = r.json()
        # GDELT usually returns {"articles":[...], ...}
        articles = data.get("articles", []) or []
        return articles, debug
    except Exception as e:
        debug["error"] = repr(e)
        return [], debug


def collect_gdelt(
    conn: sqlite3.Connection,
    keywords: List[str],
    days: int,
    max_records: int,
    status_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[FetchStats, Dict]:
    stats = FetchStats()
    articles, debug = fetch_gdelt_articles(keywords, days, max_records)

    stats.seen = len(articles)
    if status_cb:
        status_cb(f"GDELT returned {len(articles)} article records")

    for a in articles:
        try:
            url = a.get("url") or ""
            title = a.get("title") or ""
            # GDELT "seendate" often like "20250122153000"
            seendate = a.get("seendate") or ""
            published_at = None
            if seendate and re.fullmatch(r"\d{14}", str(seendate)):
                published_at = datetime.strptime(seendate, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)

            # Try to fetch and extract real page text
            text = ""
            try:
                page = request_get(url)
                if page.status_code == 200:
                    text = extract_text_from_html(page.text)
            except Exception:
                text = ""

            if text:
                stats.extracted_ok += 1

            inserted = insert_document(
                conn,
                source="GDELT",
                url=url,
                title=title,
                published_at=published_at,
                text=text,
                meta_json="",
            )
            if inserted:
                stats.inserted += 1

        except Exception:
            stats.errors += 1

    return stats, debug


def collect_rss(
    conn: sqlite3.Connection,
    feed_urls: List[str],
    status_cb: Optional[Callable[[str], None]] = None,
    max_per_feed: int = 30,
    fetch_full_text: bool = True,
) -> FetchStats:
    stats = FetchStats()

    if feedparser is None:
        if status_cb:
            status_cb("RSS: feedparser is not installed. Add `feedparser` to requirements.txt.")
        stats.errors += 1
        return stats

    for feed_url in [u.strip() for u in feed_urls if u.strip()]:
        try:
            d = feedparser.parse(feed_url)
            entries = d.entries or []
            stats.seen += len(entries)

            if status_cb:
                status_cb(f"RSS parsed {len(entries)} entries from {feed_url}")

            for e in entries[:max_per_feed]:
                url = (getattr(e, "link", "") or "").strip()
                title = (getattr(e, "title", "") or "").strip()

                # published / updated
                published_at = None
                if getattr(e, "published_parsed", None):
                    # struct_time -> datetime
                    try:
                        published_at = datetime.fromtimestamp(time.mktime(e.published_parsed), tz=timezone.utc)
                    except Exception:
                        published_at = None
                elif getattr(e, "updated_parsed", None):
                    try:
                        published_at = datetime.fromtimestamp(time.mktime(e.updated_parsed), tz=timezone.utc)
                    except Exception:
                        published_at = None

                # Use summary as fallback, optionally fetch full article
                text = ""
                summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""
                summary = normalize_text(summary)

                if fetch_full_text and url:
                    try:
                        page = request_get(url)
                        if page.status_code == 200:
                            text = extract_text_from_html(page.text)
                    except Exception:
                        text = ""

                if not text and summary:
                    text = summary

                if text:
                    stats.extracted_ok += 1

                inserted = insert_document(
                    conn,
                    source="RSS",
                    url=url,
                    title=title,
                    published_at=published_at,
                    text=text,
                    meta_json="",
                )
                if inserted:
                    stats.inserted += 1

        except Exception:
            stats.errors += 1

    return stats


def collect_official_pages(
    conn: sqlite3.Connection,
    page_urls: List[str],
    status_cb: Optional[Callable[[str], None]] = None,
) -> FetchStats:
    stats = FetchStats()

    for url in [u.strip() for u in page_urls if u.strip()]:
        stats.seen += 1
        try:
            r = request_get(url)
            if status_cb:
                status_cb(f"Official page GET {r.status_code}: {url}")

            if r.status_code != 200:
                stats.errors += 1
                continue

            text = extract_text_from_html(r.text)
            if text:
                stats.extracted_ok += 1

            inserted = insert_document(
                conn,
                source="Official",
                url=url,
                title="",
                published_at=None,
                text=text,
                meta_json="",
            )
            if inserted:
                stats.inserted += 1

        except Exception:
            stats.errors += 1

    return stats


# ----------------------------
# Analysis (simple + robust)
# ----------------------------

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between
both but by can did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no
nor not of off on once only or other our ours ourselves out over own same she should so some such
than that the their theirs them themselves then there these they this those through to too under
until up very was we were what when where which while who whom why with you your yours yourself yourselves
""".split())


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    tokens = re.findall(r"[a-z]{3,}", text)
    return [t for t in tokens if t not in STOPWORDS]


def top_terms(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    from collections import Counter

    c = Counter()
    for t in df["text"].fillna("").astype(str):
        c.update(tokenize(t))
    items = c.most_common(n)
    return pd.DataFrame(items, columns=["term", "count"])


def risk_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple “concern/expectation/law change” signals with keyword buckets.
    """
    buckets = {
        "Compliance / deadlines": ["deadline", "compliance", "enforcement", "penalty", "obligation", "ban", "banned"],
        "Cost / operations": ["cost", "price", "supply", "logistics", "capex", "opex", "investment"],
        "Materials / chemicals": ["pfas", "bpa", "recycled", "bioplastic", "compostable", "paper", "aluminium", "glass"],
        "Reuse / refill": ["reuse", "refill", "return", "deposit", "take-back"],
        "Labeling / claims": ["label", "labels", "claims", "greenwashing", "recyclable", "recycling"],
    }

    rows = []
    for _, r in df.iterrows():
        text = (r.get("title", "") or "") + " " + (r.get("text", "") or "")
        lt = text.lower()
        hit = []
        for b, keys in buckets.items():
            if any(k in lt for k in keys):
                hit.append(b)
        rows.append(", ".join(hit))

    out = df.copy()
    out["signals"] = rows
    # Summary table
    exploded = out.assign(signals=out["signals"].str.split(r"\s*,\s*")).explode("signals")
    exploded["signals"] = exploded["signals"].fillna("").astype(str).str.strip()
    exploded = exploded[exploded["signals"] != ""]
    if exploded.empty:
        return pd.DataFrame(columns=["signal", "count"])
    return (
        exploded.groupby("signals", as_index=False)
        .size()
        .rename(columns={"signals": "signal", "size": "count"})
        .sort_values("count", ascending=False)
    )


# ----------------------------
# Diagnostics
# ----------------------------

def diagnose_url(url: str) -> Dict:
    d: Dict = {"url": url}
    try:
        r = request_get(url)
        d["status_code"] = r.status_code
        d["final_url"] = r.url
        d["content_type"] = r.headers.get("Content-Type", "")
        txt = r.text or ""
        d["preview"] = txt[:250].replace("\n", " ")
        d["is_probably_xml"] = ("xml" in (d["content_type"] or "").lower()) or txt.lstrip().startswith("<?xml")
        d["is_probably_json"] = ("json" in (d["content_type"] or "").lower()) or txt.lstrip().startswith("{")
        return d
    except Exception as e:
        d["error"] = repr(e)
        return d


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Collect public info about packaging trends (news + RSS + official pages), store it locally, and analyze what’s discussed most.")

with st.sidebar:
    st.header("Settings")

    db_path = st.text_input("SQLite DB path", value=DEFAULT_DB, help="The DB file will be created automatically if it doesn't exist.")
    keywords_text = st.text_area("Keywords (one per line)", value="\n".join(DEFAULT_KEYWORDS), height=160)
    keywords = [k.strip() for k in keywords_text.splitlines() if k.strip()]

    st.divider()
    st.subheader("Collectors")

    use_gdelt = st.checkbox("Use GDELT (news)", value=True)
    days_collect = st.slider("Lookback window (days)", min_value=1, max_value=90, value=14, help="GDELT DOC API works reliably within ~90 days.")
    max_records = st.slider("Max GDELT records", min_value=10, max_value=500, value=200, step=10)

    use_rss = st.checkbox("Use RSS feeds", value=True)
    rss_text = st.text_area("RSS feed URLs (one per line)", value="\n".join(DEFAULT_RSS_FEEDS), height=170)
    rss_feeds = [u.strip() for u in rss_text.splitlines() if u.strip()]
    rss_full_text = st.checkbox("For RSS: fetch full article text", value=True)

    use_official = st.checkbox("Use official pages", value=True)
    official_text = st.text_area("Official page URLs (one per line)", value="\n".join(DEFAULT_OFFICIAL_PAGES), height=120)
    official_pages = [u.strip() for u in official_text.splitlines() if u.strip()]

    st.divider()
    danger = st.checkbox("I understand: reset DB deletes all stored data", value=False)
    reset_db = st.button("Reset DB", disabled=not danger)


conn = db_connect(db_path)
init_db(conn)

if reset_db:
    conn.close()
    # Delete tables by recreating file content (simple approach)
    import os
    try:
        os.remove(db_path)
        st.success("DB deleted. Reloading…")
    except Exception as e:
        st.error(f"Could not delete DB: {e}")
    st.stop()


tab_collect, tab_dashboard, tab_explore, tab_diag = st.tabs(["Collect", "Dashboard", "Explore data", "Diagnostics"])


with tab_collect:
    st.subheader("Collect data")
    st.write("Click **Run collection** to fetch new items and store them in your SQLite database.")

    run = st.button("Run collection", type="primary")

    status_box = st.empty()

    def cb(msg: str) -> None:
        status_box.info(msg)

    if run:
        all_msgs = []
        total_inserted = 0

        if use_gdelt:
            cb("Collecting from GDELT…")
            gdelt_stats, gdelt_debug = collect_gdelt(conn, keywords, days_collect, max_records, status_cb=cb)
            total_inserted += gdelt_stats.inserted
            st.success(f"✅ GDELT seen {gdelt_stats.seen} | extracted {gdelt_stats.extracted_ok} | inserted {gdelt_stats.inserted} | errors {gdelt_stats.errors}")
            with st.expander("GDELT debug"):
                st.json(gdelt_debug)

        if use_rss:
            cb("Collecting from RSS…")
            rss_stats = collect_rss(conn, rss_feeds, status_cb=cb, fetch_full_text=rss_full_text)
            total_inserted += rss_stats.inserted
            st.success(f"✅ RSS entries seen {rss_stats.seen} | extracted {rss_stats.extracted_ok} | inserted {rss_stats.inserted} | errors {rss_stats.errors}")

        if use_official:
            cb("Collecting from official pages…")
            off_stats = collect_official_pages(conn, official_pages, status_cb=cb)
            total_inserted += off_stats.inserted
            st.success(f"✅ Official pages seen {off_stats.seen} | extracted {off_stats.extracted_ok} | inserted {off_stats.inserted} | errors {off_stats.errors}")

        if total_inserted == 0:
            st.warning(
                "No new documents were inserted. Go to the **Diagnostics** tab to see which URLs are returning "
                "non-RSS/blocked content or whether your GDELT query is too strict."
            )
        else:
            st.success(f"Done — inserted {total_inserted} new documents.")


with tab_dashboard:
    st.subheader("Dashboard")
    days_view = st.slider("Show last N days", min_value=7, max_value=365, value=60, step=1)
    df = load_documents_df(conn, days=days_view)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents loaded", f"{len(df):,}")
    c2.metric("Sources", f"{df['source'].nunique() if not df.empty else 0}")
    c3.metric("Domains", f"{df['domain'].nunique() if not df.empty else 0}")
    c4.metric("With text", f"{int((df['text'].fillna('').str.len() > 0).sum()) if not df.empty else 0}")

    if df.empty:
        st.info("No data yet. Use the **Collect** tab first.")
    else:
        # Timeline
        st.write("### Volume over time")
        tmp = df.copy()
        tmp["date"] = (tmp["published_at"].fillna(tmp["collected_at"])).dt.date
        vol = tmp.groupby(["date", "source"], as_index=False).size().rename(columns={"size": "count"})

        import matplotlib.pyplot as plt  # allowed

        fig = plt.figure()
        for src in sorted(vol["source"].unique()):
            sub = vol[vol["source"] == src].sort_values("date")
            plt.plot(sub["date"], sub["count"], label=src)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)

        st.write("### Top terms")
        terms = top_terms(df, n=30)
        st.dataframe(terms, use_container_width=True)

        st.write("### Risk / expectation signals (simple keyword buckets)")
        sig = risk_signals(df)
        if sig.empty:
            st.info("No signals detected yet (or no text extracted). Try enabling “fetch full article text” for RSS.")
        else:
            st.dataframe(sig, use_container_width=True)


with tab_explore:
    st.subheader("Explore stored documents")
    df = load_documents_df(conn, days=365)

    if df.empty:
        st.info("No data yet. Use the **Collect** tab first.")
    else:
        sources = ["All"] + sorted(df["source"].dropna().unique().tolist())
        pick_source = st.selectbox("Filter by source", sources, index=0)
        q = st.text_input("Search in title/text", value="")

        view = df.copy()
        if pick_source != "All":
            view = view[view["source"] == pick_source]
        if q.strip():
            qq = q.lower().strip()
            view = view[
                view["title"].fillna("").str.lower().str.contains(qq)
                | view["text"].fillna("").str.lower().str.contains(qq)
            ]

        view = view.head(500).copy()
        view["published_at"] = view["published_at"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
        view["collected_at"] = view["collected_at"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
        st.dataframe(
            view[["source", "published_at", "domain", "title", "url"]],
            use_container_width=True,
            hide_index=True,
        )

        st.caption("Tip: click a URL to open it. If text extraction is empty, some sites block scraping or rely on scripts.")


with tab_diag:
    st.subheader("Diagnostics (why do I get 0?)")
    st.write(
        "This checks each URL and shows whether it looks like RSS XML / JSON, its HTTP status, redirects, and a preview."
    )

    colA, colB = st.columns(2)
    with colA:
        test_gdelt = st.button("Test GDELT request")
    with colB:
        test_all_rss = st.button("Test RSS URLs")

    if test_gdelt:
        st.write("### GDELT")
        articles, debug = fetch_gdelt_articles(keywords=keywords[:6], days=days_collect, max_records=max_records)
        st.json(debug)
        st.write(f"Articles returned: {len(articles)}")
        if len(articles) == 0:
            st.warning(
                "GDELT returned 0. Try fewer/simpler keywords (e.g., just “packaging waste” and “PPWR”), "
                "or reduce the lookback window."
            )

    if test_all_rss:
        st.write("### RSS URL tests")
        diag_rows = []
        for u in rss_feeds:
            diag_rows.append(diagnose_url(u))
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

        bad = [r for r in diag_rows if r.get("status_code", 0) != 200 or not r.get("is_probably_xml")]
        if bad:
            st.warning(
                "Some feeds don’t look like RSS XML or returned non-200 status. Replace those URLs with valid RSS feeds."
            )

    st.write("### Test any URL")
    any_url = st.text_input("Paste a URL to test", value="")
    if any_url.strip():
        st.json(diagnose_url(any_url.strip()))
