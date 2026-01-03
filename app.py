# app.py
# 3 tabs: Collect data / Explore data / Analyze (Dashboard)
# News sources:
#   - GDELT (with auto-retries: user window -> 90 days -> no dates + fallback query)
#   - RSS feeds (default: Google News RSS searches)
#   - GNews library (optional extra news source)
#   - Official pages (EU Commission / EUR-Lex)
#
# Run:
#   pip install streamlit requests feedparser pandas beautifulsoup4 python-dateutil gnews
#   streamlit run app.py

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as dtparser

# Optional: GNews library (like your example)
try:
    from gnews import GNews  # type: ignore
except Exception:
    GNews = None

# ----------------------------
# Defaults
# ----------------------------

DEFAULT_DB = "packaging_trends.db"
GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

DEFAULT_KEYWORDS = [
    "packaging waste",
    "PPWR",
    "sustainable packaging",
    "recyclable packaging",
    "recycled content packaging",
    "reusable packaging",
    "EPR packaging",
    "deposit return scheme",
    "PFAS packaging",
    "packaging labeling",
]

# Reliable RSS (Google News RSS queries) — usually returns entries quickly
DEFAULT_RSS_FEEDS = [
    "https://news.google.com/rss/search?q=packaging%20waste%20EU&hl=en&gl=EU&ceid=EU:en",
    "https://news.google.com/rss/search?q=PPWR%20packaging%20waste%20regulation&hl=en&gl=EU&ceid=EU:en",
    "https://news.google.com/rss/search?q=sustainable%20packaging%20Europe&hl=en&gl=EU&ceid=EU:en",
    "https://news.google.com/rss/search?q=reusable%20packaging%20Europe&hl=en&gl=EU&ceid=EU:en",
    "https://news.google.com/rss/search?q=EPR%20packaging%20Europe&hl=en&gl=EU&ceid=EU:en",
]

DEFAULT_OFFICIAL_URLS = [
    "https://environment.ec.europa.eu/topics/waste-and-recycling/packaging-waste_en",
    "https://eur-lex.europa.eu/eli/reg/2025/40/oj/eng",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PackagingTrendsMonitor/1.0)",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "*/*",
}

# ----------------------------
# DB
# ----------------------------

def db_connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_db(con: sqlite3.Connection) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL,
        url TEXT NOT NULL UNIQUE,
        domain TEXT,
        title TEXT,
        published_at TEXT,
        collected_at TEXT NOT NULL,
        text TEXT
    );
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_docs_source ON documents(source);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_docs_published ON documents(published_at);")
    con.commit()

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def insert_doc(
    con: sqlite3.Connection,
    source: str,
    url: str,
    title: str,
    published_at: Optional[datetime],
    text: str,
) -> bool:
    url = (url or "").strip()
    if not url:
        return False
    try:
        con.execute(
            """
            INSERT OR IGNORE INTO documents
              (source, url, domain, title, published_at, collected_at, text)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                source,
                url,
                domain_of(url),
                (title or "").strip(),
                published_at.astimezone(timezone.utc).isoformat() if published_at else None,
                datetime.now(timezone.utc).isoformat(),
                (text or "").strip(),
            ),
        )
        con.commit()
        # True if exists (inserted now or already existed)
        cur = con.execute("SELECT 1 FROM documents WHERE url=? LIMIT 1;", (url,))
        return cur.fetchone() is not None
    except Exception:
        return False

def load_docs(con: sqlite3.Connection, days: int = 180, limit: int = 5000) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    cur = con.execute(
        """
        SELECT source, url, domain, title, published_at, collected_at, text
        FROM documents
        WHERE (published_at IS NULL) OR (published_at >= ?)
        ORDER BY COALESCE(published_at, collected_at) DESC
        LIMIT ?;
        """,
        (since.isoformat(), limit),
    )
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["source", "url", "domain", "title", "published_at", "collected_at", "text"])
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["collected_at"] = pd.to_datetime(df["collected_at"], errors="coerce", utc=True)
    return df

# ----------------------------
# Extraction (fast)
# ----------------------------

def fetch(url: str, timeout: int = 20) -> requests.Response:
    return requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)

def html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article")
    target = main if main else soup

    parts: List[str] = []
    for p in target.find_all(["h1", "h2", "h3", "p", "li"]):
        t = p.get_text(" ", strip=True)
        if t and len(t) >= 40:
            parts.append(t)

    text = " ".join(parts) if parts else target.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()

def parse_date_any(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = dtparser.parse(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

# ----------------------------
# GDELT (fixed: auto-retries + fallback)
# ----------------------------

def build_gdelt_query(keywords: List[str]) -> str:
    # Quote phrases, leave single words. Limit to avoid overly long queries.
    terms = []
    for k in keywords:
        k = k.strip()
        if not k:
            continue
        terms.append(f'"{k}"' if " " in k else k)
    if len(terms) > 12:
        terms = terms[:12]
    return " OR ".join(terms) if terms else "packaging waste EU"

def gdelt_call(params: Dict) -> Tuple[List[Dict], Dict]:
    debug: Dict = {"endpoint": GDELT_ENDPOINT, "params": params}
    r = requests.get(GDELT_ENDPOINT, params=params, headers=HEADERS, timeout=25)
    debug["status_code"] = r.status_code
    debug["final_url"] = r.url
    debug["content_type"] = r.headers.get("Content-Type", "")
    debug["preview"] = (r.text or "")[:250].replace("\n", " ")

    if r.status_code != 200:
        return [], debug

    ct = (r.headers.get("Content-Type") or "").lower()
    if "json" not in ct and not (r.text or "").lstrip().startswith("{"):
        return [], debug

    data = r.json()
    return (data.get("articles", []) or []), debug

def gdelt_request(query: str, days: int, max_records: int) -> Tuple[List[Dict], Dict]:
    """
    Try in order:
      1) user date window (capped to 90)
      2) retry with 90 days
      3) retry without dates (let GDELT decide)
    """
    days = max(1, min(int(days), 90))
    end = datetime.now(timezone.utc)

    # 1) user window
    start = end - timedelta(days=days)
    p1 = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "sort": "HybridRel",
        "maxrecords": int(max_records),
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    a1, d1 = gdelt_call(p1)
    if len(a1) > 0:
        return a1, d1

    # 2) 90-day retry
    if days < 90:
        start90 = end - timedelta(days=90)
        p2 = dict(p1)
        p2["startdatetime"] = start90.strftime("%Y%m%d%H%M%S")
        p2["enddatetime"] = end.strftime("%Y%m%d%H%M%S")
        a2, d2 = gdelt_call(p2)
        d1["retry_90_days"] = d2
        if len(a2) > 0:
            return a2, d1

    # 3) no dates
    p3 = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "sort": "HybridRel",
        "maxrecords": int(max_records),
    }
    a3, d3 = gdelt_call(p3)
    d1["retry_no_dates"] = d3
    return a3, d1

def collect_gdelt(
    con: sqlite3.Connection,
    keywords: List[str],
    days: int,
    max_records: int,
    fetch_full_text: bool,
    max_full_text: int,
) -> Tuple[int, int, int, Dict]:
    query = build_gdelt_query(keywords)
    articles, debug = gdelt_request(query=query, days=days, max_records=max_records)

    # Fallback query if still 0
    if len(articles) == 0:
        fallback_query = "packaging waste EU OR PPWR OR sustainable packaging Europe"
        a2, d2 = gdelt_request(query=fallback_query, days=days, max_records=max_records)
        debug["fallback_query"] = fallback_query
        debug["fallback_debug"] = d2
        articles = a2

    seen = len(articles)
    inserted = 0
    extracted_ok = 0
    budget = max(0, int(max_full_text))

    for idx, a in enumerate(articles, start=1):
        url = (a.get("url") or "").strip()
        title = (a.get("title") or "").strip()
        seendate = (a.get("seendate") or "").strip()

        published_at = None
        if re.fullmatch(r"\d{14}", seendate):
            try:
                published_at = datetime.strptime(seendate, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            except Exception:
                published_at = None

        # Fast mode: store title as text; optional full text for first N
        text = title
        if fetch_full_text and idx <= budget and url:
            try:
                page = fetch(url, timeout=20)
                if page.status_code == 200:
                    full = html_to_text(page.text)
                    if full:
                        text = full
            except Exception:
                pass

        if text:
            extracted_ok += 1
        if insert_doc(con, "GDELT", url, title, published_at, text):
            inserted += 1

    return inserted, seen, extracted_ok, debug

# ----------------------------
# RSS (fast + reliable)
# ----------------------------

def collect_rss(
    con: sqlite3.Connection,
    feed_urls: List[str],
    fetch_full_text: bool,
    max_per_feed: int,
) -> Tuple[int, int, int]:
    inserted = 0
    entries_seen = 0
    extracted_ok = 0

    for feed_url in [u.strip() for u in feed_urls if u.strip()]:
        d = feedparser.parse(feed_url)
        entries = d.entries or []
        entries_seen += len(entries)

        for e in entries[: int(max_per_feed)]:
            url = (getattr(e, "link", "") or "").strip()
            title = (getattr(e, "title", "") or "").strip()

            published_at = None
            if getattr(e, "published", None):
                published_at = parse_date_any(getattr(e, "published", ""))
            elif getattr(e, "updated", None):
                published_at = parse_date_any(getattr(e, "updated", ""))

            summary_html = (getattr(e, "summary", "") or getattr(e, "description", "") or "").strip()
            summary_text = re.sub(
                r"\s+",
                " ",
                BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True),
            )

            text = summary_text or title

            if fetch_full_text and url:
                try:
                    page = fetch(url, timeout=20)
                    if page.status_code == 200:
                        full = html_to_text(page.text)
                        if len(full) > len(text):
                            text = full
                except Exception:
                    pass

            if text:
                extracted_ok += 1
            if insert_doc(con, "RSS", url, title, published_at, text):
                inserted += 1

    return inserted, entries_seen, extracted_ok

# ----------------------------
# GNews library (extra news source)
# ----------------------------

def collect_gnews(
    con: sqlite3.Connection,
    keywords: List[str],
    lookback_days: int,
    country: str,
    language: str,
    max_results: int,
) -> Tuple[int, int]:
    """
    Uses gnews library to fetch results similar to your example.
    Stores title/description as text for speed.
    """
    if GNews is None:
        return 0, 0

    today_dt = datetime.now()
    start_dt = today_dt - timedelta(days=int(lookback_days))

    gn = GNews(
        language=language,
        country=country,
        start_date=start_dt,
        end_date=today_dt,
        max_results=int(max_results),
    )

    # Keep topics small so it's fast
    topics = keywords[:6] if keywords else ["packaging waste", "PPWR", "sustainable packaging"]
    inserted = 0
    seen = 0

    for topic in topics:
        try:
            news = gn.get_news(topic)
        except Exception:
            continue

        for a in news:
            seen += 1
            title = (a.get("title") or "").strip()
            url = (a.get("url") or "").strip()
            desc = (a.get("description") or "").strip()
            published = a.get("published date") or a.get("published_date") or ""
            published_at = parse_date_any(str(published)) if published else None

            text = (desc or title)
            if insert_doc(con, "GNews", url, title, published_at, text):
                inserted += 1

    return inserted, seen

# ----------------------------
# Official pages
# ----------------------------

def collect_official(
    con: sqlite3.Connection,
    urls: List[str],
) -> Tuple[int, int]:
    inserted = 0
    extracted_ok = 0
    for url in [u.strip() for u in urls if u.strip()]:
        try:
            r = fetch(url, timeout=25)
            if r.status_code != 200:
                continue
            text = html_to_text(r.text)
            if not text:
                continue
            extracted_ok += 1
            if insert_doc(con, "Official", url, url, None, text):
                inserted += 1
        except Exception:
            continue
    return inserted, extracted_ok

# ----------------------------
# Analyze helpers
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
    toks = re.findall(r"[a-z]{3,}", text)
    return [t for t in toks if t not in STOPWORDS]

def top_terms(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    from collections import Counter
    c = Counter()
    for t in df["text"].fillna("").astype(str):
        c.update(tokenize(t))
    return pd.DataFrame(c.most_common(n), columns=["term", "count"])

def simple_signals(df: pd.DataFrame) -> pd.DataFrame:
    buckets = {
        "Compliance / enforcement": ["compliance", "enforcement", "penalty", "fine", "obligation", "deadline"],
        "Cost / supply": ["cost", "price", "supply", "shortage", "logistics"],
        "Reuse / refill": ["reuse", "reusable", "refill", "return", "deposit"],
        "Recycling / recycled content": ["recycle", "recyclable", "recycled", "content"],
        "Chemicals (PFAS etc.)": ["pfas", "bpa", "chemical"],
        "Labeling / greenwashing": ["label", "labels", "claim", "greenwashing"],
    }

    counts = {k: 0 for k in buckets}
    for _, row in df.iterrows():
        text = f"{row.get('title','')} {row.get('text','')}".lower()
        for b, keys in buckets.items():
            if any(k in text for k in keys):
                counts[b] += 1

    out = pd.DataFrame(sorted(counts.items(), key=lambda x: x[1], reverse=True), columns=["signal", "count"])
    return out[out["count"] > 0]


# ============================
# Streamlit UI (3 tabs)
# ============================

st.set_page_config(page_title="Packaging Trends Monitor", layout="wide")
st.title("📦 Packaging Trends Monitor (Fixed + More News Sources)")
st.caption("Collect packaging trend info (GDELT + RSS + GNews + official), explore it, and analyze key signals.")

with st.sidebar:
    st.header("Settings")

    db_path = st.text_input("SQLite DB path", DEFAULT_DB, help="Created automatically if it doesn't exist.")
    con = db_connect(db_path)
    init_db(con)

    st.subheader("Keywords")
    keywords_text = st.text_area("One per line", "\n".join(DEFAULT_KEYWORDS), height=150)
    keywords = [k.strip() for k in keywords_text.splitlines() if k.strip()]

    st.subheader("Collect windows")
    lookback_days = st.slider("Lookback days (news)", 7, 365, 180)

    st.subheader("GDELT settings")
    gdelt_days = st.slider("GDELT date window (auto-retries anyway)", 7, 90, 30)
    gdelt_max = st.slider("GDELT max records", 10, 300, 120, step=10)
    gdelt_full_text = st.checkbox("GDELT: fetch full article text (slower)", value=False)
    gdelt_full_text_n = st.slider("GDELT full-text limit", 0, 60, 15)

    st.subheader("RSS feeds (one per line)")
    rss_text = st.text_area("RSS URLs", "\n".join(DEFAULT_RSS_FEEDS), height=140)
    rss_feeds = [u.strip() for u in rss_text.splitlines() if u.strip()]
    rss_full_text = st.checkbox("RSS: fetch full article text (slower)", value=False)
    rss_max_per_feed = st.slider("RSS max items per feed", 5, 60, 25)

    st.subheader("GNews (optional extra)")
    use_gnews = st.checkbox("Enable GNews source", value=True)
    gnews_country = st.text_input("GNews country (2 letters)", "EU")
    gnews_language = st.text_input("GNews language (2 letters)", "en")
    gnews_max = st.slider("GNews max results per topic", 10, 100, 50, step=10)

    st.subheader("Official URLs (one per line)")
    official_text = st.text_area("Official URLs", "\n".join(DEFAULT_OFFICIAL_URLS), height=110)
    official_urls = [u.strip() for u in official_text.splitlines() if u.strip()]


tab_collect, tab_explore, tab_analyze = st.tabs(["Collect data", "Explore data", "Analyze (Dashboard)"])


with tab_collect:
    st.subheader("Collect data")

    c1, c2, c3, c4 = st.columns(4)
    do_gdelt = c1.checkbox("Collect GDELT", value=True)
    do_rss = c2.checkbox("Collect RSS", value=True)
    do_gnews = c3.checkbox("Collect GNews", value=True)
    do_official = c4.checkbox("Collect Official", value=True)

    if GNews is None and do_gnews:
        st.warning("GNews library not available. Install it with: pip install gnews")

    if st.button("🚀 Run collection", type="primary"):
        total_inserted = 0

        if do_gdelt:
            with st.spinner("Collecting from GDELT…"):
                ins, seen, extracted_ok, debug = collect_gdelt(
                    con,
                    keywords=keywords,
                    days=gdelt_days,
                    max_records=gdelt_max,
                    fetch_full_text=gdelt_full_text,
                    max_full_text=gdelt_full_text_n,
                )
                total_inserted += ins
                st.success(f"✅ GDELT seen {seen} | extracted {extracted_ok} | inserted {ins}")
                with st.expander("GDELT debug (open if seen=0)"):
                    st.json(debug)

        if do_rss:
            with st.spinner("Collecting from RSS…"):
                ins, entries_seen, extracted_ok = collect_rss(
                    con,
                    feed_urls=rss_feeds,
                    fetch_full_text=rss_full_text,
                    max_per_feed=rss_max_per_feed,
                )
                total_inserted += ins
                st.success(f"✅ RSS entries seen {entries_seen} | extracted {extracted_ok} | inserted {ins}")

        if do_gnews and use_gnews and GNews is not None:
            with st.spinner("Collecting from GNews…"):
                ins, seen = collect_gnews(
                    con,
                    keywords=keywords,
                    lookback_days=lookback_days,
                    country=gnews_country.strip() or "EU",
                    language=gnews_language.strip() or "en",
                    max_results=gnews_max,
                )
                total_inserted += ins
                st.success(f"✅ GNews seen {seen} | inserted {ins}")

        if do_official:
            with st.spinner("Collecting from official pages…"):
                ins, extracted_ok = collect_official(con, official_urls)
                total_inserted += ins
                st.success(f"✅ Official extracted {extracted_ok} | inserted {ins}")

        if total_inserted == 0:
            st.warning(
                "No new documents inserted. Try:\n"
                "- Increase lookback days\n"
                "- Use broader keywords like: 'packaging waste', 'sustainable packaging'\n"
                "- Turn ON GNews + RSS (they usually always return items)"
            )
        else:
            st.success(f"Done — inserted {total_inserted} documents.")

    st.info("Fast mode tip: keep full-text OFF. Titles/summaries are enough for trend dashboards.")


with tab_explore:
    st.subheader("Explore data")
    days_view = st.slider("Show data from last N days", 7, 365, 180)

    df = load_docs(con, days=days_view)
    if df.empty:
        st.info("No data yet. Go to **Collect data** first.")
    else:
        sources = ["All"] + sorted(df["source"].dropna().unique().tolist())
        pick_source = st.selectbox("Filter by source", sources, index=0)
        q = st.text_input("Search in title/text", value="")

        view = df.copy()
        if pick_source != "All":
            view = view[view["source"] == pick_source]
        if q.strip():
            qq = q.strip().lower()
            view = view[
                view["title"].fillna("").str.lower().str.contains(qq)
                | view["text"].fillna("").str.lower().str.contains(qq)
            ]

        show = view.copy()
        show["published_at"] = show["published_at"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
        show["collected_at"] = show["collected_at"].dt.strftime("%Y-%m-%d %H:%M").fillna("")

        st.dataframe(
            show[["source", "published_at", "domain", "title", "url"]],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download CSV",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="packaging_trends_export.csv",
            mime="text/csv",
        )


with tab_analyze:
    st.subheader("Analyze (Dashboard)")
    days_dash = st.slider("Analyze last N days", 7, 365, 180)

    df = load_docs(con, days=days_dash)
    if df.empty:
        st.info("No data yet. Collect first.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Documents", f"{len(df):,}")
        m2.metric("Sources", f"{df['source'].nunique()}")
        m3.metric("Domains", f"{df['domain'].nunique()}")
        m4.metric("With text", f"{int((df['text'].fillna('').str.len() > 0).sum())}")

        st.write("### Documents over time")
        tmp = df.copy()
        tmp["date"] = (tmp["published_at"].fillna(tmp["collected_at"])).dt.date
        daily = tmp.groupby("date").size().rename("count").to_frame()
        st.line_chart(daily)

        st.write("### By source")
        by_source = df["source"].value_counts().rename_axis("source").reset_index(name="count")
        st.bar_chart(by_source.set_index("source")["count"])

        st.write("### Top terms (what people talk about most)")
        terms = top_terms(df, n=25)
        st.dataframe(terms, use_container_width=True)

        st.write("### Signals (concerns / expectations buckets)")
        sig = simple_signals(df)
        if sig.empty:
            st.info("No signals detected yet (try collecting more items).")
        else:
            st.dataframe(sig, use_container_width=True)
            st.bar_chart(sig.set_index("signal")["count"])

        st.write("### Recent items")
        recent = df.head(20).copy()
        recent["published_at"] = recent["published_at"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
        st.dataframe(recent[["source", "published_at", "title", "url"]], use_container_width=True, hide_index=True)
