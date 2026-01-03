# streamlit_app.py
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import feedparser
import pandas as pd
import requests
import streamlit as st
import trafilatura
from dateutil import parser as dateparser
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# DEFAULT CONFIG
# =========================

DEFAULT_KEYWORDS = [
    "PPWR", "Packaging and Packaging Waste Regulation", "Regulation (EU) 2025/40",
    "extended producer responsibility", "EPR", "deposit return scheme", "DRS",
    "recycled content", "recyclability", "reuse targets", "packaging tax", "single-use plastics",
    "paper packaging", "plastic packaging", "bioplastic", "compostable", "refill", "reusable packaging",
    "mono-material", "PFAS", "barrier coatings",
]

DEFAULT_RSS_FEEDS = [
    "https://www.eea.europa.eu/en/newsroom/press-releases/RSS",
    "https://www.reddit.com/search.rss?q=packaging+waste+EU&sort=new",
]

DEFAULT_OFFICIAL_URLS = [
    "https://environment.ec.europa.eu/topics/waste-and-recycling/packaging-waste_en",
    "https://eur-lex.europa.eu/eli/reg/2025/40/oj/eng",
]

DB_PATH_DEFAULT = "packaging_trends.db"


LAW_PATTERNS = [
    r"\bPPWR\b",
    r"\bRegulation\s*\(EU\)\s*2025/40\b",
    r"\bDirective\s*94/62/EC\b",
    r"\bEPR\b|\bextended producer responsibility\b",
    r"\bdeposit return\b|\bDRS\b",
    r"\breuse target(s)?\b",
    r"\brecycled content\b",
    r"\brecyclab(le|ility)\b",
    r"\bsingle-?use plastics?\b",
    r"\bpackaging tax\b",
]

CONCERN_WORDS = [
    "cost", "expensive", "risk", "penalty", "fine", "uncertain", "unclear", "burden",
    "compliance", "reporting", "lawsuit", "ban", "restricted", "shortage",
]
EXPECTATION_WORDS = [
    "expect", "demand", "want", "prefer", "require", "should", "will", "must",
    "target", "timeline", "deadline", "labeling", "transparency",
]
OPPORTUNITY_WORDS = [
    "opportunity", "innovation", "growth", "reusable", "refill", "circular",
    "design for recycling", "mono-material", "lightweighting", "new market",
]


# =========================
# DATA MODEL + DB
# =========================

@dataclass
class Document:
    source: str
    url: str
    title: str
    published_at: str  # ISO
    text: str
    language: str
    metadata_json: str


def init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            title TEXT,
            published_at TEXT,
            text TEXT,
            language TEXT,
            metadata_json TEXT
        )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_published_at ON documents(published_at)")
        con.commit()


def upsert_document(doc: Document, db_path: str) -> bool:
    with sqlite3.connect(db_path) as con:
        try:
            con.execute("""
                INSERT INTO documents (source, url, title, published_at, text, language, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (doc.source, doc.url, doc.title, doc.published_at, doc.text, doc.language, doc.metadata_json))
            con.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def load_documents(db_path: str, days: int) -> pd.DataFrame:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(
            "SELECT * FROM documents WHERE published_at >= ? ORDER BY published_at DESC",
            con,
            params=(since,)
        )
    return df


# =========================
# SCRAPING / EXTRACTION
# =========================

def extract_main_text(url: str, timeout: int = 20) -> str:
    try:
        downloaded = trafilatura.fetch_url(url, timeout=timeout)
        if not downloaded:
            return ""
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        return text or ""
    except Exception:
        return ""


def detect_language(text: str) -> str:
    txt = (text or "").strip()
    if len(txt) < 80:
        return "unknown"
    try:
        return detect(txt)
    except Exception:
        return "unknown"


# =========================
# COLLECTORS
# =========================

def fetch_gdelt_articles(keywords: List[str], days: int, max_records: int) -> List[Dict[str, Any]]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    query = " OR ".join([f'"{k}"' if " " in k else k for k in keywords])
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "sort": "HybridRel",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("articles", [])


def collect_gdelt(db_path: str, keywords: List[str], days: int, max_records: int, status_cb=None) -> Tuple[int, int]:
    articles = fetch_gdelt_articles(keywords=keywords, days=days, max_records=max_records)
    inserted = 0

    for i, a in enumerate(articles, start=1):
        if status_cb:
            status_cb(f"GDELT: processing {i}/{len(articles)}")

        url = a.get("url", "") or ""
        if not url:
            continue

        title = a.get("title", "") or ""
        seendate = a.get("seendate", "")
        try:
            published_at = dateparser.parse(seendate).astimezone(timezone.utc).isoformat() if seendate else datetime.now(timezone.utc).isoformat()
        except Exception:
            published_at = datetime.now(timezone.utc).isoformat()

        text = extract_main_text(url)
        if not text or len(text) < 400:
            continue

        lang = detect_language(text)
        doc = Document(
            source="gdelt",
            url=url,
            title=title,
            published_at=published_at,
            text=text,
            language=lang,
            metadata_json=json.dumps(a, ensure_ascii=False),
        )
        if upsert_document(doc, db_path=db_path):
            inserted += 1

    return inserted, len(articles)


def collect_rss(db_path: str, feed_urls: List[str], days: int, status_cb=None) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    inserted = 0

    for feed_idx, feed_url in enumerate(feed_urls, start=1):
        if status_cb:
            status_cb(f"RSS: loading feed {feed_idx}/{len(feed_urls)}")

        d = feedparser.parse(feed_url)
        entries = getattr(d, "entries", []) or []

        for i, e in enumerate(entries, start=1):
            link = getattr(e, "link", "") or ""
            title = getattr(e, "title", "") or ""
            published = getattr(e, "published", "") or getattr(e, "updated", "") or ""

            try:
                published_dt = dateparser.parse(published).astimezone(timezone.utc) if published else datetime.now(timezone.utc)
            except Exception:
                published_dt = datetime.now(timezone.utc)

            if published_dt < cutoff:
                continue
            if not link:
                continue

            if status_cb:
                status_cb(f"RSS: processing entry {i}/{len(entries)} in feed {feed_idx}")

            text = extract_main_text(link)
            if not text or len(text) < 300:
                continue

            lang = detect_language(text)
            meta = {}
            if hasattr(e, "author"):
                meta["author"] = getattr(e, "author")
            if hasattr(e, "tags"):
                meta["tags"] = [getattr(t, "term", "") for t in getattr(e, "tags", [])]

            doc = Document(
                source="rss",
                url=link,
                title=title,
                published_at=published_dt.isoformat(),
                text=text,
                language=lang,
                metadata_json=json.dumps(meta, ensure_ascii=False),
            )
            if upsert_document(doc, db_path=db_path):
                inserted += 1

    return inserted


def collect_official(db_path: str, urls: List[str], status_cb=None) -> int:
    inserted = 0
    now = datetime.now(timezone.utc).isoformat()

    for i, url in enumerate(urls, start=1):
        if status_cb:
            status_cb(f"Official: processing {i}/{len(urls)}")

        text = extract_main_text(url)
        if not text or len(text) < 300:
            continue

        lang = detect_language(text)
        doc = Document(
            source="official",
            url=url,
            title=url,
            published_at=now,  # snapshot time
            text=text,
            language=lang,
            metadata_json=json.dumps({}, ensure_ascii=False),
        )
        if upsert_document(doc, db_path=db_path):
            inserted += 1

    return inserted


# =========================
# ANALYSIS
# =========================

def basic_preprocess(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def score_lexicon(text: str, words: List[str]) -> int:
    t = (text or "").lower()
    return sum(1 for w in words if w.lower() in t)


def find_law_mentions(text: str) -> List[str]:
    hits = []
    for pat in LAW_PATTERNS:
        if re.search(pat, text or "", flags=re.IGNORECASE):
            hits.append(pat)
    return hits


def nmf_topics(texts: List[str], n_topics: int = 8, top_words: int = 10) -> pd.DataFrame:
    vec = TfidfVectorizer(
        max_features=4000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )
    X = vec.fit_transform(texts)
    model = NMF(n_components=n_topics, random_state=42)
    model.fit(X)

    H = model.components_
    features = vec.get_feature_names_out()

    rows = []
    for k, weights in enumerate(H):
        top_idx = weights.argsort()[::-1][:top_words]
        terms = [features[i] for i in top_idx]
        rows.append({"topic": k, "top_terms": ", ".join(terms)})
    return pd.DataFrame(rows)


def analyze_df(df: pd.DataFrame, only_english: bool) -> Dict[str, Any]:
    if df.empty:
        return {"df": df}

    df = df.copy()
    df["clean_text"] = df["text"].fillna("").map(basic_preprocess)

    if only_english:
        df = df[(df["language"] == "en") | (df["language"] == "unknown")].copy()

    vader = SentimentIntensityAnalyzer()
    df["sentiment"] = df["clean_text"].map(lambda t: vader.polarity_scores(t)["compound"])
    df["concern_score"] = df["clean_text"].map(lambda t: score_lexicon(t, CONCERN_WORDS))
    df["expectation_score"] = df["clean_text"].map(lambda t: score_lexicon(t, EXPECTATION_WORDS))
    df["opportunity_score"] = df["clean_text"].map(lambda t: score_lexicon(t, OPPORTUNITY_WORDS))
    df["law_hits"] = df["clean_text"].map(find_law_mentions)

    law_counts = (
        df.explode("law_hits")
          .dropna(subset=["law_hits"])
          .groupby("law_hits")
          .size()
          .reset_index(name="mentions")
          .sort_values("mentions", ascending=False)
    )

    topics_df = nmf_topics(df["clean_text"].tolist(), n_topics=8) if len(df) >= 3 else pd.DataFrame()

    watch = df[df["concern_score"] >= 2].copy()
    watch = watch.sort_values(["concern_score", "published_at"], ascending=[False, False]).head(20)

    return {
        "df": df,
        "law_counts": law_counts,
        "topics": topics_df,
        "watch": watch,
    }


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Packaging Trends Monitor", layout="wide")

st.title("📦 Packaging Trends Monitor (EU-focused)")
st.caption("Collect public info (news + official pages + RSS) and analyze topics, concerns, expectations, and law mentions.")

with st.sidebar:
    st.header("Settings")
    db_path = st.text_input("SQLite DB path", DB_PATH_DEFAULT)

    days_collect = st.slider("Collect: lookback days", 1, 180, 14)
    max_records = st.slider("Collect: max news records (GDELT)", 10, 250, 120, step=10)

    st.subheader("Keywords (one per line)")
    keywords_text = st.text_area(" ", "\n".join(DEFAULT_KEYWORDS), height=180)
    keywords = [k.strip() for k in keywords_text.splitlines() if k.strip()]

    st.subheader("RSS feeds (one per line)")
    rss_text = st.text_area("  ", "\n".join(DEFAULT_RSS_FEEDS), height=110)
    rss_feeds = [u.strip() for u in rss_text.splitlines() if u.strip()]

    st.subheader("Official URLs (one per line)")
    official_text = st.text_area("   ", "\n".join(DEFAULT_OFFICIAL_URLS), height=90)
    official_urls = [u.strip() for u in official_text.splitlines() if u.strip()]

    st.divider()
    days_analyze = st.slider("Analyze: time window days", 1, 365, 90)
    only_english = st.checkbox("Analyze English only (recommended for VADER)", value=True)

# init db
init_db(db_path)

tab1, tab2, tab3 = st.tabs(["1) Collect", "2) Analyze", "3) Dashboard"])

with tab1:
    st.subheader("Collect data into the database")

    colA, colB, colC = st.columns(3)
    run_gdelt = colA.checkbox("Collect from GDELT (news)", value=True)
    run_rss = colB.checkbox("Collect from RSS feeds", value=True)
    run_official = colC.checkbox("Collect from official URLs", value=True)

    if st.button("🚀 Run collection", type="primary"):
        status = st.status("Starting...", expanded=True)

        def cb(msg: str):
            status.update(label=msg)

        try:
            inserted_total = 0

            if run_gdelt:
                ins, fetched = collect_gdelt(db_path, keywords, days_collect, max_records, status_cb=cb)
                status.write(f"✅ GDELT inserted {ins} (fetched {fetched})")
                inserted_total += ins

            if run_rss:
                ins = collect_rss(db_path, rss_feeds, days_collect, status_cb=cb)
                status.write(f"✅ RSS inserted {ins}")
                inserted_total += ins

            if run_official:
                ins = collect_official(db_path, official_urls, status_cb=cb)
                status.write(f"✅ Official inserted {ins}")
                inserted_total += ins

            status.update(label=f"Done. Total inserted: {inserted_total}", state="complete")
        except Exception as e:
            status.update(label=f"Error: {e}", state="error")

    st.info("Tip: If some pages don’t extract text, that’s normal (paywalls, scripts, blocked pages). Try adding more RSS sources.")

with tab2:
    st.subheader("Analyze collected data")

    df = load_documents(db_path, days_analyze)
    st.write(f"Loaded **{len(df)}** documents from the last **{days_analyze}** days.")

    if st.button("📊 Run analysis"):
        results = analyze_df(df, only_english=only_english)

        rdf = results.get("df", pd.DataFrame())
        if rdf.empty:
            st.warning("No documents to analyze.")
        else:
            st.success("Analysis complete.")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg sentiment", f"{rdf['sentiment'].mean():.2f}")
            c2.metric("Avg concern score", f"{rdf['concern_score'].mean():.2f}")
            c3.metric("Avg expectation score", f"{rdf['expectation_score'].mean():.2f}")
            c4.metric("Avg opportunity score", f"{rdf['opportunity_score'].mean():.2f}")

            st.divider()
            st.subheader("Top law/regulation mention patterns")
            law_counts = results.get("law_counts", pd.DataFrame())
            if not law_counts.empty:
                st.dataframe(law_counts.head(15), use_container_width=True)
                st.bar_chart(law_counts.head(15).set_index("law_hits")["mentions"])
            else:
                st.write("No law mention patterns found in the current window.")

            st.divider()
            st.subheader("Topics (TF-IDF + NMF)")
            topics = results.get("topics", pd.DataFrame())
            if not topics.empty:
                st.dataframe(topics, use_container_width=True)
            else:
                st.write("Not enough documents to generate topics (need at least a few).")

            st.divider()
            st.subheader("High-concern watchlist (top 20)")
            watch = results.get("watch", pd.DataFrame())
            if watch.empty:
                st.write("No high-concern items found.")
            else:
                # show clickable links
                show = watch[["published_at", "source", "title", "url", "concern_score", "sentiment"]].copy()
                st.dataframe(show, use_container_width=True)

            # downloads
            out_dir = "outputs"
            os.makedirs(out_dir, exist_ok=True)
            rdf.to_csv(f"{out_dir}/documents_scored.csv", index=False)
            law_counts.to_csv(f"{out_dir}/law_mentions.csv", index=False)
            topics.to_csv(f"{out_dir}/topics.csv", index=False)

            st.download_button(
                "⬇️ Download scored documents CSV",
                data=rdf.to_csv(index=False).encode("utf-8"),
                file_name="documents_scored.csv",
                mime="text/csv",
            )

with tab3:
    st.subheader("Quick dashboard")
    df = load_documents(db_path, days_analyze)

    if df.empty:
        st.write("No data yet. Go to **Collect** first.")
    else:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df = df.dropna(subset=["published_at"])

        st.write("Documents per source:")
        st.bar_chart(df["source"].value_counts())

        st.write("Documents over time (daily):")
        daily = df.set_index("published_at").resample("D").size().rename("count").to_frame()
        st.line_chart(daily)

        st.write("Recent documents:")
        st.dataframe(df[["published_at", "source", "title", "url"]].head(30), use_container_width=True)
