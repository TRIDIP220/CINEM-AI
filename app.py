import os
import re
from html import unescape
from html import escape
from typing import Any

import pandas as pd
import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
TAG_RE = re.compile(r"<[^>]+>")

st.set_page_config(
    page_title="CinemAI - Movie Sentiment and Recommendations",
    page_icon="CM",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #0b0f14;
        --panel: rgba(18, 24, 32, 0.82);
        --panel-soft: rgba(255, 255, 255, 0.045);
        --line: rgba(255, 255, 255, 0.10);
        --text: #f8fafc;
        --muted: #a5b4c5;
        --accent: #e11d48;
        --accent-2: #14b8a6;
        --warning: #f59e0b;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 10%, rgba(225, 29, 72, 0.13), transparent 26rem),
            radial-gradient(circle at 88% 0%, rgba(20, 184, 166, 0.11), transparent 24rem),
            var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1220px;
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4 {
        letter-spacing: 0;
    }

    .brand {
        border-bottom: 1px solid var(--line);
        margin-bottom: 1.35rem;
        padding-bottom: 1.1rem;
    }

    .brand-kicker {
        display: inline-flex;
        align-items: center;
        border: 1px solid rgba(225, 29, 72, 0.28);
        color: #fb7185;
        background: rgba(225, 29, 72, 0.09);
        border-radius: 6px;
        padding: 0.24rem 0.55rem;
        font-size: 0.74rem;
        font-weight: 750;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }

    .brand-title {
        color: var(--text);
        font-size: 2.55rem;
        line-height: 1.08;
        font-weight: 850;
        margin: 0;
    }

    .brand-title span {
        color: #fb7185;
    }

    .brand-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        margin-top: 0.45rem;
    }

    section[data-testid="stSidebar"] {
        background: #0a0d12;
        border-right: 1px solid var(--line);
    }

    .status-card, .movie-card, .metric-card, .sentiment-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.18);
    }

    .status-card {
        padding: 0.9rem;
        margin-bottom: 0.9rem;
    }

    .status-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--text);
        font-weight: 760;
        margin-bottom: 0.7rem;
    }

    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }

    .dot.online { background: #22c55e; box-shadow: 0 0 10px rgba(34, 197, 94, 0.7); }
    .dot.offline { background: #ef4444; box-shadow: 0 0 10px rgba(239, 68, 68, 0.7); }

    .status-row {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.35rem 0;
        border-top: 1px solid rgba(255,255,255,0.06);
        font-size: 0.82rem;
    }

    .status-key { color: var(--muted); }
    .status-value { color: var(--text); font-weight: 720; text-align: right; }

    .code-box {
        font-family: Consolas, monospace;
        font-size: 0.76rem;
        color: #e2e8f0;
        background: rgba(0, 0, 0, 0.28);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 6px;
        padding: 0.55rem;
        margin-top: 0.55rem;
        overflow-wrap: anywhere;
    }

    .movie-card {
        padding: 1.1rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--accent);
    }

    .movie-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 0.55rem;
    }

    .movie-title {
        color: var(--text);
        font-size: 1.22rem;
        line-height: 1.2;
        font-weight: 820;
        margin: 0;
    }

    .match-badge {
        flex: 0 0 auto;
        color: white;
        background: linear-gradient(135deg, #ef4444, #dc2626);
        border-radius: 999px;
        padding: 0.28rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 820;
    }

    .match-badge.high {
        background: linear-gradient(135deg, #10b981, #059669);
    }

    .match-badge.medium {
        background: linear-gradient(135deg, #f59e0b, #d97706);
    }

    .meta {
        color: var(--muted);
        font-size: 0.84rem;
        line-height: 1.45;
        margin-bottom: 0.55rem;
    }

    .tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin: 0.65rem 0;
    }

    .tag {
        color: #dbeafe;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.18);
        border-radius: 5px;
        padding: 0.16rem 0.48rem;
        font-size: 0.74rem;
        font-weight: 650;
    }

    .overview {
        color: #d6dee8;
        font-size: 0.9rem;
        line-height: 1.52;
        margin: 0.7rem 0;
    }

    .components {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.5rem;
        margin-top: 0.85rem;
    }

    .component {
        background: var(--panel-soft);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 7px;
        padding: 0.45rem 0.5rem;
        text-align: center;
        min-width: 0;
    }

    .component-label {
        color: var(--muted);
        font-size: 0.68rem;
        text-transform: uppercase;
        font-weight: 720;
    }

    .component-value {
        color: #fb7185;
        font-size: 1rem;
        font-weight: 850;
    }

    .sentiment-pill {
        display: inline-flex;
        color: #d1fae5;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.24);
        border-radius: 999px;
        padding: 0.34rem 0.75rem;
        margin-bottom: 1rem;
        font-size: 0.8rem;
        font-weight: 760;
        text-transform: uppercase;
    }

    .sentiment-pill.negative {
        color: #fee2e2;
        background: rgba(239, 68, 68, 0.12);
        border-color: rgba(239, 68, 68, 0.24);
    }

    .sentiment-card {
        padding: 1.2rem;
        margin-top: 1rem;
    }

    .sentiment-result {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
    }

    .metric-card {
        padding: 1rem;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 720;
        text-transform: uppercase;
    }

    .metric-value {
        color: var(--text);
        font-size: 1.65rem;
        font-weight: 850;
        margin-top: 0.2rem;
    }

    @media (max-width: 760px) {
        .brand-title { font-size: 2rem; }
        .movie-head { display: block; }
        .match-badge { display: inline-flex; margin-top: 0.55rem; }
        .components, .sentiment-result { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str) -> requests.Response:
    return requests.get(f"{BACKEND_URL}{path}", timeout=15)


def api_post(path: str, payload: dict[str, Any]) -> requests.Response:
    return requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=30)


@st.cache_data(ttl=10)
def backend_status() -> dict[str, Any] | None:
    try:
        response = api_get("/api/status")
        return response.json() if response.ok else None
    except requests.RequestException:
        return None


@st.cache_data(show_spinner=False)
def movie_titles() -> list[str]:
    try:
        response = api_get("/api/movies")
        return response.json().get("movies", []) if response.ok else []
    except requests.RequestException:
        return []


def pct(value: Any) -> int:
    try:
        return max(0, min(100, int(float(value) * 100)))
    except (TypeError, ValueError):
        return 0


def clean_text(value: Any) -> str:
    text = unescape(str(value or ""))
    text = TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_text(value: Any, fallback: str = "") -> str:
    return escape(clean_text(value) or fallback)


def render_html(markup: str) -> None:
    if hasattr(st, "html"):
        st.html(markup)
    else:
        st.markdown(markup, unsafe_allow_html=True)


def movie_card(movie: dict[str, Any]) -> str:
    match_pct = pct(movie.get("match_score", 0))
    if match_pct >= 90:
        match_class = "high"
    elif match_pct >= 70:
        match_class = "medium"
    else:
        match_class = "low"
    genres = movie.get("genres", [])[:5]
    tags = "".join(f'<span class="tag">{safe_text(genre)}</span>' for genre in genres) or '<span class="tag">No genre</span>'
    components = movie.get("score_components", {})
    component_html = "".join(
        f'<div class="component"><div class="component-label">{label}</div><div class="component-value">{pct(components.get(key, 0))}%</div></div>'
        for label, key in [
            ("Plot", "plot"),
            ("Metadata", "metadata"),
            ("Genre", "genre"),
            ("Quality", "quality"),
        ]
    )
    cast = ", ".join(safe_text(name) for name in movie.get("cast", [])[:3]) or "Unknown cast"
    director = safe_text(movie.get("director"), "Unknown director")
    overview = safe_text(movie.get("overview"), "Overview unavailable.")
    release_date = safe_text(movie.get("release_date"), "Unknown release")
    title = safe_text(movie.get("title"), "Untitled")
    rating = float(movie.get("vote_average") or 0)

    return f"""<div class="movie-card">
<div class="movie-head">
<h3 class="movie-title">{title}</h3>
<span class="match-badge {match_class}">{match_pct}% Match</span>
</div>
<div class="meta">{release_date} | Rating <b>{rating:.1f}/10</b> | Director: {director}</div>
<div class="tags">{tags}</div>
<p class="overview">{overview}</p>
<div class="meta">Cast: {cast}</div>
<div class="components">{component_html}</div>
</div>"""


render_html(
    """
    <div class="brand">
        <div class="brand-kicker">Cinematic AI Console</div>
        <h1 class="brand-title">Cinem<span>AI</span></h1>
        <div class="brand-subtitle">Explainable movie recommendations and PyTorch sentiment analysis.</div>
    </div>
    """,
)

status = backend_status()

with st.sidebar:
    st.header("System")
    if status:
        render_html(
            """<div class="status-card" style="margin-bottom: 0px !important;">
<div class="status-title" style="margin-bottom: 0px !important;"><span class="dot online"></span> FastAPI Online</div>
</div>"""
        )
        if not status.get("sentiment_models_loaded"):
            st.warning("PyTorch checkpoints are missing. The backend will use the lexicon fallback.")
            render_html('<div class="code-box">python -m backend.train --epochs 3</div>')
    else:
        render_html(
            """<div class="status-card">
<div class="status-title"><span class="dot offline"></span> FastAPI Offline</div>
<div class="status-row"><span class="status-key">Action</span><span class="status-value">Start backend</span></div>
</div>
<div class="code-box">python -m uvicorn backend.main:app --reload</div>"""
        )

    st.caption("Sentiment model: DistilBERT Transformer")
    model_type = "lstm"

titles = movie_titles() if status else []
rec_tab, sentiment_tab, chat_tab, analytics_tab = st.tabs(["Recommendations", "Sentiment Lab", "Chat Agent", "Analytics"])

with rec_tab:
    st.subheader("Personalized Movie Recommendations")
    if not status:
        st.error("Start the backend to generate recommendations.")
    elif not titles:
        st.warning("No movie titles were returned by the backend.")
    else:
        left, right = st.columns([2, 1])
        with left:
            default_index = titles.index("Avatar") if "Avatar" in titles else 0
            selected_title = st.selectbox("Movie you like", titles, index=default_index)
        with right:
            top_n = st.slider("Results count", 3, 12, 6)

        review = st.text_area(
            "Optional review",
            placeholder="Write what you liked or disliked. Sentiment can adjust the recommendation ranking.",
            height=110,
        )

        if st.button("Generate recommendations", type="primary"):
            payload = {"title": selected_title, "top_n": top_n, "review": review or None, "model_type": model_type}
            with st.spinner("Finding similar films and building explanations..."):
                response = api_post("/api/recommend", payload)

            if not response.ok:
                try:
                    st.error(response.json().get("detail", "Recommendation request failed."))
                except ValueError:
                    st.error("Recommendation request failed.")
            else:
                data = response.json()
                sentiment_value = data.get("review_sentiment")
                if sentiment_value:
                    sentiment_class = "negative" if sentiment_value == "negative" else ""
                    render_html(f'<div class="sentiment-pill {sentiment_class}">Review sentiment: {safe_text(sentiment_value)}</div>')

                for movie in data.get("recommendations", []):
                    render_html(movie_card(movie))
                    with st.expander("Why this recommendation?"):
                        explain_response = api_post(
                            "/api/explain",
                            {
                                "selected_title": selected_title,
                                "recommended_title": movie.get("title", ""),
                                "user_review": review or None,
                                "model_type": model_type,
                            },
                        )
                        if explain_response.ok:
                            st.write(explain_response.json().get("explanation", "Explanation unavailable."))
                        else:
                            st.write("Explanation unavailable.")

with sentiment_tab:
    st.subheader("Neural Sentiment Classifier")
    review_text = st.text_area(
        "Movie review text",
        placeholder="The performances were excellent and the story was moving, even though the pacing slowed down in the middle.",
        height=150,
    )

    if st.button("Analyze sentiment", type="primary"):
        if not review_text.strip():
            st.warning("Enter a review first.")
        else:
            with st.spinner("Running sentiment analysis..."):
                response = api_post("/api/sentiment", {"text": review_text})

            if response.ok:
                result = response.json()
                prediction = result["prediction"]
                label = str(prediction["label"]).title()
                confidence = float(prediction["confidence"]) * 100
                probability = float(prediction["probability"]) * 100
                source = safe_text(prediction["model_source"])

                render_html(
                    f"""<div class="sentiment-card">
<div class="sentiment-result">
<div class="metric-card"><div class="metric-label">Sentiment</div><div class="metric-value">{safe_text(label)}</div></div>
<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value">{confidence:.1f}%</div></div>
<div class="metric-card"><div class="metric-label">Positive probability</div><div class="metric-value">{probability:.1f}%</div></div>
</div>
<div class="meta" style="margin-top:0.8rem;">Model source: {source}</div>
</div>"""
                )

                # Fetch and display Aspect-Based Sentiment Analysis (ABSA)
                aspects_response = api_post("/api/sentiment/aspects", {"text": review_text})
                if aspects_response.ok:
                    aspects_data = aspects_response.json().get("aspects", {})
                    st.write("---")
                    st.markdown("<h4 style='font-size:1.1rem; color:#ffffff; margin-bottom:0.75rem;'>Aspect-Based Sentiment Breakdown</h4>", unsafe_allow_html=True)
                    
                    col_asp1, col_asp2, col_asp3 = st.columns(3)
                    for col, (asp_name, asp_info) in zip([col_asp1, col_asp2, col_asp3], aspects_data.items()):
                        with col:
                            st.markdown(f"**{asp_name.title()}**")
                            if asp_info["mentioned"]:
                                label_style = "color:#10b981;" if asp_info["label"] == "positive" else "color:#ef4444;"
                                st.markdown(f"<span style='{label_style} font-weight:bold;'>{asp_info['label'].upper()}</span> ({int(asp_info['score']*100)}% positive)", unsafe_allow_html=True)
                                st.progress(asp_info["score"])
                            else:
                                st.caption("Not mentioned in review")

                st.write("---")
                attrs = pd.DataFrame(result.get("attributions", []))
                if not attrs.empty:
                    st.write("Top word influences")
                    st.bar_chart(attrs.set_index("word")["importance"])
            else:
                st.error("Sentiment analysis failed.")

with chat_tab:
    st.subheader("CinemAI Conversation Assistant")
    
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        chat_query = st.text_input("What kind of movie are you looking for?", placeholder="e.g., A space adventure with time travel", label_visibility="collapsed")
    with col_btn:
        chat_submitted = st.button("Search", type="primary", use_container_width=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your CinemAI recommendation assistant. Ask me for recommendations or search semantically!"}
        ]
        
    # Chat input search logic
    if chat_submitted and chat_query.strip():
        # Reset chat history to show only the current query
        st.session_state.messages = [{"role": "user", "content": chat_query}]
        
        with st.spinner("Searching..."):
            chat_response = api_post("/api/chat", {"message": chat_query})
            
        if chat_response.ok:
            data = chat_response.json()
            reply = data.get("reply", "Here are some recommendations.")
            movies_list = data.get("movies", [])
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "movies": movies_list
            })
            st.rerun()
        else:
            st.error("Error communicating with chat assistant.")
            
    st.write("---")
        
    # Render chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "movies" in msg:
                for movie in msg["movies"]:
                    render_html(movie_card(movie))

with analytics_tab:
    st.subheader("Dataset Insights")
    if not status:
        st.error("Start the backend to view analytics.")
    else:
        response = api_get("/api/stats")
        if not response.ok:
            st.error("Could not load analytics from the backend.")
        else:
            stats = response.json()
            top_genre = next(iter(stats.get("genre_distribution", {})), "N/A")
            render_html(
                f"""<div class="sentiment-result">
<div class="metric-card"><div class="metric-label">TMDB Movies</div><div class="metric-value">{int(stats.get("movie_count", 0)):,}</div></div>
<div class="metric-card"><div class="metric-label">Top Genre</div><div class="metric-value">{safe_text(top_genre)}</div></div>
<div class="metric-card"><div class="metric-label">Engine</div><div class="metric-value">Hybrid v2</div></div>
</div>"""
            )

            left, right = st.columns(2)
            with left:
                st.write("Genre distribution")
                genre_df = pd.DataFrame(stats["genre_distribution"].items(), columns=["genre", "count"]).set_index("genre")
                st.bar_chart(genre_df)
            with right:
                st.write("Rating distribution")
                rating_df = pd.DataFrame(stats["rating_distribution"].items(), columns=["rating", "count"]).set_index("rating")
                st.bar_chart(rating_df)

            st.write("Popular movies: rating vs popularity")
            st.scatter_chart(pd.DataFrame(stats["scatter_metrics"]), x="vote_average", y="popularity")
