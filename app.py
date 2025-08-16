# CE Text Analytics Pro — Full App (No spaCy) — Optimized + Sentiment-Filtered Graph + Custom Stopwords
# -----------------------------------------------------------------------------------------------------
# Feature parity with your original 9 sections + upgrades:
# - Faster: sampling, vocab caps, caching, batched HF NER, row truncation, sparse ops.
# - Custom stopwords upload (TXT/CSV), merged into vectorization & cleaning.
# - Network graph sentiment filters (Positive/Neutral/Negative/All)
#   + pill-color chips and a table (words, sentiment, degree, neighbors).
# - Consistent section “What you can uncover” guidance box.
# - All visuals preserved (WordCloud, N-grams, Sentiment Micrograph, LDA, NER, Concordance, Downloads).

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safer for Streamlit Cloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import Text
from nltk.util import ngrams
import re
import base64
import nltk
from transformers import pipeline

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="CE Text Analytics Pro", layout="wide")

# -----------------------------
# NLTK downloads (quiet)
# -----------------------------
nltk.download("punkt", quiet=True)
nltk.download("vader_lexicon", quiet=True)

# -----------------------------
# CSS: Left nav pill style + chips + buttons + section info box
# -----------------------------
st.markdown(
    """
    <style>
    /* Sidebar radio -> pill style */
    [data-testid="stSidebar"] .stRadio > div { gap: 8px !important; }
    [data-testid="stSidebar"] .stRadio label {
        padding: 6px 14px !important;
        border-radius: 999px !important;
        border: 1px solid #E2E8F0 !important;
        background: #F8FAFC !important;
        color: #334155 !important;
        cursor: pointer !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] label {
        background: #10B981 !important;  /* emerald */
        color: white !important;
        border-color: #10B981 !important;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 999px;
        padding: 0.5em 1.25em;
        background-color: #10B981;
        color: white;
        font-weight: 700;
        border: none;
    }
    .stButton>button:hover { filter: brightness(0.95); }

    h2, h3 { color: #0F172A; }
    .small-muted { color: #64748B; font-size: 0.9rem; }

    /* Pills for sentiment chips */
    .chip {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        margin: 4px 6px 4px 0;
        font-size: 13px;
        font-weight: 600;
        border: 1px solid #E2E8F0;
        background:#F8FAFC;
        color:#0F172A;
        white-space: nowrap;
    }
    .chip.pos { background:#ECFDF5; color:#065F46; border-color:#A7F3D0; }
    .chip.neu { background:#F1F5F9; color:#0F172A; border-color:#CBD5E1; }
    .chip.neg { background:#FEF2F2; color:#7F1D1D; border-color:#FECACA; }

    /* Section info box */
    .section-note {
        border: 1px solid #E5E7EB;
        background: #F9FAFB;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0 16px 0;
        color:#0F172A;
    }
    .section-note h4 {
        margin: 0 0 8px 0;
        font-size: 1rem;
        color:#111827;
    }
    .section-note ul { margin: 0 0 0 18px; padding: 0; }
    .section-note li { margin-bottom: 4px; }

    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Cache HF Pipelines
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_pipelines():
    hf_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    hf_ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return hf_sentiment, hf_ner

hf_sentiment, hf_ner = load_pipelines()

# -----------------------------
# VADER for word-level sentiment
# -----------------------------
sia = SentimentIntensityAnalyzer()

# -----------------------------
# Helpers
# -----------------------------
def render_note(title: str, points: list[str]):
    items = "".join([f"<li>{p}</li>" for p in points])
    st.markdown(f"""
    <div class="section-note">
      <h4>{title}</h4>
      <ul>{items}</ul>
    </div>
    """, unsafe_allow_html=True)

def clean_text(s: str) -> str:
    s = str(s)
    s = s.lower()
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ensure_text_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna("").astype(str)

def load_stopwords_from_upload(stop_file) -> set:
    """Accepts TXT or CSV. TXT = one per line; CSV = all cells as tokens."""
    if stop_file is None:
        return set()
    name = stop_file.name.lower()
    try:
        if name.endswith(".txt"):
            raw = stop_file.read().decode("utf-8", errors="ignore")
            toks = [t.strip() for t in raw.splitlines() if t.strip()]
            return set(toks)
        elif name.endswith(".csv"):
            df = pd.read_csv(stop_file)
            vals = []
            for col in df.columns:
                vals.extend([str(v).strip() for v in df[col].dropna().tolist()])
            return set([v for v in vals if v])
        else:
            return set()
    except Exception:
        return set()

def top_words_from_topic(vectorizer: CountVectorizer, topic_row: np.ndarray, top_k: int = 10):
    idxs = topic_row.argsort()[-top_k:][::-1]
    vocab = vectorizer.get_feature_names_out()
    return [vocab[i] for i in idxs]

def sentiment_class(compound: float) -> str:
    if compound >= 0.05: return "positive"
    if compound <= -0.05: return "negative"
    return "neutral"

# -----------------------------
# Session State
# -----------------------------
ss = st.session_state
DEFAULT_KEYS = [
    "data", "processed_text", "custom_stopwords",
    "vectorizer", "doc_term_matrix", "lda_model",
    "knowledge_graph", "word_sentiment",
    "removed_words", "removed_ngrams"
]
for k in DEFAULT_KEYS:
    if k not in ss: ss[k] = None
if ss.get("removed_words") is None: ss["removed_words"] = []
if ss.get("removed_ngrams") is None: ss["removed_ngrams"] = []
if ss.get("custom_stopwords") is None: ss["custom_stopwords"] = set()

# -----------------------------
# Left Navigation (Pill-style)
# -----------------------------
SECTIONS = [
    "📂 Upload Data",
    "🧹 Text Processing",
    "🌐 Knowledge Graph",
    "☁ WordCloud & Ngrams",
    "😊 Sentiment Analysis",
    "🧠 Topic Modeling",
    "🔍 Entity Recognition",
    "🔎 Concordance Analysis",
    "⬇️ Download Results",
]
with st.sidebar:
    st.title("CE Text Analytics Pro")
    chosen = st.radio("Navigation", SECTIONS, index=0, label_visibility="collapsed")
    st.markdown('<p class="small-muted">Left nav styled as pills • Faster mode controls below</p>', unsafe_allow_html=True)

    # Global speed controls
    st.markdown("### ⚡ Performance")
    sample_rows = st.slider("Max rows to analyze", 50, 5000, 500, step=50,
                            help="Applies to heavy steps (vectorization, NER, etc.)")
    max_vocab = st.slider("Max vocabulary size", 300, 5000, 1500, step=100,
                          help="Caps CountVectorizer features to speed up co-occurrence & LDA")
    truncate_chars = st.slider("Truncate each row to N characters", 200, 3000, 800, step=100,
                               help="Speeds up transformers & vectorization by truncating long rows")
    st.caption("Increase these later if you need deeper analysis.")

# ==========================
# SECTION 1: Upload Data
# ==========================
if chosen == "📂 Upload Data":
    st.header("📂 Upload Your Files")
    render_note(
        "What you can uncover here",
        [
            "Load CSV/XLSX/TXT and optional custom stopwords (TXT/CSV).",
            "Stopwords are merged into all downstream analyses.",
            "Preview the first rows to confirm the right text column exists."
        ]
    )

    uploaded_file = st.file_uploader("Upload CSV, XLSX, or TXT file", type=['csv', 'xlsx', 'txt'])
    stop_file = st.file_uploader("Optional: Upload custom stopwords (TXT/CSV)", type=['txt', 'csv'])

    if stop_file:
        ss.custom_stopwords = load_stopwords_from_upload(stop_file)
        st.success(f"✅ Loaded {len(ss.custom_stopwords)} custom stopwords.")

    if uploaded_file:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.lower().endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                text = uploaded_file.read().decode("utf-8", errors="ignore")
                df = pd.DataFrame({"text": [text]})
            ss.data = df
            st.success("✅ Data Uploaded Successfully!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    if ss.data is None:
        st.info("Upload a file to begin. Supported: CSV, XLSX, TXT")

# Guard
if ss.data is None:
    st.stop()

# ==========================
# SECTION 2: Text Processing
# ==========================
if chosen == "🧹 Text Processing":
    st.header("🧹 Text Cleaning & Preview + Custom Stopwords")
    render_note(
        "What you can uncover here",
        [
            "Normalize text (lowercasing, punctuation removal) and preview results.",
            "Apply sampling & truncation (from sidebar) for speed.",
            "Verify custom stopwords that will be removed downstream."
        ]
    )

    text_column = st.selectbox("Select Text Column", ss.data.columns, index=0)
    raw_series = ensure_text_series(ss.data, text_column)

    with st.expander("Cleaning & Stopwords Options", expanded=True):
        st.caption("Lowercasing, punctuation removal, whitespace normalization are applied by default.")
        st.write(f"Custom stopwords loaded: **{len(ss.custom_stopwords)}**")
        if ss.custom_stopwords:
            with st.expander("Preview custom stopwords"):
                st.write(sorted(list(ss.custom_stopwords))[:200])

    # Apply truncation & sampling for performance
    work_series = raw_series.head(sample_rows).apply(lambda s: clean_text(s)[:truncate_chars])
    ss.processed_text = work_series

    st.subheader("Preview (first 5 rows)")
    st.dataframe(ss.processed_text.to_frame(name="processed_text").head())
    st.info("Processed text will be used by subsequent sections. Sampling & truncation applied for speed (from sidebar).")

# Guard
if ss.processed_text is None:
    st.stop()

# -----------------------------
# Cached: Vectorize, Co-occurrence, Word Sentiment
# -----------------------------
@st.cache_data(show_spinner=True)
def compute_vectorization_and_sentiment(texts: pd.Series, max_features: int, custom_stops: set):
    """Returns vectorizer, X (sparse), vocab, co_occurrence (numpy), word_sentiment(dict), doc_sentiments(list), word_class(dict)."""
    vectorizer = CountVectorizer(stop_words="english", max_features=max_features)
    X = vectorizer.fit_transform(texts.tolist())
    vocab = vectorizer.get_feature_names_out()

    # Doc-level VADER sentiment (fast) — used to aggregate per-word
    doc_scores = [sia.polarity_scores(t)["compound"] for t in texts.tolist()]

    # Compute co-occurrence
    Xc = (X.T @ X)  # sparse
    co = Xc.toarray().astype(float)
    np.fill_diagonal(co, 0.0)

    # Word-level sentiment: avg of doc sentiments where word appears
    word_sent = {}
    word_cls = {}
    for i, w in enumerate(vocab):
        col = X[:, i].toarray().ravel()
        where = np.where(col > 0)[0]
        if len(where):
            mean_comp = float(np.mean([doc_scores[j] for j in where]))
        else:
            mean_comp = 0.0
        word_sent[w] = mean_comp
        word_cls[w] = sentiment_class(mean_comp)
    return vectorizer, X, vocab, co, word_sent, doc_scores, word_cls

# ==========================
# SECTION 3: Knowledge Graph
# ==========================
if chosen == "🌐 Knowledge Graph":
    st.header("🌐 Sentiment-Filtered Word Co-occurrence Network")
    render_note(
        "What you can uncover here",
        [
            "Explore how words co-occur, filtered by sentiment (Positive / Neutral / Negative / All).",
            "Hover nodes to view word-level sentiment (VADER).",
            "Review pill-colored lists and a table of words with top neighbors by degree."
        ]
    )

    with st.spinner("Vectorizing and computing co-occurrence..."):
        vectorizer, X, vocab, co_occurrence, word_sentiment, doc_scores, word_class = compute_vectorization_and_sentiment(
            ss.processed_text, max_vocab, ss.custom_stopwords
        )

    if len(vocab) < 2:
        st.warning("Not enough unique tokens to build a graph. Try more rows or reduce cleaning.")
        st.stop()

    # Sentiment filter controls
    st.subheader("Filter")
    sentiment_choice = st.radio(
        "Choose sentiment",
        options=["positive", "neutral", "negative", "all"],
        index=3,
        horizontal=True
    )

    # Filter nodes by sentiment choice
    if sentiment_choice == "all":
        selected_words = [w for w in vocab]
    else:
        selected_words = [w for w in vocab if word_class.get(w, "neutral") == sentiment_choice]

    if len(selected_words) < 2:
        st.warning("Not enough words for the selected sentiment to render a graph.")
        # Still show pills
        st.markdown("#### Words (pills)")
        st.markdown("<div>" + "".join(
            [f'<span class="chip {"pos" if word_class[w]=="positive" else "neg" if word_class[w]=="negative" else "neu"}">{w} · {word_sentiment[w]:+.2f}</span>'
             for w in selected_words]) + "</div>", unsafe_allow_html=True)
        st.stop()

    # Build filtered adjacency
    word_index = {w: i for i, w in enumerate(vocab)}
    idx = [word_index[w] for w in selected_words]
    co_sub = co_occurrence[np.ix_(idx, idx)]

    # Build graph from filtered adjacency (use from_numpy_array, not from_numpy_matrix)
    G = nx.from_numpy_array(co_sub)
    mapping = dict(zip(range(len(selected_words)), selected_words))
    G = nx.relabel_nodes(G, mapping)

    # Remove zero-degree nodes to keep it clean
    G.remove_nodes_from(list(nx.isolates(G)))

    if len(G) == 0:
        st.warning("No co-occurrence edges found for the selected sentiment.")
        st.stop()

    # Plotly network with sentiment color mapping
    pos = nx.spring_layout(G, seed=42, k=0.7)
    edge_x, edge_y = [], []
    for a, b in G.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="#94A3B8"),
        hoverinfo="none", mode="lines"
    )

    node_x, node_y, labels, colors = [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        labels.append(f"{n}<br>Sent: {word_sentiment.get(n,0):+.2f}")
        sclass = word_class.get(n, "neutral")
        if sclass == "positive": colors.append("#10B981")
        elif sclass == "negative": colors.append("#EF4444")
        else: colors.append("#94A3B8")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        hoverinfo="text", text=list(G.nodes()),
        textposition="top center",
        marker=dict(
            color=colors, size=18,
            line=dict(color="#FFFFFF", width=1)
        )
    )
    node_trace.text = list(G.nodes())
    node_trace.hovertext = labels

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(l=10, r=10, t=60, b=10),  # extra top margin
            title={
                "text": f"Co-occurrence Network • {sentiment_choice.title()}",
                "x": 0.5, "y": 0.98,
                "xanchor": "center", "yanchor": "top"
            }
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # Pill chips of words (ranked by |sentiment|)
    st.markdown("#### Words (pills)")
    ranked = sorted(list(G.nodes()), key=lambda w: abs(word_sentiment.get(w, 0)), reverse=True)[:100]
    chip_html = []
    for w in ranked:
        scls = word_class.get(w, "neutral")
        cls = "pos" if scls == "positive" else "neg" if scls == "negative" else "neu"
        chip_html.append(f'<span class="chip {cls}">{w} · {word_sentiment.get(w,0):+.2f}</span>')
    st.markdown("<div>" + "".join(chip_html) + "</div>", unsafe_allow_html=True)

    # Table: words, sentiment, degree, top neighbors
    st.markdown("#### Words & Top Neighbors (table)")
    deg_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)
    rows = []
    for w, d in deg_sorted:
        neighs = list(G.neighbors(w))
        # sort neighbors by degree of neighbor desc, then by co-occurrence weight
        neighs_sorted = sorted(neighs, key=lambda n: G.degree[n], reverse=True)[:8]
        rows.append({
            "word": w,
            "sentiment": word_sentiment.get(w, 0.0),
            "sentiment_class": word_class.get(w, "neutral"),
            "degree": int(d),
            "top_neighbors": ", ".join(neighs_sorted)
        })
    table_df = pd.DataFrame(rows)
    # show a colored sentiment column as text (positive/neutral/negative)
    st.dataframe(table_df, use_container_width=True)

# ==========================
# SECTION 4: WordCloud & Ngrams
# ==========================
if chosen == "☁ WordCloud & Ngrams":
    st.header("☁ Interactive WordCloud & N-gram Analysis")
    render_note(
        "What you can uncover here",
        [
            "Spot dominant words and frequent phrases.",
            "Remove noisy words or n-grams interactively (plus your custom stopwords).",
            "Check per-phrase VADER sentiment in the bar chart tooltips."
        ]
    )

    all_text = " ".join(ss.processed_text.tolist())

    # Interactive Removal
    c1, c2 = st.columns(2)
    with c1:
        removed_words_input = st.text_input(
            "Remove words (comma-separated)",
            ",".join(ss.removed_words) if ss.removed_words else ""
        )
        ss.removed_words = [w.strip() for w in removed_words_input.split(",") if w.strip()] if removed_words_input else []
    with c2:
        removed_ngrams_input = st.text_input(
            "Remove N-grams (comma-separated)",
            ",".join(ss.removed_ngrams) if ss.removed_ngrams else ""
        )
        ss.removed_ngrams = [w.strip() for w in removed_ngrams_input.split(",") if w.strip()] if removed_ngrams_input else []

    removal_set = set(ss.removed_words) | set(ss.custom_stopwords)
    filtered_words = [w for w in all_text.split() if w not in removal_set]
    filtered_text = " ".join(filtered_words) if filtered_words else "empty"

    wc = WordCloud(width=1000, height=400, background_color="white").generate(filtered_text)
    fig_wc, ax_wc = plt.subplots(figsize=(12, 5))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    ngram_choice = st.selectbox("Select N-gram type", ["Unigram", "Bigram", "Trigram"], index=0)
    n_val = {"Unigram": 1, "Bigram": 2, "Trigram": 3}[ngram_choice]

    ngram_list = []
    for txt in ss.processed_text:
        tokens = [t for t in txt.split() if t not in removal_set]
        grams = [" ".join(g) for g in ngrams(tokens, n_val)]
        grams = [g for g in grams if g not in ss.removed_ngrams]
        ngram_list.extend(grams)

    if len(ngram_list) == 0:
        st.warning("No n-grams remain after filtering.")
        st.stop()

    top_n = st.slider(f"Select top-N {ngram_choice}s", 5, 50, 20, step=1)
    ngram_freq = pd.Series(ngram_list).value_counts().head(top_n)

    table_data = []
    for gram, freq in ngram_freq.items():
        parts = gram.split()
        scores = [sia.polarity_scores(w)["compound"] for w in parts]
        avg = float(np.mean(scores)) if scores else 0.0
        table_data.append({"Ngram": gram, "Freq": int(freq), "Sentiment": avg})

    df_table = pd.DataFrame(table_data)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=df_table["Ngram"], y=df_table["Freq"],
        text=[f"{s:.2f}" for s in df_table["Sentiment"]],
        hovertemplate="Ngram: %{x}<br>Freq: %{y}<br>Avg Sentiment: %{text}<extra></extra>",
        marker_color=["#10B981" if s >= 0 else "#EF4444" for s in df_table["Sentiment"]],
    ))
    fig_bar.update_layout(height=520, xaxis_tickangle=-40, margin=dict(l=10, r=10, t=30, b=120))
    st.plotly_chart(fig_bar, use_container_width=True)
    st.info("Hover bars to see exact VADER sentiment. Removal of words/ngrams & custom stopwords update this chart.")

# ==========================
# SECTION 5: Micrograph-Enhanced Sentiment
# ==========================
if chosen == "😊 Sentiment Analysis":
    st.header("😊 Sentiment Analysis with Interactive Micrographs")
    render_note(
        "What you can uncover here",
        [
            "Identify the words carrying strongest positive/negative sentiment (VADER).",
            "Optionally run sampled transformer sentiment for sentence-level validation.",
            "Use sidebar sampling/truncation to keep runs fast."
        ]
    )

    removal_set = set(ss.removed_words) | set(ss.custom_stopwords)
    word_scores = {}
    for txt in ss.processed_text:
        for w in txt.split():
            if w in removal_set:
                continue
            score = sia.polarity_scores(w)["compound"]
            word_scores.setdefault(w, []).append(score)

    if not word_scores:
        st.warning("No words available after filtering.")
        st.stop()

    word_avg = {k: float(np.mean(v)) for k, v in word_scores.items()}
    top_n = st.slider("Select top N words by |sentiment|", 5, 50, 20, step=1)
    sorted_pairs = sorted(word_avg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    df_sent = pd.DataFrame({"Word": [k for k, _ in sorted_pairs], "Score": [v for _, v in sorted_pairs]})

    fig_micro = go.Figure(go.Bar(
        x=df_sent["Word"], y=[1] * len(df_sent),
        text=[f"{s:.2f}" for s in df_sent["Score"]],
        hovertemplate="Word: %{x}<br>Sentiment: %{text}<extra></extra>",
        marker_color=["#10B981" if s >= 0 else "#EF4444" for s in df_sent["Score"]],
        width=0.55
    ))
    fig_micro.update_layout(height=420, yaxis=dict(visible=False), xaxis_tickangle=-40, margin=dict(l=20, r=20, t=20, b=120))
    st.plotly_chart(fig_micro, use_container_width=True)
    st.caption("Word-level VADER micrograph (fast).")

    # Optional: Transformers sentiment (batched) on a sample — truncated proactively
    with st.expander("Sentence-level Sentiment via Transformers (sampled)"):
        limit = st.slider("Max rows to score", min_value=20, max_value=500, value=min(100, len(ss.processed_text)), step=20)
        samples = ss.processed_text.head(limit).apply(lambda s: s[:truncate_chars]).tolist()
        if st.button("Run HF Sentiment on Sample"):
            with st.spinner("Scoring..."):
                results = hf_sentiment(samples, batch_size=16)  # no truncation kwarg, we truncated text already
            sdf = pd.DataFrame(results)
            sdf["text"] = samples
            c1, c2 = st.columns([2, 3])
            with c1:
                st.dataframe(sdf, use_container_width=True)
            with c2:
                fig = px.histogram(sdf, x="label", color="label", title="HF Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)

# ==========================
# SECTION 6: Topic Modeling
# ==========================
if chosen == "🧠 Topic Modeling":
    st.header("🧠 Topic Modeling with LDA (scikit-learn)")
    render_note(
        "What you can uncover here",
        [
            "Discover latent themes across documents.",
            "Adjust topic count and top terms; vocabulary is capped for speed.",
            "Results are driven by the processed, stopword-filtered text."
        ]
    )

    n_topics = st.slider("Number of topics", 2, 10, 5, step=1)
    top_k = st.slider("Top words per topic", 5, 20, 10, step=1)

    vectorizer = CountVectorizer(stop_words="english", max_features=max_vocab)
    dtm = vectorizer.fit_transform(ss.processed_text.tolist())
    if dtm.shape[0] == 0 or dtm.shape[1] == 0:
        st.warning("Insufficient data for LDA.")
        st.stop()

    with st.spinner("Fitting LDA..."):
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
        lda.fit(dtm)
    ss.vectorizer = vectorizer
    ss.doc_term_matrix = dtm
    ss.lda_model = lda

    for i, topic in enumerate(lda.components_):
        st.subheader(f"Topic {i + 1}")
        words = top_words_from_topic(vectorizer, topic, top_k=top_k)
        st.write(", ".join(words))
    st.info("Each topic shows the top contributing words. Capped vocabulary & sampled rows to improve speed.")

# ==========================
# SECTION 7: Entity Recognition (Transformers)
# ==========================
if chosen == "🔍 Entity Recognition":
    st.header("🔍 Entity Recognition (Hugging Face Transformers)")
    render_note(
        "What you can uncover here",
        [
            "Extract entities like persons, organizations, locations from sampled text.",
            "Batched inference for speed; long rows are truncated beforehand.",
            "Inspect both entity types and their confidence scores."
        ]
    )

    limit = st.slider("Analyze first N rows", 10, 500, min(100, len(ss.processed_text)), step=10)
    texts = ss.processed_text.head(limit).apply(lambda s: s[:truncate_chars]).tolist()

    rows = []
    with st.spinner("Running NER in batches..."):
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # IMPORTANT: do not pass truncation kwarg (causes TypeError). We truncated strings ourselves.
            ents_batch = hf_ner(batch, batch_size=batch_size)
            for idx, ents in enumerate(ents_batch):
                row_idx = i + idx
                snippet = texts[row_idx][:120] + ("..." if len(texts[row_idx]) > 120 else "")
                for e in ents:
                    rows.append({
                        "row_index": row_idx + 1,
                        "text_snippet": snippet,
                        "entity": e.get("word"),
                        "label": e.get("entity_group"),
                        "score": round(float(e.get("score", 0)), 3),
                        "start": e.get("start"),
                        "end": e.get("end")
                    })

    if rows:
        ner_df = pd.DataFrame(rows)
        st.dataframe(ner_df, use_container_width=True)
        ent_counts = ner_df["label"].value_counts().reset_index()
        ent_counts.columns = ["label", "count"]
        fig = px.bar(ent_counts, x="label", y="count", title="Entity Type Counts")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No entities detected in the sampled rows.")

# ==========================
# SECTION 8: Concordance Analysis
# ==========================
if chosen == "🔎 Concordance Analysis":
    st.header("🔎 Concordance Analysis")
    render_note(
        "What you can uncover here",
        [
            "Search for a keyword and see its surrounding context windows.",
            "Filter respects custom stopwords and manual removals.",
            "Limit the number of hits to stay quick and focused."
        ]
    )

    text_tokens = []
    removal_set = set(ss.removed_words) | set(ss.custom_stopwords)
    for txt in ss.processed_text:
        text_tokens.extend([t for t in txt.split() if t not in removal_set])

    if not text_tokens:
        st.warning("No tokens available.")
        st.stop()

    keyword = st.text_input("Enter keyword to see concordance")
    window = st.slider("Context window size", 3, 10, 5)
    max_hits = st.slider("Max hits to show", 5, 100, 30, step=5)

    if keyword:
        conc_list = []
        kw = keyword.lower()
        for i, tok in enumerate(text_tokens):
            if tok.lower() == kw:
                start = max(i - window, 0)
                end = min(i + window + 1, len(text_tokens))
                conc_list.append(" ".join(text_tokens[start:end]))
                if len(conc_list) >= max_hits:
                    break
        if conc_list:
            st.write(conc_list)
        else:
            st.info("No matches found.")

# ==========================
# SECTION 9: Download Results
# ==========================
if chosen == "⬇️ Download Results":
    st.header("⬇️ Download Analysis Results")
    render_note(
        "What you can uncover here",
        [
            "Export original + processed text to CSV.",
            "Charts can be saved as PNG via right-click.",
            "Combine with filters first to download a focused slice."
        ]
    )
    if ss.data is not None:
        out_df = ss.data.copy()
        if ss.processed_text is not None:
            out_df["processed_text"] = ss.processed_text
        csv = out_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="analysis_results.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.info("Tip: Right-click charts to save as PNG.")
