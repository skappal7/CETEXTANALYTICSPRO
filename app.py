# CE Text Analytics Pro — Full App (No spaCy)
# -------------------------------------------
# Features (parity with your 9-tab app):
# 1) Upload (CSV/XLSX/TXT)
# 2) Text Cleaning & Preview
# 3) Knowledge Graph (co-occurrence) with sentiment tooltips (fixed for NetworkX 3.x)
# 4) WordCloud & N-grams with interactive removal + sentiment hover
# 5) Micrograph-Enhanced Sentiment (VADER per-word)
# 6) Topic Modeling (LDA)
# 7) Entity Recognition (Hugging Face transformers)
# 8) Concordance
# 9) Download Results
#
# Extras:
# - Optional sentence-level sentiment via HF pipeline (toggle in Sentiment tab)
# - Left navigation in pill style
# - Safety guards for empty data / low-vocab
# - Caching for HF pipelines

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safer on Streamlit Cloud
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
from transformers import pipeline  # Hugging Face

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="CE Text Analytics Pro", layout="wide")

# -----------------------------
# NLTK downloads (quiet)
# -----------------------------
nltk.download("punkt", quiet=True)
nltk.download("vader_lexicon", quiet=True)

# -----------------------------
# CSS: Left nav pill style + general UI tweaks
# -----------------------------
st.markdown(
    """
    <style>
    /* Sidebar radio -> pill style */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 8px !important;
    }
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

    /* Tab-like section headers */
    h2, h3 { color: #0F172A; }
    .small-muted { color: #64748B; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Hugging Face Pipelines (cached)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_pipelines():
    # CPU-friendly default models
    hf_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    hf_ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return hf_sentiment, hf_ner

hf_sentiment, hf_ner = load_pipelines()

# -----------------------------
# VADER Sentiment (word-level)
# -----------------------------
sia = SentimentIntensityAnalyzer()

# -----------------------------
# Helpers
# -----------------------------
def clean_text(s: str) -> str:
    s = str(s)
    s = s.lower()
    # keep letters, digits, spaces
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ensure_text_series(df: pd.DataFrame, col: str) -> pd.Series:
    ser = df[col].fillna("").astype(str)
    return ser

def top_words_from_topic(vectorizer: CountVectorizer, topic_row: np.ndarray, top_k: int = 10):
    # topic_row: component array
    idxs = topic_row.argsort()[-top_k:][::-1]
    vocab = vectorizer.get_feature_names_out()
    return [vocab[i] for i in idxs]

# -----------------------------
# Session State
# -----------------------------
ss = st.session_state
DEFAULT_KEYS = [
    "data", "processed_text", "vectorizer", "doc_term_matrix",
    "lda_model", "knowledge_graph", "word_sentiment",
    "removed_words", "removed_ngrams"
]
for k in DEFAULT_KEYS:
    if k not in ss:
        ss[k] = None
if ss.get("removed_words") is None: ss["removed_words"] = []
if ss.get("removed_ngrams") is None: ss["removed_ngrams"] = []

# -----------------------------
# Left Navigation (Pill-style)
# -----------------------------
SECTIONS = [
    "📂 Upload Data",
    "🧹 Text Cleaning & Preview",
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
    st.markdown('<p class="small-muted">Left nav styled as pills</p>', unsafe_allow_html=True)

# ==========================
# SECTION 1: Upload Data
# ==========================
if chosen == "📂 Upload Data":
    st.header("📂 Upload Your Review/Transcription Files")
    uploaded_file = st.file_uploader("Upload CSV, XLSX, or TXT file", type=['csv', 'xlsx', 'txt'])

    if uploaded_file:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.lower().endswith(".xlsx"):
                # Requires openpyxl
                df = pd.read_excel(uploaded_file)
            else:
                # TXT -> single column 'text'
                text = uploaded_file.read().decode("utf-8", errors="ignore")
                df = pd.DataFrame({"text": [text]})
            ss.data = df
            st.success("✅ Data Uploaded Successfully!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    if ss.data is None:
        st.info("Upload a file to begin. Supported: CSV, XLSX, TXT")

# Guard: stop other sections if no data yet
if ss.data is None:
    st.stop()

# ==========================
# SECTION 2: Text Cleaning & Preview
# ==========================
if chosen == "🧹 Text Cleaning & Preview":
    st.header("🧹 Text Cleaning and Preview")

    text_column = st.selectbox("Select Text Column", ss.data.columns, index=0)
    raw_text_series = ensure_text_series(ss.data, text_column)

    with st.expander("Cleaning Options", expanded=True):
        st.caption("We apply lowercasing, punctuation removal, and whitespace normalization.")
        # future hooks for more cleaning toggles

    ss.processed_text = raw_text_series.apply(clean_text)
    st.subheader("Preview (first 5 rows)")
    st.dataframe(ss.processed_text.to_frame(name="processed_text").head())

    st.info("Processed text is used by subsequent sections.")

# ==========================
# SECTION 3: Knowledge Graph
# ==========================
if chosen == "🌐 Knowledge Graph":
    st.header("🌐 Interactive Knowledge Graph with Sentiment Tooltips")

    if ss.processed_text is None:
        st.warning("Go to 'Text Cleaning & Preview' first.")
        st.stop()

    # Vectorize
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(ss.processed_text)

    # Safety: need at least 2 unique terms
    if X.shape[1] < 2:
        st.warning("Not enough unique tokens to build a graph. Try a different dataset or less aggressive cleaning.")
        st.stop()

    words = vectorizer.get_feature_names_out()
    # term-term co-occurrence
    Xc = (X.T @ X)  # sparse
    co_occurrence = Xc.toarray().astype(float)
    np.fill_diagonal(co_occurrence, 0.0)

    # FIX: NetworkX 3.x removed from_numpy_matrix — use from_numpy_array
    G = nx.from_numpy_array(co_occurrence)
    # Map integer nodes to word labels
    mapping = dict(zip(range(len(words)), words))
    G = nx.relabel_nodes(G, mapping)
    ss.knowledge_graph = G

    # Word-level sentiment = average VADER compound across rows containing the word
    word_sentiment = {}
    texts_list = ss.processed_text.tolist()
    for w in words:
        scores = []
        # only compute when word appears
        for txt in texts_list:
            if w in txt:
                scores.append(sia.polarity_scores(txt)["compound"])
        word_sentiment[w] = float(np.mean(scores)) if scores else 0.0
    ss.word_sentiment = word_sentiment

    # Positions and traces
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

    node_x, node_y, node_texts, node_colors = [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_texts.append(f"{n}<br>Avg sentiment: {word_sentiment.get(n,0):.2f}")
        node_colors.append(word_sentiment.get(n, 0.0))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        hoverinfo="text", text=list(G.nodes()),
        textposition="top center",
        marker=dict(
            showscale=True, colorscale="RdYlGn",
            color=node_colors, size=18,
            colorbar=dict(title="Sentiment")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(showlegend=False, hovermode="closest",
                                     margin=dict(l=10, r=10, t=10, b=10)))
    st.plotly_chart(fig, use_container_width=True)
    st.info("Hover nodes to see average sentence-level VADER sentiment for texts containing each word.")

# ==========================
# SECTION 4: WordCloud & Ngrams
# ==========================
if chosen == "☁ WordCloud & Ngrams":
    st.header("☁ Interactive WordCloud & N-gram Analysis")

    if ss.processed_text is None:
        st.warning("Go to 'Text Cleaning & Preview' first.")
        st.stop()

    all_text = " ".join(ss.processed_text.tolist())

    # --- Interactive Removal ---
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

    # Filter words
    filtered_words = [w for w in all_text.split() if w not in ss.removed_words]
    filtered_text = " ".join(filtered_words) if filtered_words else "empty"

    # WordCloud
    wc = WordCloud(width=1000, height=400, background_color="white").generate(filtered_text)
    fig_wc, ax_wc = plt.subplots(figsize=(12, 5))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    # N-grams
    ngram_choice = st.selectbox("Select N-gram type", ["Unigram", "Bigram", "Trigram"], index=0)
    n_val = {"Unigram": 1, "Bigram": 2, "Trigram": 3}[ngram_choice]

    ngram_list = []
    for txt in ss.processed_text:
        tokens = [t for t in txt.split() if t not in ss.removed_words]
        grams = [" ".join(g) for g in ngrams(tokens, n_val)]
        # remove configured n-grams
        grams = [g for g in grams if g not in ss.removed_ngrams]
        ngram_list.extend(grams)

    if len(ngram_list) == 0:
        st.warning("No n-grams remain after filtering.")
        st.stop()

    top_n = st.slider(f"Select top-N {ngram_choice}s", 5, 50, 20, step=1)
    ngram_freq = pd.Series(ngram_list).value_counts().head(top_n)

    # Build table with sentiment (average of components' word scores)
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
    st.info("Hover bars to see exact VADER sentiment. Removing words/ngrams updates this chart.")

# ==========================
# SECTION 5: Micrograph-Enhanced Sentiment
# ==========================
if chosen == "😊 Sentiment Analysis":
    st.header("😊 Sentiment Analysis with Interactive Micrographs")

    if ss.processed_text is None:
        st.warning("Go to 'Text Cleaning & Preview' first.")
        st.stop()

    # Word-level VADER micrographs (your original logic)
    word_scores = {}
    for txt in ss.processed_text:
        for w in txt.split():
            if w in ss.removed_words:
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

    st.caption("Above uses word-level VADER to mimic your original micrograph view.")

    # Optional: sentence-level sentiment with HF (toggle)
    with st.expander("Sentence-level Sentiment (Hugging Face)"):
        limit = st.slider("Max number of rows to score", min_value=10, max_value=300, value=50, step=10)
        sample_texts = ss.processed_text.head(limit).tolist()
        if st.button("Run HF Sentiment on Sample"):
            results = hf_sentiment(sample_texts)
            sdf = pd.DataFrame(results)
            sdf["text"] = sample_texts
            c1, c2 = st.columns([2, 3])
            with c1:
                st.dataframe(sdf)
            with c2:
                fig = px.histogram(sdf, x="label", color="label", title="HF Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)

# ==========================
# SECTION 6: Topic Modeling
# ==========================
if chosen == "🧠 Topic Modeling":
    st.header("🧠 Topic Modeling with LDA (scikit-learn)")

    if ss.processed_text is None:
        st.warning("Go to 'Text Cleaning & Preview' first.")
        st.stop()

    n_topics = st.slider("Number of topics", 2, 10, 5, step=1)
    max_features = st.slider("Max vocabulary size", 500, 5000, 2000, step=100)
    top_k = st.slider("Top words per topic", 5, 20, 10, step=1)

    vectorizer = CountVectorizer(stop_words="english", max_features=max_features)
    dtm = vectorizer.fit_transform(ss.processed_text)
    if dtm.shape[0] == 0 or dtm.shape[1] == 0:
        st.warning("Insufficient data for LDA.")
        st.stop()

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
    lda.fit(dtm)
    ss.vectorizer = vectorizer
    ss.doc_term_matrix = dtm
    ss.lda_model = lda

    for i, topic in enumerate(lda.components_):
        st.subheader(f"Topic {i + 1}")
        words = top_words_from_topic(vectorizer, topic, top_k=top_k)
        st.write(", ".join(words))
    st.info("Each topic shows the top contributing words.")

# ==========================
# SECTION 7: Entity Recognition (Transformers)
# ==========================
if chosen == "🔍 Entity Recognition":
    st.header("🔍 Entity Recognition (Hugging Face Transformers)")
    if ss.processed_text is None:
        st.warning("Go to 'Text Cleaning & Preview' first.")
        st.stop()

    limit = st.slider("Analyze first N rows", 5, 200, 20, step=5)
    texts = ss.processed_text.head(limit).tolist()

    rows = []
    with st.spinner("Running NER..."):
        for idx, txt in enumerate(texts, start=1):
            ents = hf_ner(txt[:1000])  # clip long rows for speed
            for e in ents:
                rows.append({
                    "row_index": idx,
                    "text_snippet": txt[:120] + ("..." if len(txt) > 120 else ""),
                    "entity": e.get("word"),
                    "label": e.get("entity_group"),
                    "score": round(float(e.get("score", 0)), 3),
                    "start": e.get("start"),
                    "end": e.get("end")
                })

    if rows:
        ner_df = pd.DataFrame(rows)
        st.dataframe(ner_df)
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

    if ss.processed_text is None:
        st.warning("Go to 'Text Cleaning & Preview' first.")
        st.stop()

    text_tokens = [tok for txt in ss.processed_text for tok in txt.split()]
    if not text_tokens:
        st.warning("No tokens available.")
        st.stop()

    keyword = st.text_input("Enter keyword to see concordance")
    window = st.slider("Context window size", 3, 10, 5)
    max_hits = st.slider("Max hits to show", 5, 100, 30, step=5)

    if keyword:
        conc_list = []
        for i, tok in enumerate(text_tokens):
            if tok.lower() == keyword.lower():
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
    if ss.data is not None:
        out_df = ss.data.copy()
        if ss.processed_text is not None:
            # Append processed text as a column for download
            out_df["processed_text"] = ss.processed_text
        csv = out_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="analysis_results.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.info("Tip: Right-click charts to save as PNG.")
