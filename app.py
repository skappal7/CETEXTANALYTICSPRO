# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import Text
from nltk.util import ngrams
import re
import base64
import nltk

# ---- NLTK Downloads ----
nltk.download('punkt')
nltk.download('vader_lexicon')

# ---- CSS Styling ----
st.markdown("""
<style>
.stButton>button {
    border-radius: 50px;
    padding: 0.5em 1.5em;
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    margin: 5px;
}
.stTabs [role="tab"] {
    font-weight: bold;
    color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

# ---- Initialize NLP Tools ----
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

sia = SentimentIntensityAnalyzer()

# ---- App State ----
for key in ['data','processed_text','vectorizer','doc_term_matrix','lda_model','knowledge_graph',
            'word_sentiment','removed_words','removed_ngrams']:
    if key not in st.session_state: st.session_state[key] = None
if 'removed_words' not in st.session_state: st.session_state.removed_words = []
if 'removed_ngrams' not in st.session_state: st.session_state.removed_ngrams = []

# ---- Tabs ----
tabs = st.tabs(["Upload Data", "Text Cleaning & Preview", "Knowledge Graph",
                "WordCloud & Ngrams", "Sentiment Analysis",
                "Topic Modeling", "Entity Recognition",
                "Concordance Analysis", "Download Results"])

# ==========================
# TAB 1: Upload Data
# ==========================
with tabs[0]:
    st.header("Upload Your Review/Transcription Files")
    uploaded_file = st.file_uploader("Upload CSV, XLSX, or TXT file", type=['csv','xlsx','txt'])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            text = uploaded_file.read().decode('utf-8')
            df = pd.DataFrame({'text':[text]})
        st.session_state.data = df
        st.success("Data Uploaded Successfully!")
        st.dataframe(df.head())

# ==========================
# TAB 2: Text Cleaning & Preview
# ==========================
with tabs[1]:
    if st.session_state.data is not None:
        st.header("Text Cleaning and Preview")
        text_column = st.selectbox("Select Text Column", st.session_state.data.columns)
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            return text
        st.session_state.processed_text = st.session_state.data[text_column].apply(clean_text)
        st.dataframe(st.session_state.processed_text.to_frame().head())

# ==========================
# TAB 3: Knowledge Graph
# ==========================
with tabs[2]:
    if st.session_state.processed_text is not None:
        st.header("Interactive Knowledge Graph with Sentiment Tooltips")
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(st.session_state.processed_text)
        words = vectorizer.get_feature_names_out()
        co_occurrence = (X.T @ X).toarray()
        np.fill_diagonal(co_occurrence, 0)
        G = nx.from_numpy_matrix(co_occurrence)
        mapping = dict(zip(range(len(words)), words))
        G = nx.relabel_nodes(G, mapping)
        st.session_state.knowledge_graph = G

        word_sentiment = {w: np.mean([sia.polarity_scores(txt)['compound'] for txt in st.session_state.processed_text if w in txt]) for w in words}
        
        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'),
                                hoverinfo='none', mode='lines')

        node_x, node_y, node_text, node_color = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}<br>Sentiment: {word_sentiment[node]:.2f}")
            node_color.append(word_sentiment[node])

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=list(G.nodes()),
            textposition='top center',
            marker=dict(showscale=True, colorscale='RdYlGn', color=node_color, size=20,
                        colorbar=dict(title='Sentiment Score'))
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(showlegend=False, hovermode='closest'))
        st.plotly_chart(fig, use_container_width=True)
        st.info("Hover nodes to see average sentiment for that word.")

# ==========================
# TAB 4: WordCloud & Ngrams
# ==========================
with tabs[3]:
    if st.session_state.processed_text is not None:
        st.header("Interactive WordCloud & Ngram Analysis with Hover Tooltips")
        all_text = ' '.join(st.session_state.processed_text.tolist())

        # --- Interactive Word Removal ---
        removed_words_input = st.text_input("Remove words (comma-separated)", ','.join(st.session_state.removed_words))
        st.session_state.removed_words = [w.strip() for w in removed_words_input.split(',')] if removed_words_input else []

        # --- Interactive Ngram Removal ---
        removed_ngrams_input = st.text_input("Remove Ngrams (comma-separated)", ','.join(st.session_state.removed_ngrams))
        st.session_state.removed_ngrams = [w.strip() for w in removed_ngrams_input.split(',')] if removed_ngrams_input else []

        # --- Filtered Text ---
        filtered_words = [w for w in all_text.split() if w not in st.session_state.removed_words]
        filtered_text = ' '.join(filtered_words)

        # --- WordCloud ---
        wc = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)
        fig_wc, ax = plt.subplots(figsize=(12,6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig_wc)

        # --- Ngrams ---
        ngram_range = st.selectbox("Select Ngram Type", ['Unigram','Bigram','Trigram'])
        n = {'Unigram':1,'Bigram':2,'Trigram':3}[ngram_range]
        ngram_list = []
        for txt in st.session_state.processed_text:
            tokens = [t for t in txt.split() if t not in st.session_state.removed_words]
            ngram_list += [' '.join(gram) for gram in ngrams(tokens,n) if ' '.join(gram) not in st.session_state.removed_ngrams]

        # Top N selection
        top_n = st.slider(f"Select top N {ngram_range}s", 5, 50, 20)
        ngram_freq = pd.Series(ngram_list).value_counts().head(top_n)

        # --- Micrograph Table with Plotly Bars (interactive) ---
        table_data = []
        for gram, freq in ngram_freq.items():
            words_in_gram = gram.split()
            scores = [sia.polarity_scores(w)['compound'] for w in words_in_gram]
            avg_score = np.mean(scores)
            table_data.append({'Ngram':gram, 'Freq':freq, 'Sentiment':avg_score})
        df_table = pd.DataFrame(table_data)

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=df_table['Ngram'],
            y=df_table['Freq'],
            text=[f"{s:.2f}" for s in df_table['Sentiment']],
            hovertemplate='Ngram: %{x}<br>Freq: %{y}<br>Sentiment: %{text}<extra></extra>',
            marker_color=['green' if s>=0 else 'red' for s in df_table['Sentiment']]
        ))
        fig_bar.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.info("Hover over bars to see exact sentiment score. Removal of words/ngrams updates this chart dynamically.")

# ==========================
# TAB 5: Micrograph-Enhanced Sentiment Analysis
# ==========================
with tabs[4]:
    if st.session_state.processed_text is not None:
        st.header("Sentiment Analysis with Interactive Micrographs")
        word_scores = {}
        for txt in st.session_state.processed_text:
            for w in txt.split():
                if w in st.session_state.removed_words:
                    continue
                score = sia.polarity_scores(w)['compound']
                word_scores.setdefault(w, []).append(score)
        word_avg = {k: np.mean(v) for k,v in word_scores.items()}

        top_n = st.slider("Select top N words for sentiment visualization", 5, 50, 20)
        sorted_words = dict(sorted(word_avg.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n])
        df_sent = pd.DataFrame({'Word':list(sorted_words.keys()), 'Score':list(sorted_words.values())})

        fig_micro = go.Figure(go.Bar(
            x=df_sent['Word'],
            y=[1]*len(df_sent),
            text=[f"{s:.2f}" for s in df_sent['Score']],
            hovertemplate='Word: %{x}<br>Sentiment: %{text}<extra></extra>',
            marker_color=['green' if s>=0 else 'red' for s in df_sent['Score']],
            width=0.5
        ))
        fig_micro.update_layout(height=400, yaxis=dict(visible=False), xaxis_tickangle=-45)
        st.plotly_chart(fig_micro, use_container_width=True)
        st.info("Hover bars to see exact sentiment per word.")

# ==========================
# TAB 6: Topic Modeling
# ==========================
with tabs[5]:
    if st.session_state.processed_text is not None:
        st.header("Topic Modeling with LDA")
        vectorizer = CountVectorizer(stop_words='english')
        dtm = vectorizer.fit_transform(st.session_state.processed_text)
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(dtm)
        st.session_state.vectorizer = vectorizer
        st.session_state.doc_term_matrix = dtm
        st.session_state.lda_model = lda
        for i, topic in enumerate(lda.components_):
            st.subheader(f"Topic {i+1}")
            top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
            st.write(", ".join(top_words))
        st.info("Each topic shows the top 10 contributing words.")

# ==========================
# TAB 7: Entity Recognition
# ==========================
with tabs[6]:
    if st.session_state.processed_text is not None:
        st.header("Entity Recognition")
        entities_list = []
        for doc in nlp.pipe(st.session_state.processed_text, disable=["tagger","parser"]):
            entities_list.append([(ent.text, ent.label_) for ent in doc.ents])
        st.write(entities_list)
        st.info("Named entities extracted using spaCy.")

# ==========================
# TAB 8: Concordance Analysis
# ==========================
with tabs[7]:
    if st.session_state.processed_text is not None:
        st.header("Concordance Analysis")
        text_tokens = [token for txt in st.session_state.processed_text for token in txt.split()]
        text_obj = Text(text_tokens)
        keyword = st.text_input("Enter keyword to see concordance")
        if keyword:
            conc_list = []
            for i in range(len(text_tokens)):
                if text_tokens[i].lower() == keyword.lower():
                    start = max(i-5,0)
                    end = min(i+6,len(text_tokens))
                    conc_list.append(' '.join(text_tokens[start:end]))
            st.write(conc_list)
        st.info("Shows context around the keyword within the text.")

# ==========================
# TAB 9: Download Results
# ==========================
with tabs[8]:
    st.header("Download Analysis Results")
    if st.session_state.data is not None:
        csv = st.session_state.data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="analysis_results.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.info("Graphs can be right-clicked to save as PNG images.")
