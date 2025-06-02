import streamlit as st
from collections import Counter
from langdetect import detect
import spacy
import pandas as pd
import csv
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from spacy import displacy

# ----- Core Logic Class -----
class TextProcessor:
    def __init__(self, filename):
        self.filename = filename
        self.text = []
        self.tokens = []
        self.lemmas = []
        self.pos = []
        self.sentences = []
        self.sent_docs = []
        self.entities = []
        self.language = None
        self.doc = None
        self.nlp_models = self._load_models()

    def _load_models(self):
        models = {}
        for lang, model_name in {"en": "en_core_web_sm", "fr": "fr_core_news_sm"}.items():
            try:
                models[lang] = spacy.load(model_name)
            except OSError:
                from spacy.cli import download
                download(model_name)
                models[lang] = spacy.load(model_name)
        return models

    def load_text(self, strict=True):
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File {self.filename} does not exist.")

        with open(self.filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if strict and (len(lines) < 5 or len(lines) > 100):
            print(f"Warning: File contains {len(lines)} lines, which does not meet the recommended range (5-100).")

        self.text = lines
        return lines

    def detect_language(self):
        joined_text = '\n'.join(self.text)
        self.language = detect(joined_text)
        if self.language not in self.nlp_models:
            raise ValueError(f"Unsupported language detected: {self.language}")
        return self.language

    def process_text(self):
        if not self.text:
            raise ValueError("Text not loaded. Call load_text() first.")

        if not self.language:
            self.detect_language()

        nlp = self.nlp_models[self.language]
        self.doc = nlp('\n'.join(self.text))

        self.tokens = [token.text for token in self.doc if not token.is_space]
        self.lemmas = [token.lemma_ for token in self.doc if not token.is_space]
        self.pos = [token.pos_ for token in self.doc if not token.is_space]
        self.sentences = list(self.doc.sents)
        self.sent_docs = [sent.as_doc() for sent in self.sentences]
        self.entities = [(ent.text, ent.label_) for ent in self.doc.ents]

        return self.doc

    def word_counts(self, remove_stopwords=False):
        words = [t for t in self.tokens if not t.lower() in self._get_stop_words()] if remove_stopwords else self.tokens
        return Counter(words)

    def _get_stop_words(self):
        if not self.language:
            return set()
        return self.nlp_models[self.language].Defaults.stop_words

    def write_csv(self, output_file):
        if not (self.tokens and self.lemmas and self.pos):
            raise ValueError("No processed data to write. Call process_text() first.")

        word_counts = self.word_counts()
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['index', 'Token', 'Lemma', 'POS', 'word_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for idx, (token, lemma, pos) in enumerate(zip(self.tokens, self.lemmas, self.pos), start=1):
                writer.writerow({
                    'index': idx,
                    'Token': token,
                    'Lemma': lemma,
                    'POS': pos,
                    'word_count': word_counts[token]
                })

# ----- Streamlit App -----
st.set_page_config(page_title="Text Processor", layout="wide")
st.title("📝 Text Processor GUI")

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file:
    temp_path = "uploaded_text.txt"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    if 'processor' not in st.session_state:
        st.session_state.processor = TextProcessor(temp_path)
        st.session_state.lines = st.session_state.processor.load_text(strict=False)
        st.session_state.language = st.session_state.processor.detect_language()
        st.session_state.doc = st.session_state.processor.process_text()
        st.session_state.word_counts = st.session_state.processor.word_counts(remove_stopwords=True)

    processor = st.session_state.processor

    st.success(f"Loaded {len(st.session_state.lines)} lines.")
    st.info(f"Detected Language: {st.session_state.language.upper()}")

    remove_stop = st.checkbox("Remove stop words in frequency & wordcloud", value=True)

    if st.button("🔍 Re-process Text"):
        st.session_state.doc = processor.process_text()
        st.session_state.word_counts = processor.word_counts(remove_stopwords=remove_stop)

    word_counts = st.session_state.word_counts

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Token Table", "🧾 Sentences", "🧠 Entities", "📊 Stats", "☁️ Word Cloud", "🌐 Dependency"])

    with tab1:
        df = pd.DataFrame({
            "Token": processor.tokens,
            "Lemma": processor.lemmas,
            "POS": processor.pos,
            "Word Count": [word_counts[t] for t in processor.tokens]
        })
        st.dataframe(df, use_container_width=True)
        csv_name = "output_from_gui.csv"
        processor.write_csv(csv_name)
        with open(csv_name, "rb") as f:
            st.download_button("📥 Download CSV", data=f, file_name=csv_name, mime="text/csv")

    with tab2:
        for i, sent in enumerate(processor.sentences, 1):
            st.markdown(f"**Sentence {i}:** {sent.text.strip()}")

    with tab3:
        if processor.entities:
            entity_df = pd.DataFrame(processor.entities, columns=["Entity", "Label"])
            st.dataframe(entity_df, use_container_width=True)
        else:
            st.info("No named entities found.")

    with tab4:
        st.subheader("📊 Word Frequency (Top 20)")
        top_words = word_counts.most_common(20)
        top_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])
        st.bar_chart(top_df.set_index("Word"))

        st.subheader("📊 POS Distribution (Bar Chart)")
        pos_counts = Counter(processor.pos)
        pos_df = pd.DataFrame(pos_counts.items(), columns=["POS", "Count"]).sort_values(by="Count")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(pos_df["POS"], pos_df["Count"], color="skyblue")
        ax.set_xlabel("Count")
        ax.set_title("POS Tag Distribution")
        st.pyplot(fig)

    with tab5:
        wc = WordCloud(width=800, height=300, background_color="white").generate(" ".join(
            [t for t in processor.tokens if t.lower() not in processor._get_stop_words()]))
        fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    with tab6:
        if processor.sent_docs:
            sentence_options = [sent.text.strip() for sent in processor.sentences[:5]]
            selected_idx = st.selectbox("Select a sentence to visualize (max 5 shown):", range(len(sentence_options)),
                                        format_func=lambda i: sentence_options[i])
            selected_doc = processor.sent_docs[selected_idx]
            html = displacy.render(selected_doc, style="dep", page=True)
            st.components.v1.html(html, height=350, scrolling=True)
        else:
            st.warning("No sentence data available for dependency visualization.")
