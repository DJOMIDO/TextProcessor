# 📝 TextProcessor (Streamlit GUI Version)

An interactive NLP app built with **Streamlit** and **spaCy** that allows users to upload a `.txt` file and explore linguistic features of the text. It supports both **English** and **French**, automatically detecting the input language.

---

## 🚀 Features

- 📄 Upload and analyze `.txt` files
- 🌍 Language detection (English / French)
- ✂️ Sentence segmentation
- 🧠 Named Entity Recognition (NER)
- 📋 Tokenization, Lemmatization, POS tagging
- 📊 Word frequency chart
- 🧩 POS distribution bar chart
- ☁️ Word cloud with optional stop word removal
- 🌐 Dependency syntax visualization
- 📥 CSV export of token analysis

---

## 📁 Project Structure

```
TextProcessor/
├── text_processor_app.py     # Main Streamlit app
├── text_processor.py         # NLP processing class
├── requirements.txt          # Python dependencies
├── Makefile                  # Project automation (venv, run, clean)
├── .gitignore                # Excluded files and folders
└── sample_text.txt           # Example input file
```

---

## ⚙️ Setup Instructions

### 🧼 Clean Setup with Makefile (Recommended)

```bash
# Clone the repository
git clone <repo-url>
cd TextProcessor

# Create virtual environment and install dependencies
make venv

# Download required spaCy models
make models

# Run the Streamlit app
make run
```

---

## 🧪 Sample Text File

Use the provided `sample_text.txt` or upload your own `.txt` file (5–100 lines recommended).

---

## 📦 Dependencies

- Python 3.8+
- streamlit
- spacy
- langdetect
- pandas
- wordcloud
- matplotlib

All dependencies are listed in `requirements.txt`

---

## 📄 License

MIT License. This project is open for educational, demo, and NLP practice purposes.

---
