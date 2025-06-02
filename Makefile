# Makefile for TextProcessor project

VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
STREAMLIT := $(VENV_DIR)/bin/streamlit

.PHONY: all venv run models clean

all: venv

venv:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(STREAMLIT) run text_processor_app.py

models:
	$(PYTHON) -m spacy download en_core_web_sm
	$(PYTHON) -m spacy download fr_core_news_sm

clean:
	rm -rf $(VENV_DIR)
	rm -rf __pycache__ .streamlit *.csv uploaded_text.txt