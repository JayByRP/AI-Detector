#!/bin/bash
echo "Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm