
README: Financial Sentiment Classification

This project compares classical machine learning models and FinBERT for financial sentiment analysis using the Financial PhraseBank dataset.

==============================
1. Software Requirements
==============================

- Python 3.9+
- pip or conda (for dependency management)

Recommended environment managers:
- virtualenv
- conda

==============================
2. Required Libraries
==============================

Install all required packages via pip:

    pip install -r requirements.txt

Or install manually:

    pandas
    numpy
    matplotlib
    seaborn
    nltk
    spacy
    scikit-learn
    imbalanced-learn
    gensim
    torch
    transformers
    datasets
    wordcloud
    ipython

Also run:

    python -m spacy download en_core_web_sm

And in Python:

    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


==============================
3. Running the Code
==============================

- Open the Jupyter Notebook or script.
- Ensure `Sentences_AllAgree.txt is in the same directory as the Notebook
- Run all cells sequentially to:
    - Preprocess the data
    - Train classical models with TF-IDF, BoW, or Word2Vec
    - Fine-tune and evaluate FinBERT
    - Display evaluation metrics and generate visualizations


