### TF-IDF Implementation in Python
A from-scratch implementation of the Term Frequency–Inverse Document Frequency (TF-IDF) algorithm in Python 3.12. This project demonstrates how to calculate TF-IDF scores for documents without using any pre-built TF-IDF libraries, while supporting tokenization, term frequency calculation, and inverse document frequency computation.

### Prerequisites:
- Python (>=3.12)
- Poetry for dependency management.

### Install Dependencies
Ensure you have poetry with pip install poetry or similar. Then run:
python -m poetry install


### Usage
Either run with a default corpus:
python tfidf/tfidf.py

Or using a json file with list of documents:
python tfidf/tfidf.py --corpus path/to/corpus.json

If using the latter, each document should be a string inside a JSON array.
[
    "Document one text here.",
    "Document two text here."
]


### Output

The script outputs a CSV file with the following columns:
doc_index – the index of the document in the corpus
term – the tokenized term
tf_idf_score – the TF-IDF weight of the term
Terms in each document are sorted by descending TF-IDF score.


### Testing
Run the automated tests with:
pytest

### Linting
Check and auto-format code using Ruff:

ruff check .
ruff format .

You can configure line length and other rules in pyproject.toml.