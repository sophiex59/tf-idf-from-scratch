# TF-IDF from scratch

[![CI](https://github.com/sophiex59/tf-idf-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/sophiex59/tf-idf-from-scratch/actions/workflows/ci.yml)

A Python implementation of Term Frequency–Inverse Document Frequency without a pre-built TF-IDF library.

## Usage

Install dependencies and generate `tfidf.csv` from the bundled sample corpus:

```sh
poetry install
poetry run python tfidf/tfidf.py
```

To process a JSON file containing a list of document strings:

```sh
poetry run python tfidf/tfidf.py --corpus path/to/corpus.json
```

For example, the input JSON can be:

```json
[
  "Document one text here.",
  "Document two text here."
]
```

## Output

The CSV contains `doc_index`, `term`, and `tf_idf_score`; terms in each document are sorted by descending TF-IDF score.

## Testing

```sh
poetry run pytest
```

## Linting

```sh
poetry run ruff check .
poetry run ruff format .
```
