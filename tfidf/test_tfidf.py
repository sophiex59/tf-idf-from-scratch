import math

from tfidf import (
    tokenise_doc,
    compute_term_frequency,
    compute_inverse_doc_frequency,
    compute_tf_idf,
    build_tfidf,
)


# -------------------------
# Tokenisation tests
# -------------------------
def test_tokenise_basic():
    text = "Hello, World!"
    assert tokenise_doc(text) == ["hello", "world"]


def test_tokenise_contractions():
    text = "I didn't know"
    assert tokenise_doc(text) == ["i", "didn't", "know"]


def test_tokenise_empty():
    assert tokenise_doc("") == []


# -------------------------
# Term Frequency tests
# -------------------------
def test_tf_basic():
    tokens = ["a", "b", "a"]
    tf = compute_term_frequency(tokens)

    assert tf["a"] == 2 / 3
    assert tf["b"] == 1 / 3


def test_tf_empty():
    assert compute_term_frequency([]) == {}


# -------------------------
# Inverse Document Frequency tests
# -------------------------
def test_idf_basic():
    corpus = [
        ["a", "b"],
        ["a"],
        ["b"],
    ]
    idf = compute_inverse_doc_frequency(corpus, 3)

    assert idf["a"] > 0
    assert idf["b"] > 0


def test_idf_exact_math():
    corpus = [["x"], ["x"], ["y"]]
    idf = compute_inverse_doc_frequency(corpus, 3)

    # x appears in 2/3 docs
    assert math.isclose(idf["x"], math.log(3 / 2))
    # y appears in 1/3 docs
    assert math.isclose(idf["y"], math.log(3 / 1))


# -------------------------
# TF-IDF tests
# -------------------------
def test_tfidf_basic():
    tf = {"a": 0.5, "b": 0.25}
    idf = {"a": 2.0, "b": 4.0}

    tfidf = compute_tf_idf(tf, idf)

    assert tfidf["a"] == 1.0
    assert tfidf["b"] == 1.0


# -------------------------
# Full pipeline tests
# -------------------------


def test_build_tfidf_structure():
    corpus = [
        "apple banana",
        "banana",
    ]

    df = build_tfidf(corpus)

    # Expected columns
    assert list(df.columns) == ["doc_index", "term", "tf_idf_score"]

    # Should not be empty
    assert len(df) > 0


def test_build_tfidf_sorted_within_doc():
    corpus = [
        "apple apple banana",
        "banana",
    ]

    df = build_tfidf(corpus)

    # For doc 0, first term should have >= score than subsequent ones
    doc0 = df[df["doc_index"] == 0]["tf_idf_score"].tolist()
    assert doc0 == sorted(doc0, reverse=True)
