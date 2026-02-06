from typing import List, Dict
from math import log
import pandas as pd
from collections import Counter
import argparse
import json
import re


def tokenise_doc(text: str) -> List[str]:
    """
    Breaks a string into a list of individual words, lowered, and stripped of punctuation.
    However, keeps contractions like didn't.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    return tokens


def compute_term_frequency(tokens: List[str]) -> Dict[str, float]:
    """
    Number of times a term appears in a document, divded by total number of terms, normalised.
    """
    total_words = len(tokens)

    # If there is an empty doc
    if total_words == 0:
        return {}
    term_counts = Counter(tokens)
    tf = {}
    for term, count in term_counts.items():
        tf[term] = count / total_words

    return tf


def compute_inverse_doc_frequency(
    tokenised_corpus: List[List[str]], doc_count: int
) -> Dict[str, float]:
    """
    Calculated as logarithmically scaled inverse fraction of the total number of documents that contain the
    term.
    """
    doc_frequencies = {}
    for doc in tokenised_corpus:
        unique_terms = set(doc)
        for term in unique_terms:
            doc_frequencies[term] = doc_frequencies.get(term, 0) + 1
    idf = {}
    for term, freq in doc_frequencies.items():
        idf[term] = log(doc_count / freq)
    return idf


def compute_tf_idf(tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
    """
    Per term in corpus:
    TF-IDF(term) = TF(term) * IDF(term
    """
    tfidf = {}
    for term, tf_value in tf.items():
        tfidf_value = tf_value * idf.get(term, 0)
        tfidf[term] = tfidf_value
    return tfidf


def build_tfidf(corpus: List[str]) -> pd.DataFrame:
    """
    Pipeline from list of raw text documents to dataframe of tf-idfs.
    """
    doc_count = len(corpus)
    tokenised_corpus = [tokenise_doc(doc) for doc in corpus]

    # Counts in how many docs each unique term appears.
    idf = compute_inverse_doc_frequency(tokenised_corpus, doc_count)

    df_rows = []
    for index, doc in enumerate(tokenised_corpus):
        # Counts frequency of term appearing in a doc.
        tf = compute_term_frequency(doc)

        # Multiplies TF and IDF per term to get final weights.
        tfidf = compute_tf_idf(tf, idf)

        for term, score in tfidf.items():
            # Filter 0 score terms to keep the dataframe sparse.
            if score > 0:
                df_rows.append((index, term, score))

    # Creates a dataframe, then sorting terms from highest to lowest within each document
    df = pd.DataFrame(df_rows, columns=["doc_index", "term", "tf_idf_score"])
    df = df.sort_values(by=["doc_index", "tf_idf_score"], ascending=[True, False]).reset_index(
        drop=True
    )

    return df


if __name__ == "__main__":
    corpus = [
        """Once upon a time, in a faraway land, there was a brave knight named Sir Lancelot. He was known for his
    strength, courage, and chivalry. One day, the King of the land asked Sir Lancelot to rescue his daughter, who
    had been kidnapped by an evil sorcerer. Sir Lancelot set out on his quest, facing many dangers along the way.
    But he never gave up, and eventually he rescued the princess and defeated the sorcerer. The people of the land
    cheered for Sir Lancelot, and he became a legend in his own time.""",
        """In a village at the foot of a mountain, there lived a poor farmer named Jack. He had a small farm and a cow,
    which was his only possession. One day, the cow stopped giving milk, and Jack didn't know what to do. So he
    decided to sell the cow at the market. On the way, he met a stranger who offered to trade five magic beans for
    the cow. Jack agreed, and when he got home, his mother was furious. But that night, the magic beans grew
    into a giant beanstalk, and Jack climbed it to find a castle in the clouds. There, he met a giant who had a goose
    that laid golden eggs. Jack stole the goose and ran down the beanstalk, but the giant followed him. Jack
    chopped down the beanstalk, and the giant fell to his death. Jack and his mother lived happily ever after with
    the golden eggs.""",
        """In a kingdom ruled by a wicked queen, there lived a beautiful princess named Snow White. The queen was
    jealous of Snow White's beauty, and ordered a huntsman to kill her. But the huntsman couldn't do it, so he left
    Snow White in the forest. There, she met seven dwarfs who took her in and cared for her. But the queen found
    out that Snow White was still alive, and disguised herself as an old woman to give Snow White a poisoned
    apple. Snow White fell into a deep sleep, but a prince came and woke her with a kiss. They lived happily ever
    after, and the queen got what she deserved.""",
        """In a world of magic and wonder, there was a young wizard named Harry Potter. He had been orphaned as a
    baby, and was raised by his cruel relatives. But one day, he received a letter from Hogwarts School of
    Witchcraft and Wizardry, inviting him to attend. There, he learned about his true heritage and his destiny to
    defeat the dark wizard Voldemort. Harry made many friends at Hogwarts, including Hermione Granger and
    Ron Weasley. Together, they faced many challenges and battles, but in the end, Harry was able to vanquish
    Voldemort and bring peace to the wizarding world.""",
        """In a land of dragons and knights, there was a beautiful princess named Fiona. She had been cursed by a
    wicked sorcerer and turned into an ogre. One day, a brave knight named Shrek was sent to rescue her from a
    tower. But when he found her, he discovered that she was an ogre. They didn't get along at first, but eventually
    they fell in love. Along the way, they met many fairy tale characters, including a talking donkey and a
    gingerbread man. Together, they defeated the evil Lord Farquaad and lived happily ever after in the swamp.""",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=str,
        required=False,
        help="Path to corpus JSON file. If not provided, default corpus list will be used.",
    )
    args = parser.parse_args()
    if args.corpus:
        with open(args.corpus, "r", encoding="utf-8") as f:
            corpus = json.load(f)

    df = build_tfidf(corpus)

    file_name = "tfidf.csv"
    df.to_csv(file_name, index=False)
    print(f"Saved the result to {file_name}")

    print("First 20 rows of your results: ")
    print(df.head(20))
