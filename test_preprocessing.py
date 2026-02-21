"""
Unit tests for the text preprocessing pipeline.
Run with: pytest test_preprocessing.py -v
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """Preprocess text for sentiment analysis."""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)


def test_lowercase_conversion():
    result = preprocess_text("GREAT Movie Was AMAZING")
    assert result == result.lower(), "Output should be fully lowercase"


def test_html_tag_removal():
    result = preprocess_text("<b>good</b> <i>movie</i> loved it")
    assert "<" not in result and ">" not in result, "HTML tags should be removed"
    assert "good" in result
    assert "movie" in result


def test_url_removal():
    result = preprocess_text("Check http://example.com this great movie")
    assert "http" not in result
    assert "example" not in result


def test_stop_words_removed():
    result = preprocess_text("this is a very good movie and I liked it")
    words = result.split()
    common_stops = {"this", "is", "a", "very", "and", "it"}
    for w in words:
        assert w not in common_stops, f"Stop word '{w}' should have been removed"


def test_short_words_filtered():
    result = preprocess_text("I am so ok but the movie was fantastic")
    words = result.split()
    for w in words:
        assert len(w) > 2, f"Word '{w}' is too short, should be filtered"


def test_punctuation_removal():
    result = preprocess_text("Wow!!! This movie... is great??? 10/10")
    assert all(c.isalpha() or c.isspace() for c in result), "No punctuation should remain"


def test_empty_after_cleaning():
    result = preprocess_text("the is a an")
    # all stop words and/or short words, should return empty or near-empty
    assert all(len(w) > 2 for w in result.split() if w)


def test_real_review_positive():
    review = "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."
    result = preprocess_text(review)
    assert "movie" in result
    assert "fantastic" in result or "superb" in result
    assert len(result) > 0


def test_real_review_negative():
    review = "Terrible film. I was bored the entire time and the ending made no sense whatsoever."
    result = preprocess_text(review)
    assert "terrible" in result or "bored" in result
    assert len(result) > 0


if __name__ == "__main__":
    test_lowercase_conversion()
    test_html_tag_removal()
    test_url_removal()
    test_stop_words_removed()
    test_short_words_filtered()
    test_punctuation_removal()
    test_empty_after_cleaning()
    test_real_review_positive()
    test_real_review_negative()
    print("All 9 tests passed.")
