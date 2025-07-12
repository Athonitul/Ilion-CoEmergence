# ilion_anchor_filter_lite.py

"""
Ilion: TF-IDF based Semantic Filtering (Lite Version)

A simplified semantic anchor filter using TF-IDF and cosine similarity.
Requires only scikit-learn. No transformers or external model downloads.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tfidf_anchor_filter(tokens, anchors, threshold=0.3):
    """
    Filters tokens that are semantically similar to any anchor above threshold.
    Uses TF-IDF vectorization + cosine similarity.
    """
    all_texts = anchors + tokens
    vectorizer = TfidfVectorizer().fit(all_texts)
    vecs = vectorizer.transform(all_texts)

    anchor_vecs = vecs[:len(anchors)]
    token_vecs = vecs[len(anchors):]

    similarities = cosine_similarity(token_vecs, anchor_vecs)
    filtered = [tokens[i] for i, row in enumerate(similarities) if any(sim > threshold for sim in row)]

    return filtered

# Example usage
if __name__ == "__main__":
    tokens = ["Verticality", "drift", "Anchor", "resonance", "truth", "vital"]
    anchors = ["vertical", "truth"]
    result = tfidf_anchor_filter(tokens, anchors, threshold=0.3)
    print("Filtered Tokens:", result)
