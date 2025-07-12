
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the embedding model (MiniLM or similar)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_anchors(text):
    # Auto-extracts anchors as meaningful sentences ending in punctuation
    sentences = re.findall(r'[^.!?]+[.!?]', text)
    anchors = [s.strip() for s in sentences if len(s.split()) > 4]
    return anchors

def compute_tfidf_corpus_matrix(corpus):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def inject_corpus_filter(target_text, anchor_texts, corpus_texts, sim_threshold=0.7):
    # Generate embeddings for semantic matching
    anchor_embeddings = model.encode(anchor_texts)
    target_embedding = model.encode([target_text])[0]

    # Calculate cosine similarity between target and anchors
    similarities = cosine_similarity([target_embedding], anchor_embeddings)[0]

    # Filter anchors by semantic threshold
    semantic_matches = [anchor_texts[i] for i, score in enumerate(similarities) if score >= sim_threshold]

    # TF-IDF corpus relevance
    tfidf_matrix, vectorizer = compute_tfidf_corpus_matrix(corpus_texts + [target_text])
    cosine_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    corpus_matches = [corpus_texts[i] for i, score in enumerate(cosine_scores) if score >= 0.3]

    return {
        "semantic_matches": semantic_matches,
        "corpus_matches": corpus_matches
    }

# Example usage
if __name__ == "__main__":
    sample_text = "This is a test. This method aligns context semantically. It also works with corpus injection."
    corpus = ["Contextual awareness is key.", "Semantic alignment ensures coherence.", "Injection of corpus improves adaptability."]

    anchors = extract_anchors(sample_text)
    print("Extracted Anchors:", anchors)

    results = inject_corpus_filter("Ensure semantic coherence with context.", anchors, corpus)
    print("Semantic Matches:", results["semantic_matches"])
    print("Corpus Matches:", results["corpus_matches"])
