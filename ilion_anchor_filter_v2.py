# ilion_anchor_filter_v2.py

"""
Ilion: Semantic Anchor Filtering (v2)
Now using cosine similarity over sentence embeddings.
Inspired by the Ilion framework's vertical discernment layer.
"""

from sentence_transformers import SentenceTransformer, util
import torch

def vertical_anchor_filter_v2(tokens, anchors, model_name='all-MiniLM-L6-v2', threshold=0.5):
    """
    Filters a list of tokens based on semantic similarity to given anchors using cosine similarity.
    Parameters:
        tokens (list): List of strings to filter.
        anchors (list): Anchor strings to match semantically.
        model_name (str): SentenceTransformer model (default: MiniLM-L6-v2).
        threshold (float): Cosine similarity threshold.
    Returns:
        list: Filtered tokens with similarity ≥ threshold.
    """
    model = SentenceTransformer(model_name)
    token_embs = model.encode(tokens, convert_to_tensor=True)
    anchor_embs = model.encode(anchors, convert_to_tensor=True)

    results = []
    for i, token_emb in enumerate(token_embs):
        max_sim = torch.max(util.cos_sim(token_emb, anchor_embs)).item()
        if max_sim >= threshold:
            results.append((tokens[i], round(max_sim, 4)))
    return results

# Example usage
if __name__ == "__main__":
    tokens = ["Verticality", "drift", "Anchor", "resonance", "truth", "vital"]
    anchors = ["vertical", "truth"]
    filtered = vertical_anchor_filter_v2(tokens, anchors)
    print("Filtered Tokens:", filtered)
