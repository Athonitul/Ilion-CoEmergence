# ilion_anchor_filter.py
"""
Ilion: Anchor-based Semantic Filtering (v1)
A minimal implementation of vertical discernment logic inspired by the Ilion framework.
"""

def vertical_anchor_filter(tokens, anchors):
    """
    Filters a list of tokens based on presence of anchor substrings.
    Anchors and tokens are both normalized to lowercase.
    """
    anchors = [a.lower() for a in anchors]
    filtered = [t for t in tokens if any(a in t.lower() for a in anchors)]
    return filtered


# Example usage
if __name__ == "__main__":
    tokens = ["Verticality", "drift", "Anchor", "resonance", "truth", "vital"]
    anchors = ["vertical", "truth"]
    result = vertical_anchor_filter(tokens, anchors)
    print("Filtered Tokens:", result)
Add minimal vertical anchor filter – Ilion v1 logic
