import numpy as np

def explain_prediction(text, vectorizer, model):
    """
    Given an input text, extracts the words that contributed the most to the model's prediction.
    It maps the non-zero TF-IDF features to the Random Forest's feature importances.
    Returns a sorted list of (word, importance_score).
    """
    # Clean and transform text
    vec = vectorizer.transform([text])
    
    # Get non-zero features in this specific text
    non_zero_idx = vec.nonzero()[1]
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    words_in_text = feature_names[non_zero_idx]
    
    # Cross reference with model feature importances
    importances = model.feature_importances_
    
    word_scores = []
    for idx, word in zip(non_zero_idx, words_in_text):
        word_scores.append((word, importances[idx]))
        
    # Sort by importance (highest first)
    word_scores.sort(key=lambda x: x[1], reverse=True)
    
    return word_scores

def highlight_text(text, important_words):
    """
    Simple utility to wrap important words in HTML highlight tags for Streamlit rendering.
    """
    highlighted_text = text
    # Extract just the words from the tuple
    for word, score in important_words[:5]: # Top 5 words
        # Use simple naive replacement for visual effect
        # HTML tag wrapping in Streamlit requires unsafe_allow_html=True
        replacement = f"<span style='background-color: rgba(255, 69, 0, 0.4); border-radius: 3px; padding: 0 2px;'>{word}</span>"
        
        # Simple case-insensitive replacement (careful with subwords, but fine for demo)
        import re
        highlighted_text = re.sub(rf'\b({word})\b', replacement, highlighted_text, flags=re.IGNORECASE)
        
    return highlighted_text
