from textblob import TextBlob

high_risk_keywords = [
    "depressed", "hopeless", "end it all", "give up", "worthless", 
    "pain", "hurt myself", "pointless", "no way out", "suffering",
    "kill myself", "suicide", "can't go on", "die"
]

def analyze_text(text):
    """
    Analyzes the text for sentiment, emotion, and extracts risk-oriented keywords.
    Returns a dictionary of extracted features.
    """
    if not isinstance(text, str):
        return {"sentiment": 0.0, "risk_keywords": [], "emotion": "Neutral"}
        
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    
    text_lower = text.lower()
    
    # Find overlapping keywords
    found_keywords = [kw for kw in high_risk_keywords if kw in text_lower]
    
    # Simple proxies for emotion based on polarity and keyword presence
    if sentiment < -0.5 and len(found_keywords) > 0:
        emotion = "Fear/Sadness"
    elif sentiment < -0.2:
        emotion = "Anger/Disgust"
    elif sentiment > 0.4:
        emotion = "Happiness"
    else:
        emotion = "Neutral"
        
    return {
        "sentiment": sentiment,
        "risk_keywords": found_keywords,
        "emotion": emotion
    }
