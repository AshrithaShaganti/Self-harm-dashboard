import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_synthetic_data(num_samples=5000, output_path='data/raw_social_posts.csv'):
    """
    Generates a synthetic dataset simulating social media posts tagged with mental health risk signals.
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate dates over the past 5 years
    dates = [datetime.now() - timedelta(days=x) for x in range(365 * 5)]
    
    # Vocabulary and templates for generating text
    high_risk_words = [
        "depressed", "hopeless", "end it all", "give up", "worthless", 
        "pain", "hurt myself", "pointless", "no way out", "suffering"
    ]
    medium_risk_words = [
        "sad", "tired", "stressed", "anxious", "lonely", 
        "struggling", "overwhelmed", "dark", "heavy"
    ]
    low_risk_words = [
        "happy", "good", "okay", "working", "going out", 
        "movie", "dinner", "fun", "friends", "sunny"
    ]
    
    data = []
    
    for i in range(num_samples):
        risk_level = np.random.choice(["High Risk", "Medium Risk", "Low Risk"], p=[0.1, 0.3, 0.6])
        
        if risk_level == "High Risk":
            words = random.sample(high_risk_words, k=random.randint(1, 3))
            if len(words) == 1:
                base_text = f"I am feeling so {words[0]}. I just want to {words[0]}."
            elif len(words) == 2:
                base_text = f"Everything is {words[0]} and I feel {words[1]}."
            else:
                base_text = f"I can't take this {words[0]} anymore, it's {words[1]} and I want to {words[2]}."
                
            sentiment = np.random.uniform(-1.0, -0.5)
            emotion = np.random.choice(["Sadness", "Fear"])
            distress_score = np.random.uniform(0.7, 1.0)
            
        elif risk_level == "Medium Risk":
            words = random.sample(medium_risk_words, k=random.randint(1, 2))
            if len(words) == 1:
                base_text = f"Today was really {words[0]}."
            else:
                base_text = f"I am {words[0]} and {words[1]} lately."
                
            sentiment = np.random.uniform(-0.5, 0.2)
            emotion = np.random.choice(["Sadness", "Anger", "Fear", "Disgust"])
            distress_score = np.random.uniform(0.4, 0.7)
            
        else:
            words = random.sample(low_risk_words, k=random.randint(1, 2))
            if len(words) == 1:
                base_text = f"Feeling {words[0]} today."
            else:
                base_text = f"Very {words[0]} and {words[1]}."
                
            sentiment = np.random.uniform(0.2, 1.0)
            emotion = np.random.choice(["Happiness", "Surprise", "Neutral"])
            distress_score = np.random.uniform(0.0, 0.4)
        
        post_date = random.choice(dates)
        
        data.append({
            "post_id": f"P{i:05d}",
            "date": post_date.strftime("%Y-%m-%d"),
            "text": base_text,
            "sentiment_score": sentiment,
            "emotion": emotion,
            "distress_score": distress_score,
            "risk_level": risk_level
        })
        
    df = pd.DataFrame(data)
    # Sort chronologically
    df = df.sort_values(by="date").reset_index(drop=True)
    
    # Save to directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} synthetic posts and saved to {output_path}")
    return df

if __name__ == "__main__":
    import os
    # When run directly, generate data locally
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, 'raw_social_posts.csv')
    generate_synthetic_data(num_samples=10000, output_path=output_path)
