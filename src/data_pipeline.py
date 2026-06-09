import pandas as pd
import re

def clean_text(text):
    """
    Standard text cleaning: lowercase, remove links, references, and special characters.
    """
    if not isinstance(text, str):
         return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text.strip()

def preprocess_data(df):
    """
    Simulates deduplication and text cleaning for the pipeline.
    """
    # Simulate deduplication
    df_clean = df.drop_duplicates(subset=['text']).copy()
    
    # Text cleaning
    df_clean['clean_text'] = df_clean['text'].apply(clean_text)
    
    return df_clean

if __name__ == "__main__":
    # Test pipeline on synthetic data
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    raw_data_path = os.path.join(project_root, 'data', 'raw_social_posts.csv')
    processed_data_path = os.path.join(project_root, 'data', 'processed_social_posts.csv')
    
    if os.path.exists(raw_data_path):
        df = pd.read_csv(raw_data_path)
        print(f"Loaded raw data: {df.shape[0]} rows.")
        df_processed = preprocess_data(df)
        df_processed.to_csv(processed_data_path, index=False)
        print(f"Processed and cleaned data. Final rows: {df_processed.shape[0]}. Saved to {processed_data_path}")
    else:
        print("Raw data not found. Please run data_generator.py first.")
