import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_risk_classification_model(data_path, models_dir):
    """
    Trains a Random Forest classifier to predict Risk Levels from text TF-IDF features.
    Saves the trained model and vectorizer to disk.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Ensure columns exist
    if 'clean_text' not in df.columns:
        # Fallback if preprocessing wasn't run fully
        df['clean_text'] = df['text']
        
    df = df.dropna(subset=['clean_text', 'risk_level'])
    
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=1500, stop_words='english')
    X_text = vectorizer.fit_transform(df['clean_text'])
    y = df['risk_level']
    
    X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    print("Evaluating Model...")
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    
    print("Saving Models...")
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
        
    with open(os.path.join(models_dir, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(clf, f)
        
    print(f"Models saved successfully to {models_dir}.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    # Target processed data
    data_path = os.path.join(project_root, 'data', 'processed_social_posts.csv')
    
    if not os.path.exists(data_path):
        # Fallback to raw if processed missing for some reason
        data_path = os.path.join(project_root, 'data', 'raw_social_posts.csv')
        
    model_dir = os.path.join(project_root, 'models')
    
    if os.path.exists(data_path):
        train_risk_classification_model(data_path, model_dir)
    else:
        print(f"Error: Could not find training data at {data_path}. Please generate data first.")
