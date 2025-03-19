import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import streamlit as st

# Sample dataset for recommendations (food, travel, wellness)
data = {
    "category": ["food", "food", "travel", "travel", "wellness", "wellness"],
    "name": ["Sushi Delight", "Vegan Paradise", "Beach Getaway", "Mountain Trek", "Spa Retreat", "Yoga Studio"],
    "description": [
        "Fresh sushi with authentic flavors",
        "Healthy and delicious plant-based meals",
        "Relax at a beautiful sunny beach",
        "Hike through scenic mountains",
        "Full-body relaxation and massage services",
        "Experience deep meditation and flexibility training"
    ]
}
df = pd.DataFrame(data)

# Function to recommend items based on user input
def recommend(category, user_input):
    category_df = df[df["category"] == category]
    if category_df.empty:
        return "No recommendations found."
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(category_df["description"])
    
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()
    
    top_indices = similarity.argsort()[-3:][::-1]
    return category_df.iloc[top_indices]["name"].tolist(), similarity[top_indices]

# Sentiment Analysis for refining recommendations
def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return "positive"
    elif sentiment < 0:
        return "negative"
    else:
        return "neutral"

# Performance Evaluation Function
def evaluate_performance():
    categories = df["category"].unique()
    accuracy_scores = []
    
    for category in categories:
        sample_input = df[df["category"] == category].iloc[0]["description"]
        _, similarities = recommend(category, sample_input)
        avg_similarity = np.mean(similarities) * 100
        accuracy_scores.append(avg_similarity)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=categories, y=accuracy_scores, palette="viridis")
    plt.xlabel("Category")
    plt.ylabel("Average Similarity Score (%)")
    plt.title("Recommendation System Performance Evaluation")
    st.pyplot(plt)

# Streamlit App UI
st.title("Smart Recommendation System")
category = st.selectbox("Choose a category:", ["food", "travel", "wellness"])
user_input = st.text_area("Describe what you're looking for:")

if st.button("Get Recommendations"):
    recommendations, scores = recommend(category, user_input)
    sentiment = analyze_sentiment(user_input)
    st.write(f"Your mood seems {sentiment}.")
    st.write("Top Recommendations:")
    for rec, score in zip(recommendations, scores):
        st.write(f"- {rec} (Similarity: {score:.2f})")
    
    st.write("\n**Performance Evaluation:**")
    evaluate_performance()
