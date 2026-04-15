import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import difflib
import os

# --- CONFIG: path to your CSV (adjust if needed) ---
DATA_PATH = "./mnt/data/netflix_data.csv"

# Cache the data loading and model building for performance
@st.cache_data
def load_data_and_model():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}. Please check path.")
        st.stop()

    netflix_data = pd.read_csv(DATA_PATH)

    # Ensure 'title' and 'description' columns exist
    if 'title' not in netflix_data.columns:
        st.error("'title' column not found in the CSV.")
        st.stop()
    if 'description' not in netflix_data.columns:
        netflix_data['description'] = ''

    # Fill missing values
    for col in ['title', 'cast', 'listed_in', 'description']:
        if col not in netflix_data.columns:
            netflix_data[col] = ''
        netflix_data[col] = netflix_data[col].fillna('')

    # Create combined features
    netflix_data['combined_features'] = (
        netflix_data['title'] + ' ' +
        netflix_data['cast'] + ' ' +
        netflix_data['listed_in'] + ' ' +
        netflix_data['description']
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(netflix_data['combined_features'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Map titles to indices
    indexes = pd.Series(netflix_data.index, index=netflix_data['title']).drop_duplicates()

    return netflix_data, cosine_sim, indexes

# Helper functions
def get_closest_title(query_title, n_matches=1, cutoff=0.5):
    titles = netflix_data['title'].astype(str).tolist()
    matches = difflib.get_close_matches(query_title, titles, n=n_matches, cutoff=cutoff)
    return matches[0] if matches else None

def get_recommendations(movie_title, top_n=10):
    if movie_title not in indexes:
        close = get_closest_title(movie_title, cutoff=0.45)
        if close:
            movie_title = close
            st.info(f"No exact match. Showing results for '{close}' (closest match).")
        else:
            st.error("Movie not found. Try a different title or check spelling.")
            return []

    movie_idx = indexes[movie_title]
    similarity_scores = list(enumerate(cosine_sim[movie_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda pair: pair[1], reverse=True)
    top_scores = similarity_scores[1: top_n + 1]
    recommended_indices = [pair[0] for pair in top_scores]
    return netflix_data['title'].iloc[recommended_indices].tolist()

# Load data
netflix_data, cosine_sim, indexes = load_data_and_model()

# Streamlit UI
st.title("Netflix Movie Recommendation System")
st.write("Enter a movie title to get recommendations based on content similarity.")

movie_input = st.text_input("Enter a movie title:", "")

if st.button("Get Recommendations"):
    if movie_input.strip():
        recommendations = get_recommendations(movie_input.strip(), top_n=10)
        if recommendations:
            st.success(f"Recommendations for '{movie_input}':")
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.error("No recommendations found.")
    else:
        st.warning("Please enter a movie title.")

st.write("---")
st.write("Data loaded from Netflix dataset. Recommendations use TF-IDF and cosine similarity.")