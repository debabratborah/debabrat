import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies_data = pd.read_csv("movies.csv")


selected_features = ["genres", "keywords", "tagline", "cast", "director"]
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna("")
combined_features = movies_data["genres"] + " " + movies_data["keywords"] + " " + movies_data["tagline"] + " " + movies_data["cast"] + " " + movies_data["director"]


vectorizer = TfidfVectorizer()
feature_vectorizer = vectorizer.fit_transform(combined_features)


similarity = cosine_similarity(feature_vectorizer)


st.title("Movie Recommendation System")

movie_name = st.text_input("Enter a Movie Name")

if st.button("Recommend"):
    list_of_all_titles = movies_data['title'].to_list()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if find_close_match:
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        recommendations = [movies_data['title'].iloc[i[0]] for i in sorted_similar_movies[1:11]]
        st.write(f"Recommendations for {movie_name}:")
        for movie in recommendations:
            st.write(movie)
    else:
        st.write("No recommendations found")
