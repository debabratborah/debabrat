import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the movies dataset
movies_data = pd.read_csv("movies.csv")

# Combine selected features into a single string
selected_features = ["genres", "keywords", "tagline", "cast", "director"]
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna("")
combined_features = movies_data["genres"] + " " + movies_data["keywords"] + " " + movies_data["tagline"] + " " + movies_data["cast"] + " " + movies_data["director"]

# Convert the combined features to a vector
vectorizer = TfidfVectorizer()
feature_vectorizer = vectorizer.fit_transform(combined_features)

# Compute the cosine similarity
similarity = cosine_similarity(feature_vectorizer)

@app.route('/')
def home():
    return render_template('index.HTML')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']
    list_of_all_titles = movies_data['title'].to_list()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if len(find_close_match) > 0:
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        recommendations = [movies_data['title'].iloc[i[0]] for i in sorted_similar_movies[1:11]]
        return render_template('index.HTML', recommendations=recommendations, movie_name=movie_name)
    else:
        return render_template('index.HTML', recommendations=[], movie_name="No movie found")

if __name__ == '__main__':
    app.run(debug=True)
