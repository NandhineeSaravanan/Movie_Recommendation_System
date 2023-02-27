# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 22:25:44 2023

@author: sneka
"""

# import library
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


# Define the FastAPI app
app = FastAPI()
pickle_in = open("model.pkl","rb")
model=pickle.load(pickle_in)


# Load the movies dataset
movies_df = pd.read_csv('movie.csv')


# Create a TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')



# Fit the vectorizer on the movie titles
tfidf.fit(movies_df['title'])


# Transform the movie titles into vectors
title_vectors = tfidf.transform(movies_df['title'])

# Create a KNN model with cosine similarity
model = NearestNeighbors(metric='cosine')
model.fit(title_vectors)

# Create a function that takes in a movie title and returns recommended movies
def get_recommendations(movie_title, num_recommendations=10):
    # Transform the movie title into a vector
    title_vector = tfidf.transform([movie_title])

    # Get the indices of the most similar movies
    distances, indices = model.kneighbors(title_vector, n_neighbors=num_recommendations+1)

    # Get the titles of the recommended movies
    recommended_movies = []
    for i in range(1, len(distances.flatten())):
        recommended_movies.append(movies_df['title'][indices.flatten()[i]])

    return recommended_movies

# Define the input data model
class MovieTitle(BaseModel):
    title: str

# Define the API endpoint for the movie recommendation system
@app.post('/recommendations')
def get_movie_recommendations(movie_title: MovieTitle):
    recommendations = get_recommendations(movie_title.title)
    return {'recommended_movies': recommendations}