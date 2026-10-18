"""Sparse, reproducible recommendation models built from the bundled MovieLens data."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path(__file__).resolve().parent / 'dataset'


def load_catalog():
    movies = pd.read_csv(DATA / 'movies.csv')
    links = pd.read_csv(DATA / 'links.csv')
    movies = movies.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
    movies['tmdbId'] = movies['tmdbId'].astype('Int64')
    return movies


def content_model(movies):
    return TfidfVectorizer(tokenizer=lambda value: value.split('|'),
                           token_pattern=None, lowercase=False).fit_transform(movies['genres'])


def content_recommendations(movies, matrix, title, limit=5):
    matches = np.flatnonzero(movies['title'].eq(title))
    if not len(matches):
        return movies.iloc[:0]
    index = matches[0]
    scores = cosine_similarity(matrix[index], matrix).ravel()
    order = np.argsort(-scores, kind='stable')
    order = order[order != index][:limit]
    return movies.iloc[order]


def collaborative_model():
    ratings = pd.read_csv(DATA / 'ratings.csv')
    users = np.sort(ratings.userId.unique())
    movie_ids = np.sort(ratings.movieId.unique())
    matrix = csr_matrix((ratings.rating, (np.searchsorted(users, ratings.userId),
                                         np.searchsorted(movie_ids, ratings.movieId))),
                        shape=(len(users), len(movie_ids)))
    return users, movie_ids, matrix


def collaborative_recommendations(movies, model, user_id, limit=5):
    users, movie_ids, matrix = model
    indexes = np.flatnonzero(users == user_id)
    if not len(indexes):
        return movies.iloc[:0]
    index = indexes[0]
    weights = cosine_similarity(matrix[index], matrix).ravel()
    weights[index] = 0
    neighbors = np.argsort(-weights)[:30]
    selected = matrix[neighbors]
    numerator = np.asarray(selected.T @ weights[neighbors]).ravel()
    denominator = np.asarray((selected != 0).T @ weights[neighbors]).ravel()
    scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    scores[matrix[index].indices] = -1
    order = np.argsort(-scores, kind='stable')
    ids = movie_ids[order[scores[order] > 0]][:limit]
    return movies.set_index('movieId').reindex(ids).dropna(subset=['title']).reset_index()
