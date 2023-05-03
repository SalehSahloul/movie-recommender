#import libs
import pandas as pd
import numpy as np

def get_movie_titles_and_ratings_df():
    # reading user_movie data frame
    ratings_df = pd.read_csv('data/ratings.csv')

    # change it from long into wide format
    ratings_df = ratings_df.pivot(index='userId', columns='movieId', values='rating')

    # Drop movies (columns) with less than 50 ratings
    ratings_df = ratings_df.dropna(thresh=100, axis=1)

    # reading movie data frame
    movie_df = pd.read_csv('data/movies.csv')

    # get titles of movies to use them instead of the movie ids
    movie_titles = list()
    for id in list(ratings_df.columns):
        temp = list(movie_df[movie_df["movieId"]==id]["title"])
        movie_titles.append(temp[0])

    return movie_titles, ratings_df

def recommend_nmf(query, model, n=10):
    """
    Filters and recommends the top n movies for any given input query based on a trained NMF model. 
    Returns a list of n movie ids.

    Parameters:
    --------------
    query: user input as a dictionary
    model: trained NMF model
    n: number of top rated movies to recommend

    return:
    --------------
    recommended_movie_titles: a list with names of top n rated movies
    """
    movie_titles, ratings_df = get_movie_titles_and_ratings_df()

    # create a data frame of the user query (dictionary)
    user_input = pd.DataFrame(query, index = ['new_user'], columns=ratings_df.columns)

    # fill in the NaN values using the same imputation as training data
    # calculate the mean value of each column
    col_mean = ratings_df.mean()
    user_input_imputed = user_input.fillna(value=col_mean)

    # change the lables of the columns from movie ids into movie titles
    user_input_imputed.columns = movie_titles

    # get the Q Matrix (movie features matrix)
    Q = model.components_

    # get the P Matrix (user features matrix) and transform it into a data frame
    P_user = model.transform(user_input_imputed)
    P_user = pd.DataFrame(P_user, index = ['new_user'])

    # calculate the user-movie matrix and create a data frame 
    R_user = np.dot(P_user, Q)
    R_user = pd.DataFrame(R_user, index=['new_user'], columns=movie_titles)

    # sort it according to rank starting with best rated movies
    R_user_transposed = R_user.T.sort_values(by='new_user', ascending=False)

    # Get a list with sorted movie titles
    recommendables = list(R_user_transposed.index)

    # create a list of the movie titles in the user quiry dictionary 
    user_initial_ratings_list = list(query.keys())

    # filter out movies already seen by the user
    recommendations = list()
    recommendations = [movie for movie in recommendables if movie not in user_initial_ratings_list]

    # get titles of the top n recommended movies
    recommended_movie_titles = recommendations[:n]
    
    # return the top-k highest rated movie titles
    return recommended_movie_titles