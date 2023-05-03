from flask import Flask, render_template, request
from recommender import recommend_nmf, get_movie_titles_and_ratings_df
import pickle

# Load the Model
with open('factorizer_NMF.pkl', 'rb') as file_in:
    fitted_model = pickle.load(file_in)

app = Flask(__name__)

@app.route("/")
def homepage():
    movie_titles, _ = get_movie_titles_and_ratings_df()
    return render_template("home.html", movies=movie_titles)

@app.route("/results")
def recommender():
    all_inputs = request.args.to_dict()
    user_query = {key:int(value) for key, value in all_inputs.items() if key.startswith('movie-')}
    n = int(all_inputs.get('n')) # get the value of n (number of recommendations)
    top_n_movies = recommend_nmf(user_query, fitted_model, n)
    return render_template("results.html", movies = top_n_movies)


if __name__ == "__main__":
    app.run(port=5000, debug=True)