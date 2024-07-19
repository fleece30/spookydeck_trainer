import json

import numpy as np
import pandas as pd

from cast_scraper.main import read_csv
from cast_scraper.scraper import setup, get_people
from populate_db.main import DBPopulator
from rephrase_overviews.scraper import login, rephrase_overviews
from text_prep.preprocessor import apply_text_to_lower_and_strip_to_dataframe
from sklearn.metrics.pairwise import cosine_similarity

from train.data import create_data_for_training

import sys


def predict(movie_id_1, movie_id_2):
    np_embed = np.load('movie_embeddings.npy')
    with open('movie_id_to_index.json', 'r') as f:
        movie_id_to_index = json.load(f)

    embedding_1 = np_embed[movie_id_to_index[str(movie_id_1)]]
    embedding_2 = np_embed[movie_id_to_index[str(movie_id_2)]]

    embedding_1 = embedding_1.reshape(1, -1)
    embedding_2 = embedding_2.reshape(1, -1)
    similarity = cosine_similarity(embedding_1, embedding_2)
    print(similarity[0][0])


def rephrase_overviews_main():
    starting_index, ending_index, output_file_name_suffix = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rows = read_csv()
    driver = setup()
    login(driver)
    rephrase_overviews(driver, rows[starting_index:ending_index], sys.argv[3])


def run_cast_scraper():
    starting_index, ending_index, output_file_name_suffix = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rows = read_csv()
    driver = setup()
    get_people(driver, rows[starting_index:ending_index], sys.argv[3])


def populate_db():
    populate = DBPopulator()
    populate.create_indexes()
    populate.import_csv('../rephrased_overviews_movies_with_cast.csv')
    populate.close()


def create_embeddings():
    df = pd.read_csv("rephrased_overviews_movies_with_cast.csv")
    apply_text_to_lower_and_strip_to_dataframe(df)

    clean_overviews = df['clean_overviews'].iloc[:].tolist()
    movie_ids = df['id'].iloc[:].tolist()

    create_data_for_training(clean_overviews, movie_ids)


def main():
    create_embeddings()


if __name__ == "__main__":
    main()
