import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import cast_scraper.main
from cast_scraper.main import read_csv
from cast_scraper.scraper import setup, get_people
from populate_db.main import DBPopulator
from rephrase_overviews.scraper import login, rephrase_overviews
from rephrase_overviews.main import read_csv as ro_read
from text_prep.preprocessor import apply_text_to_lower_and_strip_to_dataframe
from sklearn.metrics.pairwise import cosine_similarity
from new_movie_fetcher.fetch_ids_from_imdb import fetch_ids_from_imdb, remove_duplicates as r
from new_movie_fetcher.fetch_movie_data_from_tmdb_api import populate_new_movies_csv
from populate_db.add_similar_movies_together import DBPopulatorForSimilarMovies
from train.triplet_movie_pairs import TripletMoviePairDataset
from train.siamese_network import train
from eval.main import main as get_sim

from train.data import create_data_for_training

from signal import signal, SIGPIPE, SIG_DFL

signal(SIGPIPE, SIG_DFL)


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
    rows = ro_read()
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
    populate.import_csv('final_movie_data.csv')
    populate.close()


def get_skipped_movies():
    populate = DBPopulatorForSimilarMovies()
    populate.get_nodes_not_in_db()


def populate_db_with_similar():
    start_index, end_index = int(sys.argv[1]), int(sys.argv[2])
    populate = DBPopulatorForSimilarMovies()
    populate.create_indexes()
    np_embed = np.load('movie_embeddings.npy')
    df = pd.read_csv('final_movie_data.csv')
    movie_ids = df['id'].tolist()
    df_subset = df[start_index:end_index]

    # Initialize tqdm progress bar
    pbar = tqdm(total=len(df_subset), desc="Processing rows")

    # Lock for thread-safe operations on progress bar
    lock = Lock()

    def process_row(idx, row):
        print(f"Processing row {idx}")  # Debug statement
        query_index = idx
        similar_movie_ids, similar_movie_indices = get_sim(query_index, np_embed, movie_ids, 10)
        for sim in similar_movie_indices:
            print(f"Inserting data for similar movie index {sim}")  # Debug statement
            populate.insert_data(row, df.iloc[int(sim)])
        with lock:
            pbar.update(1)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_row, idx, row) for idx, row in df_subset.iterrows()]
        print(f"Submitted {len(futures)} tasks")  # Debug statement

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(e)

    populate.close()


def create_embeddings():
    df = pd.read_csv("final_movie_data.csv")
    apply_text_to_lower_and_strip_to_dataframe(df)

    clean_overviews = df['clean_overviews'].iloc[:].tolist()
    movie_ids = df['id'].iloc[:].tolist()

    create_data_for_training(clean_overviews, movie_ids)


def remove_duplicates():
    cast_scraper.main.remove_duplicates()


def fetch_new_movies():
    r()


def populate_new_movies_csv_main():
    with open('movie_ids_unique.txt', 'r') as f:
        movie_ids = f.readlines()
    for idx, movie_id in enumerate(movie_ids):
        movie_ids[idx] = movie_id.replace("\n", "")
    populate_new_movies_csv(movie_ids)


def remove_na():
    df = pd.read_csv("rephrased_overviews_movies_with_cast_unique.csv")
    df.replace('NA', None)
    df.to_csv('final_movie_data.csv', index=False)


def generate_lists_for_chunk(similarity_matrix_chunk, similarity_threshold, dissimilarity_threshold):
    similar_movies = {}
    dissimilar_movies = {}

    for i, sim_scores in enumerate(similarity_matrix_chunk):
        similar = np.where(sim_scores > similarity_threshold)[0].tolist()
        dissimilar = np.where(sim_scores < dissimilarity_threshold)[0].tolist()
        similar_movies[i] = similar
        dissimilar_movies[i] = dissimilar

    return similar_movies, dissimilar_movies


def weighted_random_choice(similar_movies, sim_scores):
    total = sum(sim_scores)
    r = random.uniform(0, total)
    upto = 0
    for movie, score in zip(similar_movies, sim_scores):
        if upto + score >= r:
            return movie
        upto += score
    return similar_movies[-1]  # Fallback in case of rounding errors


def create_triplets(similarity_matrix, start, num_triplets_per_movie=15, similarity_threshold=0.6,
                    dissimilarity_threshold=0.3):
    similar_movies, dissimilar_movies = {}, {}
    for i, sim_scores in tqdm(enumerate(similarity_matrix), total=similarity_matrix.shape[0], desc="Generating lists"):
        similar_indices = np.where(sim_scores > similarity_threshold)[0]
        dissimilar_indices = np.where(sim_scores < dissimilarity_threshold)[0]

        # Sort indices based on similarity scores
        similar_indices = similar_indices[np.argsort(-sim_scores[similar_indices])]
        dissimilar_indices = dissimilar_indices[np.argsort(sim_scores[dissimilar_indices])]

        similar_movies[i + start] = similar_indices.tolist()
        dissimilar_movies[i + start] = dissimilar_indices.tolist()

    triplets = []

    for idx, anchor in tqdm(enumerate(similar_movies), total=len(similar_movies), desc="Generating triplets"):
        count = 0
        max_attempts = 100

        while count < num_triplets_per_movie and max_attempts > 0:
            if len(similar_movies[anchor]) > 0 and len(dissimilar_movies[anchor]) > 0:
                # Assign weights based on index (higher index, lower weight)
                similar_weights = np.linspace(1, 0.1, len(similar_movies[anchor]))
                dissimilar_weights = np.linspace(1, 0.1, len(dissimilar_movies[anchor]))

                # Weighted random choice for positive and negative examples
                positive = weighted_random_choice(similar_movies[anchor], similar_weights)
                negative = weighted_random_choice(dissimilar_movies[anchor], dissimilar_weights)

                if anchor != positive:
                    count += 1
                    triplets.append((anchor, positive, negative))
            max_attempts -= 1

    return triplets


def process_and_save_triplets(np_embed, similarity_matrix):
    triplets = []

    print("Creating triplets...")
    # Process the similarity matrix in batches of 1000
    batch_size = 1000
    num_batches = (similarity_matrix.shape[0] + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Processing batches..."):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, similarity_matrix.shape[0])
        triplets += create_triplets(similarity_matrix[start_idx:end_idx], start=start_idx)
    print("Finished creating triplets...")

    with open("triplets.txt", "w") as f:
        for triplet in triplets:
            f.write(str(triplet[0]) + " " + str(triplet[1]) + " " + str(triplet[2]) + " " + "\n")


def start_training(np_embed, similarity_matrix):
    print("Loading triplets...")
    with open('triplets.txt', 'r') as f:
        temp_triplets = f.readlines()

    triplets = []

    for triplet in temp_triplets:
        temp = triplet.split(" ")
        triplets.append((int(temp[0]) - 1, int(temp[1]) - 1, int(temp[2]) - 1))

    triplet_dataset = TripletMoviePairDataset(np_embed, triplets, similarity_matrix)
    data_loader = DataLoader(triplet_dataset, batch_size=32, shuffle=True)
    print("Finished loading triplets...")
    print("Starting training...")
    train(data_loader, np_embed.shape[1], 20)


def main():
    movie_id = sys.argv[1]
    np_embed = np.load('movie_embeddings.npy')

    similarity_matrix = cosine_similarity(np_embed)

    # create_triplets(np_embed, similarity_matrix)
    # start_training(np_embed, similarity_matrix)

    with open('movie_id_to_index.json', 'r') as f:
        movie_id_to_index = json.load(f)

    query_index = movie_id_to_index[str(movie_id)]

    df = pd.read_csv('final_movie_data.csv')
    movie_ids = df['id'].tolist()

    get_sim(query_index, np_embed, movie_ids, 20)


if __name__ == "__main__":
    main()
