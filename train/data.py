import json

import numpy as np

from train.sbert import extract_embeddings, setup_sbert


def create_data_for_training(clean_overviews, movie_ids):
    model = setup_sbert()

    movie_embeddings = extract_embeddings(texts=clean_overviews, model=model)

    movie_embeddings_np = movie_embeddings.cpu().numpy()
    np.save('movie_embeddings.npy', movie_embeddings_np)

    movie_id_to_index = {int(id): idx for idx, id in enumerate(movie_ids)}
    with open('movie_id_to_index.json', 'w') as f:
        json.dump(movie_id_to_index, f)
