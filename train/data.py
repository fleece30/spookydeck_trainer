import numpy as np

from train.bert import extract_embeddings, setup_sbert


def create_data_for_training(clean_overviews):
    model = setup_sbert()

    movie_embeddings = extract_embeddings(texts=clean_overviews, model=model)

    movie_embeddings_np = movie_embeddings.cpu().numpy()
    np.save('movie_embeddings.npy', movie_embeddings_np)
