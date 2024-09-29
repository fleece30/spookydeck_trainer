import torch
import torch.nn.functional as F
from train.siamese_network import SiameseNetwork


def get_movie_embedding(model, embedding):
    with torch.no_grad():
        latent_representation = model(torch.tensor(embedding).unsqueeze(0))
    return latent_representation


def compute_similarities(model, query_embedding, embeddings):
    query_latent = get_movie_embedding(model, query_embedding)
    # print("query latent: ", query_latent)
    all_latents = []
    with torch.no_grad():
        for emb in embeddings:
            latent = model(torch.tensor(emb).unsqueeze(0).float())
            all_latents.append(latent)

    all_latents = torch.stack(all_latents).squeeze(1)
    # print(all_latents.shape)
    similarities = F.cosine_similarity(query_latent, all_latents)
    return similarities


def get_similar_movies(model, query_index, embeddings, movie_ids, top_n=20):
    query_embedding = embeddings[query_index]
    similarities = compute_similarities(model, query_embedding, embeddings)
    # print(similarities)
    # Get the indices of the top N similar movies
    top_n_indices = torch.topk(similarities, top_n + 1).indices  # top_n + 1 to exclude the movie itself
    top_n_indices = top_n_indices[top_n_indices != query_index]  # Remove the query movie itself
    # print(top_n_indices)
    top_n_movie_ids = [movie_ids[i] for i in top_n_indices[:top_n]]

    return top_n_movie_ids, top_n_indices


def main(query_index, embeddings, movie_ids, top_n=20):
    model = SiameseNetwork(embeddings.shape[1])
    model.load_state_dict(torch.load('./siamese_network.pt'))
    model.eval()
    # print(movie_ids)
    similar_movie_ids, top_n_indices = get_similar_movies(model, query_index, embeddings, movie_ids, top_n)
    # print(f"Top {top_n} movies similar to movie ID {movie_ids[query_index]}: {similar_movie_ids}")
    return similar_movie_ids, top_n_indices
