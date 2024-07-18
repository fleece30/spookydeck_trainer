import numpy as np
import pandas as pd
from text_prep.preprocessor import apply_text_to_lower_and_strip_to_dataframe
from sklearn.metrics.pairwise import cosine_similarity

from train.data import create_data_for_training


def predict():
    np_embed = np.load('movie_embeddings.npy')
    print(np_embed.shape)

    embedding_1 = np_embed[1654]
    embedding_2 = np_embed[1664]

    embedding_1 = embedding_1.reshape(1, -1)
    embedding_2 = embedding_2.reshape(1, -1)
    similarity = cosine_similarity(embedding_1, embedding_2)
    print(similarity[0][0])


def main():
    df = pd.read_csv("rephrased_overviews_movies_with_cast.csv")
    apply_text_to_lower_and_strip_to_dataframe(df)

    clean_overviews = df['clean_overviews'].iloc[:5000].tolist()

    create_data_for_training(clean_overviews)


if __name__ == "__main__":
    main()
