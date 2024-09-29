import numpy as np
import torch
from torch.utils.data import Dataset


class TripletMoviePairDataset(Dataset):
    def __init__(self, embeddings, triplets, similarity_matrix):
        self.embeddings = embeddings
        self.triplets = triplets
        self.similarity_matrix = similarity_matrix

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        cosine_sim = self.similarity_matrix[anchor, positive]

        return self.embeddings[anchor], self.embeddings[positive], self.embeddings[negative], cosine_sim
