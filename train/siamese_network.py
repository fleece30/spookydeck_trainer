import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        # self.bn3 = nn.BatchNorm1d(32)
        # self.fc4 = nn.Linear(32, 16)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.selu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.selu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        # x = F.selu(self.bn3(self.fc3(x)))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x


def triplet_loss(anchor, positive, negative, margin):
    positive_distance = F.pairwise_distance(anchor, positive, p=2)
    negative_distance = F.pairwise_distance(anchor, negative, p=2)
    loss = F.relu(positive_distance - negative_distance + margin)
    return loss.mean()


def combined_loss(anchor_output, positive_output, negative_output, triplet_margin, cosine_margin, alpha):
    triplet_loss_value = triplet_loss(anchor_output, positive_output, negative_output, triplet_margin)
    cos_sim_pos = F.cosine_similarity(anchor_output, positive_output)
    cos_sim_neg = F.cosine_similarity(anchor_output, negative_output)
    cos_sim_loss_value = F.relu(cosine_margin - cos_sim_pos + cos_sim_neg).mean()

    loss = alpha * triplet_loss_value + (1 - alpha) * cos_sim_loss_value
    return loss


def train(data_loader, embedding_dim, num_epochs):
    model = SiameseNetwork(embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for anchor, positive, negative, cosine_sim in data_loader:
            optimizer.zero_grad()
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)
            loss = combined_loss(anchor_output, positive_output, negative_output, triplet_margin=0.2, cosine_margin=0.1, alpha=0.8)
            # loss = triplet_loss(anchor_output, positive_output, negative_output, margin=0.15)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f'Epoch: {epoch}, Loss: {total_loss / len(data_loader)}')

    torch.save(model.state_dict(), './siamese_network.pt')
