import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda
from torch.utils.data import Dataset
from PIL import Image
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = Compose(
    [
    Resize((160, 160)), 
    ToTensor(), 
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> float:
        # Loss = max(||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + margin, 0)
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        
        return losses.mean()

# Triplet Dataset
class TripletDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx]['id']

        if self.transform:
            image = self.transform(image)

        return image, label, img_path

# Triplet Selection
def get_triplets(embeddings, labels, margin=0.2):
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # Calcula a matriz de distância euclidiana
    triplets = []
    
    for anchor_idx in range(dist_matrix.size(0)):
        anchor_label = labels[anchor_idx]
        positive_indices = (labels == anchor_label).nonzero().view(-1)
        negative_indices = (labels != anchor_label).nonzero().view(-1)
        
        if len(positive_indices) < 2:
            continue

        anchor_pos_distances = dist_matrix[anchor_idx][positive_indices]
        anchor_neg_distances = dist_matrix[anchor_idx][negative_indices]

        # Escolha do hardest positive
        positive_idx = positive_indices[anchor_pos_distances.argmax()]
        hardest_pos_distance = anchor_pos_distances.max()

        # Escolha de semi-hard negatives
        semi_hard_negatives = anchor_neg_distances[(anchor_neg_distances < hardest_pos_distance) & (anchor_neg_distances > hardest_pos_distance - margin)]
        if len(semi_hard_negatives) > 0:
            negative_idx = negative_indices[semi_hard_negatives.argmin()]
            triplets.append((anchor_idx, positive_idx.item(), negative_idx.item()))

    return triplets



def parse_args():
    parser = argparse.ArgumentParser(description="Treinar a facenet")
    parser.add_argument('--num_samples', type=int, default=1800, help='Mini batch size para seleção de triplets (default: 1800)')
    parser.add_argument('--batch_size', type=int, default=64, help='Tamanho do batch (default: 64)')
    parser.add_argument('--epochs', type=int, default=16, help='Número de epochs (default: 16)')
    parser.add_argument('--margin', type=float, default=0.2, help='Margem para triplet loss (default: 0.2)')
    
    return parser.parse_args()