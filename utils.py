import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
from PIL import Image
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = Compose(
    [
    Resize((224, 224)), 
    Lambda(lambda x: x.convert('RGB')), 
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
    def __init__(self, paths: List[str], labels: List[int], transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.paths[idx])
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        
        return img, label

# Triplet Selection
def get_triplets(embeddings: np.ndarray, labels: np.ndarray, mini_batch: int = 180, margin: float = 0.2) -> List[Tuple[int, int, int]]:
    triplets = []
    total_size = len(labels)
    print(f'Total size {total_size} | mini_batch {mini_batch}')
    for i in range(0, total_size, mini_batch):
        end = min(i + mini_batch, total_size)
        batch_embeddings = embeddings[i:end]
        batch_labels = labels[i:end]
        
        for j in range(len(batch_labels)):
            anchor_embedding = batch_embeddings[j]
            anchor_label = batch_labels[j]
            
            positive_indices = np.where(batch_labels == anchor_label)[0]
            negative_indices = np.where(batch_labels != anchor_label)[0]
            
            if len(positive_indices) <= 1:
                print('Nenhum positivo encontrado para a sample', j)
                continue
            
            positive_indices = positive_indices[positive_indices != j]
            
            pos_distances = np.linalg.norm(anchor_embedding - batch_embeddings[positive_indices], axis=1)
            pos_idx = positive_indices[np.argmax(pos_distances)]
            
            neg_distances = np.linalg.norm(anchor_embedding - batch_embeddings[negative_indices], axis=1)
            neg_idx = negative_indices[np.argmin(neg_distances)]
            
            if np.min(neg_distances) < np.max(pos_distances) + margin:
                triplets.append((i + j, i + pos_idx, i + neg_idx))
    
    print('Triplet montado')
    return triplets

def parse_args():
    parser = argparse.ArgumentParser(description="Treinar a facenet")
    parser.add_argument('--minibatch', type=int, default=64, help='Mini batch size para seleção de triplets (default: 64)')
    parser.add_argument('--epochs', type=int, default=16, help='Número de epochs (default: 16)')
    parser.add_argument('--margin', type=float, default=0.5, help='Margem para triplet loss (default: 0.5)')
    
    return parser.parse_args()