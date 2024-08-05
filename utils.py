import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda
from torch.utils.data import Dataset
from PIL import Image
import argparse
import pandas as pd
import numpy as np
import torch

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
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        anchor_path = self.dataframe.iloc[idx, 0]
        positive_path = self.dataframe.iloc[idx, 1]
        negative_path = self.dataframe.iloc[idx, 2]
                
        anchor_image = Image.open(anchor_path)
        positive_image = Image.open(positive_path)
        negative_image = Image.open(negative_path)

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image

# Triplet Selection
def offline_triplet_selection(embeddings_df, minibatch=1800, margin=0.2, max_triplets=None):
    triplets = []

    # Reordena aleatoriamente e extrai os embeddings diretamente
    embeddings_df = embeddings_df.sample(frac=1).reset_index(drop=True)

    # Corrige a extração de embeddings para lidar com arrays aninhados
    embeddings = np.stack(embeddings_df['embedding'].apply(lambda x: np.array(x[0], dtype=np.float32)).values)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)

    ids = embeddings_df['id'].to_numpy()
    
    for i in range(0, len(embeddings_df), minibatch):
        batch_indices = list(range(i, min(i + minibatch, len(embeddings_df))))
        batch_embeddings = embeddings[batch_indices]
        batch_ids = ids[batch_indices]

        distances = torch.cdist(batch_embeddings, batch_embeddings, p=2)
        
        for anchor_idx in range(len(batch_indices)):
            anchor_id = batch_ids[anchor_idx]
            positive_mask = batch_ids == anchor_id
            
            # Evita a própria âncora como positivo
            positive_mask[anchor_idx] = False
            positive_mask = torch.tensor(positive_mask)
            
            for positive_idx in torch.where(positive_mask)[0]:
                d_ap = distances[anchor_idx, positive_idx]
                
                negative_mask = (distances[anchor_idx] > d_ap) & (distances[anchor_idx] < d_ap + margin) & (~positive_mask)
                
                for negative_idx in torch.where(negative_mask)[0]:
                    dist = distances[anchor_idx, negative_idx].item() - d_ap.item()
                    triplets.append((batch_indices[anchor_idx], batch_indices[positive_idx], batch_indices[negative_idx], dist))

                    if max_triplets is not None and len(triplets) >= max_triplets:
                        break
                    
    triplets_df = pd.DataFrame(triplets, columns=['anchor_idx', 'positive_idx', 'negative_idx', 'dist'])

    # Mapeia os índices de volta para os caminhos das imagens usando apply e loc
    triplets_df['anchor_path'] = triplets_df['anchor_idx'].apply(lambda x: embeddings_df['path'].loc[x])
    triplets_df['positive_path'] = triplets_df['positive_idx'].apply(lambda x: embeddings_df['path'].loc[x])
    triplets_df['negative_path'] = triplets_df['negative_idx'].apply(lambda x: embeddings_df['path'].loc[x])

    triplets_img_paths_df = triplets_df[['anchor_path', 'positive_path', 'negative_path', 'dist']]
    
    # Retornar max_triplets escolhidos aleatoriamente
    if max_triplets is not None:
        triplets_img_paths_df = triplets_img_paths_df.sample(n=max_triplets).reset_index(drop=True)
        
    return triplets_img_paths_df

def parse_args():
    parser = argparse.ArgumentParser(description="Treinar a facenet")
    parser.add_argument('--num_triplets', type=int, default=100_000, help='Número de triplets (default: 100.000)')
    parser.add_argument('--minibatch', type=int, default=1800, help='Tamanho do minibatch (default: 1800)')
    parser.add_argument('--batch_size', type=int, default=8, help='Tamanho do batch (default: 8)')
    parser.add_argument('--accumulation', type=int, default=64, help='Acumulação de gradiente (default: 64)')
    parser.add_argument('--epochs', type=int, default=8, help='Número de epochs (default: 8)')
    parser.add_argument('--margin', type=float, default=0.2, help='Margem para triplet loss (default: 0.2)')
    parser.add_argument('--num_workers', type=int, default=0, help='Número de workers para o DataLoader (default: 0)')
    parser.add_argument('--data_path', type=str, default='./data/LFW/lfw-faces/', help='Caminho para o dataset (default: ./data/lfw-faces/)')
    parser.add_argument('--device', type=str, default='cuda', help='Dispositivo para treinamento (default: cuda)')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/', help='Caminho para salvar os checkpoints (default: ./checkpoints/)')
    
    return parser.parse_args()
