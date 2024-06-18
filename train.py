import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op")

from utils import device, transform, TripletLoss, TripletDataset, get_triplets, parse_args
from nn2 import FaceNet

CHECKPOINT_PATH = './checkpoints/'

def train(model, dataset, optimizer, triplet_loss, checkpoint_path: str, epochs, batch_size: int = 64, num_samples: int = 1800):
    for epoch in range(epochs):
        sampled_indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(sampled_indices), num_workers=2, pin_memory=True)

        model.train()
        epoch_loss = 0

        for images, labels, paths in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            embeddings = []
            images = images.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                embeddings = model(images)

            triplets = get_triplets(embeddings, labels)
            triplets = torch.tensor(triplets, dtype=torch.long, device=device)
            print(f'Triplets: {triplets.shape[0]}')
            total_loss = 0

            for anchor, positive, negative in triplets:
                anchor = embeddings[anchor].unsqueeze(0)
                positive = embeddings[positive].unsqueeze(0)
                negative = embeddings[negative].unsqueeze(0)
                loss = triplet_loss(anchor, positive, negative)
                loss.backward()
                total_loss += loss.item()

            optimizer.step()
            epoch_loss += total_loss / len(triplets) if triplets.nelement() != 0 else 0

        torch.save(model.state_dict(), os.path.join(checkpoint_path, f'epoch_{epoch+1}.pt'))
        print(f"Epoch = {epoch+1} | Loss = {epoch_loss / len(dataloader)}")

# ---------------------------------------------------------------------------------------------------------------------

args = parse_args()

print(f'Device: {device}')
print(f'Device name: {torch.cuda.get_device_name()}\n')

# Carregar os dados
lfw = pd.read_csv('./data/lfw_train.csv')

dataset = TripletDataset(dataframe = lfw, 
                         transform = transform)

torch.set_float32_matmul_precision('high')

facenet = FaceNet().to(device)
facenet = torch.compile(facenet)

# Parâmetros
batch_size = args.batch_size

triplet_loss = TripletLoss(margin=args.margin)
adamW = optim.AdamW(facenet.parameters(), lr=3e-4)

losses = train(
    model            = facenet,
    dataset          = dataset,
    optimizer        = adamW,
    triplet_loss     = triplet_loss,
    checkpoint_path  = CHECKPOINT_PATH,
    epochs           = args.epochs,
    batch_size       = batch_size,
    num_samples      = args.num_samples
)