import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op")

from utils import transform, TripletLoss, TripletDataset, parse_args, offline_triplet_selection
from models.nn2 import FaceNet

CHECKPOINT_PATH = './checkpoints/'

import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

def train(model: torch.nn.Module,
          dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          triplet_loss: torch.nn.Module,
          epochs: int = 8,
          accumulation: int = 1,
          checkpoint_path: str = './checkpoints/',
          device: str = 'cuda') -> list:

    model.to(device)
    model.train()
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    losses = []

    epoch_progress = tqdm(range(epochs), desc='Epochs', unit='epoch')
    for epoch in epoch_progress:
        train_loss = 0
        
        batch_progress = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch', leave=True)
        for i, (anchor, positive, negative) in enumerate(batch_progress):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
            
            loss.backward()
            
            if (i + 1) % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            loss_value = loss.item()
            train_loss += loss_value
            batch_progress.set_postfix(loss=f'{loss_value:.4f}')
        
        epoch_loss = train_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}')
        losses.append(epoch_loss)
                
        epoch_progress.set_description(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        epoch_progress.refresh()

        torch.save(model.state_dict(), checkpoint_path + f'epoch_{epoch+1}.pt')
        
    return losses

# ---------------------------------------------------------------------------------------------------------------------

args = parse_args()

device = args.device
batch_size = args.batch_size

print(f'Device: {device}')
print(f'Device name: {torch.cuda.get_device_name()}\n')

embeddings_df = pd.read_pickle('./data/lfw_train_embeddings.pkl')
triplets_df = offline_triplet_selection(embeddings_df, args.minibatch, args.num_triplets, args.margin)

triplets_df['anchor_path'] = triplets_df['anchor_path'].apply(lambda x: f'./data/triplets_df-faces/{x}')
triplets_df['positive_path'] = triplets_df['positive_path'].apply(lambda x: f'./data/triplets_df-faces/{x}')
triplets_df['negative_path'] = triplets_df['negative_path'].apply(lambda x: f'./data/triplets_df-faces/{x}')

dataset = TripletDataset(dataframe = triplets_df, 
                         transform = transform)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

torch.set_float32_matmul_precision('high')

facenet = FaceNet().to(device)
facenet = torch.compile(facenet)

triplet_loss = TripletLoss(margin=args.margin)
adamW = optim.AdamW(facenet.parameters(), lr=3e-4)

losses = train(
    model            = facenet,
    dataloader       = dataloader,
    optimizer        = adamW,
    triplet_loss     = triplet_loss,
    epochs           = args.epochs,
    accumulation     = 64/batch_size,
    checkpoint_path  = CHECKPOINT_PATH,
    device           = args.device
)