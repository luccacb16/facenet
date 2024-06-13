import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op")

from utils import device, transform, TripletLoss, TripletDataset, get_triplets
from nn2 import FaceNet

CHECKPOINT_PATH = './checkpoints/'

def train(model: nn.Module, 
          checkpoint_path: str,
          triplet_loss: nn.Module, 
          dataloader: DataLoader, 
          optimizer: optim.Optimizer, 
          epochs: int, 
          batch_size: int,
          accumulation: int = 1):
    
    losses = []
    print('Entrou em train')
    for epoch in range(epochs):
        print('Entrou em epoch')
        model.train()
        epoch_loss = 0.0
        accumulated_loss = 0.0  # To track accumulated loss for gradient accumulation
        
        tqdm_progress = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch+1}', leave=False)
        for step, (triplet_data, triplet_labels) in tqdm_progress:
            print(type(triplet_data))
            print(triplet_data.shape)
            
            triplet_data = triplet_data.to(device)
            triplet_labels = triplet_labels.to(device)
            print('Entrou em step')
            # Calcular os embeddings dos triplets
            print('Calculando os primeiros embeddings')
            with torch.no_grad():
                triplet_embeddings = model(triplet_data)
            print('Primeiros embeddings calculados')
            
            # Selecionar os triplets
            triplet_embeddings_np = triplet_embeddings.detach().cpu().numpy()
            triplet_labels_np = triplet_labels.detach().cpu().numpy()
            print('Começando seleção')
            triplets = get_triplets(triplet_embeddings_np, triplet_labels_np, batch_size, triplet_loss.margin)
            print('Selecionou!')
            if triplets:
                num_triplets = len(triplets)
                num_batches = (num_triplets + batch_size - 1) // batch_size

                for batch_idx in range(num_batches):
                    print('Entrou no batch')
                    batch_start = batch_idx * batch_size
                    batch_end = min((batch_idx + 1) * batch_size, num_triplets)
                    batch_triplets = triplets[batch_start:batch_end]
                    
                    anchors = torch.stack([triplet_embeddings[a] for a, _, _ in batch_triplets])
                    positives = torch.stack([triplet_embeddings[p] for _, p, _ in batch_triplets])
                    negatives = torch.stack([triplet_embeddings[n] for _, _, n in batch_triplets])
                    
                    anchors = anchors.to(device)
                    positives = positives.to(device)
                    negatives = negatives.to(device)
                    
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        loss = triplet_loss(anchors, positives, negatives) / accumulation
                    loss.backward()
                    
                    accumulated_loss += loss.item()
                    
                    # Atualiza os parâmetros após 'accumulation' passos
                    if (step + 1) % accumulation == 0:
                        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip grad norm
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        epoch_loss += accumulated_loss
                        accumulated_loss = 0.0
                        
            tqdm_progress.set_postfix(loss=accumulated_loss)
        
        # Salvamento do checkpoint do modelo
        torch.save(model.state_dict(), checkpoint_path + f'epoch_{epoch+1}.pt')
        
        # Salvar o loss da época
        losses.append(epoch_loss / len(dataloader))
        
        print(f'epoch {epoch+1}/{epochs} | loss: {epoch_loss / len(dataloader):.6f} | norm: {norm:.4f}')
    
    return losses

# ---------------------------------------------------------------------------------------------------------------------

print(f'Device: {device}')
print(f'Device name: {torch.cuda.get_device_name()}\n')

# Carregar os dados
lfw = pd.read_csv('./data/lfw_train.csv')

# .apply em 'path'
lfw['path'] = lfw['path'].apply(lambda x: './data/lfw-faces/' + x.split('/')[-1])

dataset = TripletDataset(paths     = lfw['path'].values, 
                         labels    = lfw['id'].values, 
                         transform = transform)

torch.set_float32_matmul_precision('high')

facenet = FaceNet().to(device)
facenet = torch.compile(facenet)
print('Modelo compilado')

# Parâmetros
batch_size = 8

triplet_loss = TripletLoss(margin=0.5)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

adamW = optim.AdamW(facenet.parameters(), lr=3e-4)

losses = train(
    model            = facenet,
    checkpoint_path  = CHECKPOINT_PATH,
    triplet_loss     = triplet_loss,
    dataloader       = dataloader,
    optimizer        = adamW,
    epochs           = 16,
    batch_size       = batch_size,
    accumulation     = 64 / batch_size
)
