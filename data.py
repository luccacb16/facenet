from facenet_pytorch import MTCNN, extract_face, InceptionResnetV1
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda
from PIL import Image
import os
from tqdm.notebook import tqdm as tqdm
import pandas as pd
import numpy as np

from .utils import device, transform

# Se não baixou os dados ainda
if not os.path.exists('./data/'):
    os.makedirs('./data/')
    os.system('!kaggle datasets download -d jessicali9530/lfw-dataset -p ./data/')

# Se já não foi unzipado
if not os.path.exists('./data/lfw-deepfunneled/'):
    os.system('!unzip -q ./data/lfw-dataset.zip -d ./data/')
    
# Se já não limpou o diretório
os.system('rm -f ./data/lfw-dataset.zip')
os.system('rm -f ./data/*.csv')

# --------------------------------------------------------------------------------------------
# Extraindo as faces

print(f'Device name: {torch.cuda.get_device_name(0)}')

RAW_LFW_PATH = './data/lfw-deepfunneled/lfw-deepfunneled/'
FACES_PATH = './data/lfw-faces/'

mtcnn = MTCNN(keep_all=True, device=device)

def extract_faces(raw_images_path: str, faces_images_path: str) -> pd.DataFrame:
    if not os.path.exists(faces_images_path):
        os.makedirs(faces_images_path)
        
    subfolders = os.listdir(raw_images_path)
    
    nobox = []
    df = []
    
    for id, folder in enumerate(tqdm(subfolders, desc='Extraindo rostos', unit='img')):
        imgs = os.listdir(f'{raw_images_path}/{folder}')
        
        for file in imgs:
            save_path = os.path.join(faces_images_path, file.split('/')[-1])
            df.append((id, save_path))
            
            img = Image.open(os.path.join(raw_images_path + folder, file))
        
            boxes, _ = mtcnn.detect(img)
            
            if boxes is not None:
                extract_face(img, boxes[0], save_path=save_path)
            else:
                nobox.append(file)
                
    df = pd.DataFrame(df, columns=['id', 'path'])
    
    return nobox, df

nobox, df = extract_faces(RAW_LFW_PATH, FACES_PATH)

# Salvando o csv
df.to_csv('./data/lfw_faces.csv', index=False)

print(f'Número de imagens sem face detectada: {len(nobox)}')
print(f'Número de imagens com face detectada: {len(df)}')
print(f'Número de identidades: {df["id"].nunique()}')

os.system('rm -rf ../data/lfw-deepfunneled/')

# --------------------------------------------------------------------------------------------
# Split de treino e teste

ids_count = df['id'].value_counts()
valid_ids = ids_count[ids_count >= 5].index

shuffled_ids = valid_ids.to_numpy()
np.random.seed(42)
np.random.shuffle(shuffled_ids)

test_ids = shuffled_ids[:49]

# Criar os dataframes de treino e teste
test_df = df[df['id'].isin(test_ids)]
train_df = df[~df['id'].isin(test_ids)]

print(f"Identidades no conjunto de teste: {test_df['id'].nunique()}")
print(f"Imagens no conjunto de teste: {len(test_df)}\n")

print(f"Identidades no conjunto de treino: {train_df['id'].nunique()}")
print(f"Imagens no conjunto de treino: {len(train_df)}")

# --------------------------------------------------------------------------------------------
# Extraindo os embeddings

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_embeddings(model, train_df, transform, device) -> list:
    embeddings = []
    for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
        img = Image.open(row['path'])
        img = transform(img).to(device)
        img_embedding = model(img.unsqueeze(0)).detach().cpu().numpy()
        embeddings.append(img_embedding)
    
    return embeddings

embeddings = extract_embeddings(resnet, train_df, transform, device)
train_df['embedding'] = embeddings

train_df.to_pickle('../data/lfw_train_embeddings.pkl')
test_df.to_csv('../data/lfw_test.csv', index=False)