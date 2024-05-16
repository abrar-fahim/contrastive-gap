
import torch
import matplotlib.pyplot as plt

import sys
import os
import wandb

from tqdm import tqdm
import random
import numpy as np



# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 

from src.config import *
from dataset_processors.mscoco_processor import MSCOCOProcessor
from clips.clip_assembler import ClipAssembler
from sklearn.decomposition import PCA



training_hyperparameters['temperature'] = 0.07
training_hyperparameters['encoder1_modality'] = 'image'
training_hyperparameters['encoder2_modality'] = 'image'
training_hyperparameters['same_inputs'] = True
training_hyperparameters['clip_projection_dim'] = 512
training_hyperparameters['vision_model'] = 'VIT'
training_hyperparameters['use_train_as_val'] = True
training_hyperparameters['dataset'] = ClipDatasets.MSCOCO.value
training_hyperparameters['validation_dataset_size'] = 2048
training_hyperparameters['validation_batch_size'] = 2048
training_hyperparameters['use_small_trainloader'] = True
training_hyperparameters['small_train_loader_dataset_size'] = 2048
training_hyperparameters['batch_size'] = 256
training_hyperparameters['seed'] = 2

training_hyperparameters['num_workers'] = 0
training_hyperparameters['prefetch_factor'] = None
training_hyperparameters['persistent_workers'] = False

wandb.init(config=training_hyperparameters)


torch.manual_seed(wandb.config['seed'])
random.seed(wandb.config['seed'])
np.random.seed(wandb.config['seed'])
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

processor = MSCOCOProcessor()

checkpoint_path = '/Users/abrarfahim/Documents/UofA/CLIP/clipverse/paper_checkpoints/all factors for gap accounted for/T0.07_Lit_2_scratch_I1I1E1E2_512_train_as_val_2048_mscoco_VIT.pt'

device = torch.device(config_cuda_device if torch.cuda.is_available() else "cpu")


checkpoint = torch.load(checkpoint_path, map_location=device)





with torch.no_grad():

    clip_model = ClipAssembler().clip_model.to(device)
    clip_model.load_state_dict(checkpoint['model_state_dict'])

    clip_model.eval()

    all_image_embeddings = []
    all_text_embeddings = []

    n = 2

    for (imgs, caps) in tqdm(processor.train_dataloader):
        image_embeddings = clip_model.encoder1_features(imgs)['embeds']
        text_embeddings = clip_model.encoder2_features(caps)['embeds']

        all_image_embeddings.append(image_embeddings)
        all_text_embeddings.append(text_embeddings)

        if len(all_image_embeddings) >= n:
            break
    
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    all_embeddings = torch.cat([all_image_embeddings, all_text_embeddings], dim=0)

    # pca to 2d
    print('Performing PCA')
    pca = PCA(n_components=2, )
    pca.fit(all_embeddings.cpu().numpy())
    all_embeddings_pca = pca.transform(all_embeddings.cpu().numpy())

    image_embeddings_pca = all_embeddings_pca[:all_image_embeddings.shape[0]]
    text_embeddings_pca = all_embeddings_pca[all_image_embeddings.shape[0]:]
    

    # scatter plot the embeddings
    plt.scatter(image_embeddings_pca[:, 0], image_embeddings_pca[:, 1], label='Image Embeddings', )
    plt.scatter(text_embeddings_pca[:, 0], text_embeddings_pca[:, 1], label='Text Embeddings')
    plt.legend()
    plt.show()


