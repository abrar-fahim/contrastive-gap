'''
Compute rank of dataset of varying batch_sizes
technically, its the rank of the CLIP embeddings
'''
import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from src.config import *

# import pca
from sklearn.decomposition import PCA


from clips.hf_clip import HFClip
from dataset_processors.mscoco_processor import MSCOCOProcessor
import random
import wandb
import numpy as np
from clips.clip_assembler import ClipAssembler
from clips.hf_clip import HFClipOutput

def main():



    training_hyperparameters['temperature'] = 0.07
    training_hyperparameters['encoder1_modality'] = 'image'
    training_hyperparameters['encoder2_modality'] = 'text'
    training_hyperparameters['same_inputs'] = False
    training_hyperparameters['clip_projection_dim'] = 512
    training_hyperparameters['vision_model'] = 'VIT'
    training_hyperparameters['use_train_as_val'] = True
    training_hyperparameters['dataset'] = ClipDatasets.MSCOCO.value
    training_hyperparameters['validation_dataset_size'] = 2048
    training_hyperparameters['validation_batch_size'] = 2048
    training_hyperparameters['use_small_trainloader'] = True
    training_hyperparameters['small_train_loader_dataset_size'] = 2048
    training_hyperparameters['seed'] = 2
    training_hyperparameters['num_workers'] = 4


    if wandb.run == None: # so that wandb doesnt reset config in case this run is part of a sweep
        wandb.init(
            project="clipverse", 
            # track hyperparameters and run metadata
            config=training_hyperparameters,
            # name=generate_csv_file_name(clip_model)
        )


    # set seed
    torch.manual_seed(wandb.config['seed'])
    random.seed(wandb.config['seed'])
    np.random.seed(wandb.config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

            
    device = torch.device(config_cuda_device if torch.cuda.is_available() else "cpu")

    dataset_processor = MSCOCOProcessor()

    clip_model = ClipAssembler().clip_model.to(device)


    for (imgs, captions) in dataset_processor.train_dataloader:

        with torch.no_grad():

            clip_outputs: HFClipOutput = clip_model(imgs, captions, return_all=True, output_loss=False)

            image_embeds = clip_outputs.image_embeds
            text_embeds = clip_outputs.text_embeds

            # image_embeds = image_embeds.T
            # text_embeds = text_embeds.T

            print('image embeds ', image_embeds.shape)
            print('text embeds ', text_embeds.shape)

            # compute rank of image_embeds
            U, S, Vh = torch.linalg.svd(image_embeds)

            print('image embeds torch rank ', torch.linalg.matrix_rank(image_embeds, ))

            print('image embeds S ', S)
            print('image my rank ', torch.count_nonzero(S > 1))
            print( ' --- ' )

            # # compute rank of text_embeds
            # U, S, Vh = torch.linalg.svd(text_embeds)
            # print('text embeds rank ', torch.linalg.matrix_rank(text_embeds, ))
            # print('text embeds S ', S)

            # pca rank
            pca = PCA(n_components=min(image_embeds.shape[0], image_embeds.shape[1]))
            pca.fit(image_embeds.cpu().numpy())

            print('pca singular values IMAGES: ', pca.singular_values_)
            print('image explained variances: ', pca.explained_variance_ratio_)






        break

    wandb.finish()

if __name__ == '__main__':
    main()