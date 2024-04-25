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


from clips.hf_clip import HFClip
from dataset_processors.mscoco_processor import MSCOCOProcessor
import random
import wandb
import numpy as np
from clips.clip_assembler import ClipAssembler
from clips.hf_clip import HFClipOutput

def main():


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

            print('image embeds ', image_embeds.shape)
            print('text embeds ', text_embeds.shape)

            # compute rank of image_embeds
            U, S, Vh = torch.linalg.svd(image_embeds)

            print('image embeds rank ', torch.linalg.matrix_rank(image_embeds, ))

            print('image embeds S ', S)
            print( ' --- ' )

            # compute rank of text_embeds
            U, S, Vh = torch.linalg.svd(text_embeds)
            print('text embeds rank ', torch.linalg.matrix_rank(text_embeds, ))
            print('text embeds S ', S)






        break

    wandb.finish()

if __name__ == '__main__':
    main()