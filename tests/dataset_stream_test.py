# conceptual captions streaming test

from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.iter import HttpReader, LineReader
import torchdata.datapipes as dp
import aiohttp
from PIL import Image
import io
from typing import Optional
from typing import List
from typing import Sequence, Tuple
import asyncio
from typing import Generator
import torch
import matplotlib.pyplot as plt

import sys
import os
import wandb
import time

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 

from src.config import training_hyperparameters



wandb.init(config=training_hyperparameters)

torch.manual_seed(wandb.config['seed'])


from tqdm import tqdm

from dataset_processors.conceptual_captions_processor import ConceptualCaptionsProcessor

processor = ConceptualCaptionsProcessor()

pbar = tqdm(processor.train_dataloader)

n_images = 0

start_time = time.time()
# for image, caption in tqdm(data_pipe):
for image, caption in pbar:

    

    # print('caption 0 ', caption[0])

    if image == None:
        print('Image is None')
        continue

    current_time = time.time()

    n_images += image.shape[0]

    pbar.set_description(f'Rate for processing {n_images} images (per second): {n_images / (current_time - start_time)}')
    # print(f"Caption: {caption}")
    # print(f"Image size: {image.shape}")
    # # display first image
    # plt.imshow(image[0].permute(1, 2, 0))
    # plt.show()

    
    # Don't do anything here.  We just want to test the loading speed.
    pass


# for i, (image, caption) in enumerate(data_pipe):
#     if i >= 10:
#         break
#     print(f"Caption {i + 1}: {caption}")
#     print(f"Image size: {image.size}")