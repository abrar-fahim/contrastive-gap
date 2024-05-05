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
import random
import numpy as np

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 

from src.config import training_hyperparameters




wandb.init(config=training_hyperparameters)




from tqdm import tqdm

from dataset_processors.conceptual_captions_processor import ConceptualCaptionsProcessor

torch.manual_seed(wandb.config['seed'])

random.seed(wandb.config['seed'])
np.random.seed(wandb.config['seed'])
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

processor = ConceptualCaptionsProcessor()

# for image, caption in tqdm(data_pipe):
for image, caption in tqdm(processor.train_dataloader):
    # print(f"Caption: {caption}")
    # print(f"Image size: {image.shape}")
    # Don't do anything here.  We just want to test the loading speed.
    pass


# for i, (image, caption) in enumerate(data_pipe):
#     if i >= 10:
#         break
#     print(f"Caption {i + 1}: {caption}")
#     print(f"Image size: {image.size}")