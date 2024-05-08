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

from tqdm import tqdm
import random
import numpy as np



# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 

from src.config import *
from dataset_processors.mscoco_processor import MSCOCOProcessor
from dataset_processors.conceptual_captions_processor import ConceptualCaptionsProcessor



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
training_hyperparameters['seed'] = 2

wandb.init(config=training_hyperparameters)


torch.manual_seed(wandb.config['seed'])
random.seed(wandb.config['seed'])
np.random.seed(wandb.config['seed'])
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

processor = MSCOCOProcessor()




with torch.no_grad():
    pass    
