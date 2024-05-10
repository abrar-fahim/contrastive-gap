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
from torchdata.datapipes.iter import FileLister, FileOpener, Decompressor

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 

from src.config import training_hyperparameters
from src.evaluator import Evaluator

from src.config import *
from tqdm import tqdm
from dataset_processors.mscoco_processor import MSCOCOProcessor
from clips.clip_assembler import ClipAssembler


training_hyperparameters['temperature'] = 0.07
training_hyperparameters['encoder1_modality'] = 'image'
training_hyperparameters['encoder2_modality'] = 'text'
training_hyperparameters['same_inputs'] = False
training_hyperparameters['clip_projection_dim'] = 512
training_hyperparameters['vision_model'] = 'VIT'
training_hyperparameters['use_train_as_val'] = False
training_hyperparameters['dataset'] = ClipDatasets.MSCOCO.value
training_hyperparameters['validation_dataset_size'] = 2048
training_hyperparameters['validation_batch_size'] = 2048
training_hyperparameters['use_small_trainloader'] = True
training_hyperparameters['small_train_loader_dataset_size'] = 2048
training_hyperparameters['seed'] = 2


wandb.init(config=training_hyperparameters)



# set seed
torch.manual_seed(wandb.config['seed'])
random.seed(wandb.config['seed'])
np.random.seed(wandb.config['seed'])
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"



device = torch.device(config_cuda_device if torch.cuda.is_available() else "cpu")

evaluator = Evaluator(MSCOCOProcessor())


clip_model = ClipAssembler().clip_model.to(device)

checkpoint_path = 'checkpoints/T0.07_Lit_2_scratch_I1C2E1E2_1024_val_as_val_2048.pt'

checkpoint = torch.load(checkpoint_path)

model_state_dict = checkpoint['model_state_dict']

clip_model.load_state_dict(model_state_dict)


evaluator.evaluate_model(clip_model, 0, 0)