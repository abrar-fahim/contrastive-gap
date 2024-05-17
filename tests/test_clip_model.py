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
from dataset_processors.cifar100_processor import CIFAR100Processor
from dataset_processors.cifar10_processor import CIFAR10Processor
from dataset_processors.imagenet_processor import ImageNet1k
from clips.clip_assembler import ClipAssembler


training_hyperparameters['temperature'] = 0.01
training_hyperparameters['encoder1_modality'] = 'image'
training_hyperparameters['encoder2_modality'] = 'text'
training_hyperparameters['same_inputs'] = False
training_hyperparameters['clip_projection_dim'] = 64
training_hyperparameters['vision_model'] = 'VIT'
training_hyperparameters['use_train_as_val'] = False
training_hyperparameters['dataset'] = ClipDatasets.MSCOCO.value
training_hyperparameters['validation_dataset_size'] = 32
training_hyperparameters['validation_batch_size'] = 32
training_hyperparameters['use_small_trainloader'] = True
training_hyperparameters['small_train_loader_dataset_size'] = 32
training_hyperparameters['seed'] = 2
training_hyperparameters['train_from_scratch'] = True


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
# evaluator = Evaluator(MSCOCOProcessor())


clip_model = ClipAssembler().clip_model.to(device)

# checkpoint_path = 'checkpoints/T0.01_uniform_2_finetune_MLP_I1C2E1E2_512_val_as_val_2048_conceptual_captions_VIT_pretrained.pt'
# checkpoint_path = 'checkpoints/T0.01_Lit_2_finetune_MLP_I1C2E1E2_512_val_as_val_2048_conceptual_captions_VIT_pretrained.pt'

default_checkpoint_path = 'checkpoints/T0.07_Lit_2_scratch_I1C2E1E2_512_val_as_val_2048_conceptual_captions_VIT.pt'

uniform_checkpoint_path = 'checkpoints/T0.07_uniform_2_scratch_I1C2E1E2_512_val_as_val_2048_conceptual_captions_VIT.pt'

uniform_finetune_checkpoint_path = 'checkpoints/T0.01_uniform_align_2_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT16_pretrained.pt'

default_finetune_checkpoint_path = 'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained.pt'

# checkpoint = torch.load(default_checkpoint_path)
checkpoint = torch.load(default_finetune_checkpoint_path, map_location=device)

model_state_dict = checkpoint['model_state_dict']

clip_model.load_state_dict(model_state_dict)

# evaluator.get_dataset_zero_shot_acc(clip_model, CIFAR10Processor())
# evaluator.get_dataset_zero_shot_acc(clip_model, CIFAR100Processor())
evaluator.get_dataset_zero_shot_acc(clip_model, ImageNet1k())
# evaluator.get_dataset_linear_probe_accuracy(clip_model, CIFAR100Processor())

# evaluator.get_dataset_metrics(clip_model, CIFAR10Processor())




# evaluator.evaluate_model(clip_model, 0, 0)