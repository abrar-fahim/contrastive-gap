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


config_cuda_device = 'cuda:5'

training_hyperparameters['temperature'] = 0.01
training_hyperparameters['encoder1_modality'] = 'image'
training_hyperparameters['encoder2_modality'] = 'text'
training_hyperparameters['same_inputs'] = False
training_hyperparameters['clip_projection_dim'] = 32
training_hyperparameters['vision_model'] = 'VIT'
training_hyperparameters['use_train_as_val'] = False
training_hyperparameters['dataset'] = ClipDatasets.MSCOCO.value
training_hyperparameters['validation_dataset_size'] = 21
training_hyperparameters['validation_batch_size'] = 21
training_hyperparameters['use_small_trainloader'] = True
training_hyperparameters['small_train_loader_dataset_size'] = 32
training_hyperparameters['seed'] = 2
training_hyperparameters['train_from_scratch'] = True
training_hyperparameters['cuda_device'] = config_cuda_device





wandb.init(config=training_hyperparameters)


with torch.no_grad():


    # set seed
    torch.manual_seed(wandb.config['seed'])
    random.seed(wandb.config['seed'])
    np.random.seed(wandb.config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"



    device = torch.device(config_cuda_device if torch.cuda.is_available() else "cpu")

    val_batch_cache_file = 'datasets/mscoco/val_batch_cache_mscoco_full_5k.pt'

    # evaluator = Evaluator(MSCOCOProcessor(), val_batch_cache_file)
    evaluator = Evaluator(MSCOCOProcessor())


    clip_model = ClipAssembler().clip_model.to(device)


    # 3D
    # checkpoint_path = 'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_3_val_as_val_512_mscoco_VIT_pretrained_FINAL.pt'
    # checkpoint_path = 'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_3_val_a[s_val_512_mscoco_VIT_pretrained_FINAL.pt'

    # 16D
    # checkpoint_path = 'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_16_val_as_val_512_mscoco_VIT_pretrained_FINAL.pt'
    # checkpoint_path = 'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_16_val_as_val_512_mscoco_VIT_pretrained_FINAL.pt'

    # 64D
    # checkpoint_path = 'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_FINAL.pt'
    # checkpoint_path = 'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_FINAL.pt'


    '''
    CUAXU
    '''

    #32D
    # checkpoint_path = 'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt'
    # checkpoint_path = 'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt'


    #64D
    # checkpoint_path = 'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt'
    # checkpoint_path = 'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt'





    # 128D 
    # checkpoint_path = 'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt'
    # checkpoint_path = 'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt'



    # checkpoint = torch.load(default_checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_state_dict = checkpoint['model_state_dict']

    clip_model.load_state_dict(model_state_dict)

    # clip_model.half()

    evaluator.set_val_outputs(clip_model, output_loss=False)

    evaluator.evaluate_model(clip_model, 0, 0)

    # evaluator.get_val_image_classification_acc()

    # evaluator.get_dataset_zero_shot_acc(clip_model, CIFAR10Processor())
    # evaluator.get_dataset_zero_shot_acc(clip_model, CIFAR100Processor())
    # evaluator.get_dataset_zero_shot_acc(clip_model, ImageNet1k())
    # evaluator.get_dataset_linear_probe_accuracy(clip_model, CIFAR100Processor())
    # evaluator.get_dataset_linear_probe_accuracy(clip_model, CIFAR10Processor())

    # evaluator.get_dataset_metrics(clip_model, CIFAR10Processor())




    # evaluator.evaluate_model(clip_model, 0, 0)