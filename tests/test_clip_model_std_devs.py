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

import json

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
from dataset_processors.caltech101_processor import Caltech101Processor
from dataset_processors.dtd_processor import DTDProcessor
from clips.clip_assembler import ClipAssembler

DIM = 128



config_cuda_device = 'cuda:5'

training_hyperparameters['temperature'] = 0.01
training_hyperparameters['encoder1_modality'] = 'image'
training_hyperparameters['encoder2_modality'] = 'text'
training_hyperparameters['same_inputs'] = False
training_hyperparameters['clip_projection_dim'] = DIM
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



d32_checkpoints = [
    'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt',
    'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt',
    'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt',

    'checkpoints/T0.01_Lit_24_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
    'checkpoints/T0.01_Lituniform_align_24_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
    'checkpoints/T0.01_Lituniform_align_xuniform_24_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',

    'checkpoints/T0.01_Lit_44_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
    'checkpoints/T0.01_Lituniform_align_44_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
    'checkpoints/T0.01_Lituniform_align_xuniform_44_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt'

]

d64_checkpoints = [
    'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt',
    'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt',
    'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt',

    'checkpoints/T0.01_Lit_24_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
    'checkpoints/T0.01_Lituniform_align_24_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
    'checkpoints/T0.01_Lituniform_align_xuniform_24_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',

    'checkpoints/T0.01_Lit_44_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
    'checkpoints/T0.01_Lituniform_align_44_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
    'checkpoints/T0.01_Lituniform_align_xuniform_44_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt'

]

d128_checkpoints = [
    'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt',
    'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt',
    'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt',

    'checkpoints/T0.01_Lit_24_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
    'checkpoints/T0.01_Lituniform_align_24_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
    'checkpoints/T0.01_Lituniform_align_xuniform_24_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',

    'checkpoints/T0.01_Lit_44_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
    'checkpoints/T0.01_Lituniform_align_44_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
    'checkpoints/T0.01_Lituniform_align_xuniform_44_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt'

]



def get_gap_stuff(clip_model, evaluator: Evaluator):
    ranks = evaluator.get_rank()

    uniformity_loss = evaluator.get_mscoco_uniformity()



    return {
        'mean_cosine_similarity': evaluator.get_mean_cosine_similarity(clip_model.get_temperature()),
        'linear_seperability_accuracy': evaluator.get_linear_seperability(),
        'centroid_euclidean_distance': evaluator.get_centroid_euclidean_distance(),

        'val_image_classification_acc': evaluator.get_val_image_classification_acc(return_all=True),

        'get_val_image_retrieval_acc': evaluator.get_val_image_retrieval_acc(return_all=True),

        'image_variances': ranks['image_explained_variance_ratios'],
        'text_variances': ranks['text_explained_variance_ratios'],

        'image_uniformity_loss': uniformity_loss['image_uniformity_loss'].item(),
        'text_uniformity_loss': uniformity_loss['text_uniformity_loss'].item(),
        'total_uniformity_loss': uniformity_loss['total_uniformity_loss'].item(),
        'cross_encoder_uniform_loss': uniformity_loss['cross_encoder_uniform_loss'].item(),
        'alignment_loss': evaluator.get_mscoco_alignment().item(),
        
    }

def get_zs_stuff(clip_model, evaluator: Evaluator):

    return {
        'imagenet_zs_acc': evaluator.get_dataset_zero_shot_acc(clip_model, ImageNet1k()),
        
        'dtd_zs_acc': evaluator.get_dataset_zero_shot_acc(clip_model, DTDProcessor()),
        
        'caltech101_zs_acc': evaluator.get_dataset_zero_shot_acc(clip_model, Caltech101Processor()),
        
        
        'cifar10_zs_acc': evaluator.get_dataset_zero_shot_acc(clip_model, CIFAR10Processor()),
        
        'cifar100_zs_acc': evaluator.get_dataset_zero_shot_acc(clip_model, CIFAR100Processor()),



        # 'cifar100_gap_stuff': evaluator.get_dataset_metrics(clip_model, CIFAR100Processor()),
        # 'imagenet_gap_stuff': evaluator.get_dataset_metrics(clip_model, ImageNet1k()),
        # 'caltech101_gap_stuff': evaluator.get_dataset_metrics(clip_model, Caltech101Processor()),
        # 'cifar10_gap_stuff': evaluator.get_dataset_metrics(clip_model, CIFAR10Processor()),
        # 'dtd_gap_stuff': evaluator.get_dataset_metrics(clip_model, DTDProcessor()),
        
      
    }

    # evaluator.get_dataset_zero_shot_acc(clip_model, CIFAR10Processor())
    # evaluator.get_dataset_zero_shot_acc(clip_model, CIFAR100Processor())
    # evaluator.get_dataset_zero_shot_acc(clip_model, ImageNet1k())

def get_lp_stuff(clip_model, evaluator: Evaluator):

    return {
        # 'imagenet_lp_acc': evaluator.get_dataset_linear_probe_accuracy(clip_model, ImageNet1k()),
        'dtd_lp_acc': evaluator.get_dataset_linear_probe_accuracy(clip_model, DTDProcessor()),
        'caltech101_lp_acc': evaluator.get_dataset_linear_probe_accuracy(clip_model, Caltech101Processor()),
        'cifar10_lp_acc': evaluator.get_dataset_linear_probe_accuracy(clip_model, CIFAR10Processor()),
        'cifar100_lp_acc': evaluator.get_dataset_linear_probe_accuracy(clip_model, CIFAR100Processor()),
    }

    # evaluator.get_dataset_linear_probe_accuracy(clip_model, CIFAR100Processor())
    # evaluator.get_dataset_linear_probe_accuracy(clip_model, CIFAR10Processor())

    # evaluator.get_dataset_linear_probe_accuracy(clip_model, ImageNet1k())

    # evaluator.get_dataset_linear_probe_accuracy(clip_model, DTDProcessor())
    # evaluator.get_dataset_linear_probe_accuracy(clip_model, Caltech101Processor())


wandb.init(config=training_hyperparameters)


def get_gap_stuff_all(checkpoints: list[str], evaluator: Evaluator, dim: int):
    # checkpoints has checkpoint paths for each dimensionaltiy

    clip_model = ClipAssembler().clip_model.to(device)

    all_gap_stuff = { # for one dimensionality only
        
        'CLIP': {
            'mean_cosine_similarity': [],
            'linear_seperability_accuracy': [],
            'centroid_euclidean_distance': [],

            'val_image_classification_acc': {
                1: [],
                3: [],
                5: [],
                10: []
            },

            'get_val_image_retrieval_acc': {
                1: [],
                3: [],
                5: [],
                10: []
            },

            'image_variances': [],
            'text_variances': [], # not gonna do std devs for this

            'image_uniformity_loss': [],
            'text_uniformity_loss': [],
            'total_uniformity_loss': [],
            'cross_encoder_uniform_loss': [],
            'alignment_loss': [],
            },
        'CUA': {
            'mean_cosine_similarity': [],
            'linear_seperability_accuracy': [],
            'centroid_euclidean_distance': [],

            'val_image_classification_acc': {
                1: [],
                3: [],
                5: [],
                10: []
            },

            'get_val_image_retrieval_acc': {
                1: [],
                3: [],
                5: [],
                10: []
            },

           

            'image_variances': [],
            'text_variances': [], # not gonna do std devs for this

            'image_uniformity_loss': [],
            'text_uniformity_loss': [],
            'total_uniformity_loss': [],
            'cross_encoder_uniform_loss': [],
            'alignment_loss': [],
            

        },
        'CUAXU': {
            'mean_cosine_similarity': [],
            'linear_seperability_accuracy': [],
            'centroid_euclidean_distance': [],
        'val_image_classification_acc': {
                    1: [],
                    3: [],
                    5: [],
                    10: []
                },

            'get_val_image_retrieval_acc': {
                1: [],
                3: [],
                5: [],
                10: []
            },

            'image_variances': [],
            'text_variances': [], # not gonna do std devs for this

            'image_uniformity_loss': [],
            'text_uniformity_loss': [],
            'total_uniformity_loss': [],
            'cross_encoder_uniform_loss': [],
            'alignment_loss': [],

        }
                

    }


    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model_state_dict = checkpoint['model_state_dict']

        clip_model.load_state_dict(model_state_dict)

        # clip_model.half()

        evaluator.set_val_outputs(clip_model, output_loss=False)

        gap_stuff = get_gap_stuff(clip_model, evaluator)

        name: str = None

        topks = [1, 3, 5, 10]

        if 'xuniform' in checkpoint_path:
            name = 'CUAXU'

        elif 'uniform_align' in checkpoint_path:
            name = 'CUA'

        else:
            name = 'CLIP'

        # maintain arrays of items for each key
        for key in gap_stuff:

            if key in ['image_variances', 'text_variances']:
                continue
            if key in ['val_image_classification_acc', 'get_val_image_retrieval_acc']:
                for topk in topks:
                    all_gap_stuff[name][key][topk].append(gap_stuff[key][topk])
                continue


            all_gap_stuff[name][key].append(gap_stuff[key])


    # compute mean and std dev for each key

    names = ['CLIP', 'CUA', 'CUAXU']

    print(' --- ALL GAP STUFF --- ')
    print(all_gap_stuff)
    print()

    all_gap_stuff_output = all_gap_stuff.copy()


    for name in names:
        for key in all_gap_stuff[name]:

            if key in ['image_variances', 'text_variances']:
                continue

            if key in ['val_image_classification_acc', 'get_val_image_retrieval_acc']:
                for topk in topks:
                    all_gap_stuff_output[name][key][topk] = {
                        'mean': np.mean(all_gap_stuff[name][key][topk]),
                        'std_dev': np.std(all_gap_stuff[name][key][topk], ddof=1),
                        '2_std_dev': 2 * np.std(all_gap_stuff[name][key][topk], ddof=1)
                    }
                continue
            all_gap_stuff_output[name][key] = {
                'mean': np.mean(all_gap_stuff[name][key]),
                'std_dev': np.std(all_gap_stuff[name][key], ddof=1),
                '2_std_dev': 2 * np.std(all_gap_stuff[name][key], ddof=1)
            }

    print(' --- ALL GAP STUFF --- ')
    print(all_gap_stuff_output)
    print()






    # write checkpoint path to file
    with open(f'paper_evals/all_gap_{dim}.txt', 'w') as f:

        print({
            'dim': dim,
            'all_gap_stuff': all_gap_stuff_output
        }, file=f)

# set seed
torch.manual_seed(wandb.config['seed'])
random.seed(wandb.config['seed'])
np.random.seed(wandb.config['seed'])
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

device = torch.device(config_cuda_device if torch.cuda.is_available() else "cpu")

val_batch_cache_file = 'datasets/mscoco/val_batch_cache_mscoco_full_5k.pt'

mscoco_evaluator = Evaluator(MSCOCOProcessor(), val_batch_cache_file)

with torch.no_grad():

    if DIM == 32:
        get_gap_stuff_all(d32_checkpoints, mscoco_evaluator, 32)
    elif DIM == 64:
        get_gap_stuff_all(d64_checkpoints, mscoco_evaluator, 64)
    elif DIM == 128:
        get_gap_stuff_all(d128_checkpoints, mscoco_evaluator, 128)

    # evaluator = Evaluator(MSCOCOProcessor())



    # checkpoint = torch.load(default_checkpoint_path)
    

    



    # write both checkpoint file and gap stuff to same file
    



    
    # zs_stuff = get_zs_stuff(clip_model, evaluator)
    # with open(f'paper_evals/{checkpoint_path.split("/")[-1]}_zeroshot.txt', 'w') as f:

    #     print({
    #         'checkpoint_path': checkpoint_path,
    #         'gap_stuff': zs_stuff
    #     }, file=f)


    # lp_stuff = get_lp_stuff(clip_model, evaluator)
    # with open(f'paper_evals/{checkpoint_path.split("/")[-1]}_linearprobe.txt', 'w') as f:

    #     print({
    #         'checkpoint_path': checkpoint_path,
    #         'gap_stuff': lp_stuff
    #     }, file=f)


    


    

    # evaluator.evaluate_model(clip_model, 0, 0)



    # evaluator.get_val_image_classification_acc()


    # evaluator.get_dataset_linear_probe_accuracy(clip_model, CIFAR100Processor())
    # evaluator.get_dataset_linear_probe_accuracy(clip_model, CIFAR10Processor())

    # evaluator.get_dataset_metrics(clip_model, CIFAR10Processor())




    # evaluator.evaluate_model(clip_model, 0, 0)