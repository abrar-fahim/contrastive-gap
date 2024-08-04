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
from dataset_processors.conceptual_captions_processor import ConceptualCaptionsProcessor
from dataset_processors.cifar100_processor import CIFAR100Processor
from dataset_processors.cifar10_processor import CIFAR10Processor
from dataset_processors.imagenet_processor import ImageNet1k
from dataset_processors.caltech101_processor import Caltech101Processor
from dataset_processors.dtd_processor import DTDProcessor
from dataset_processors.food101_processor import Food101Processor
from clips.clip_assembler import ClipAssembler


config_cuda_device = 'cuda:5'

training_hyperparameters['temperature'] = 0.01
training_hyperparameters['encoder1_modality'] = 'image'
training_hyperparameters['encoder2_modality'] = 'text'
training_hyperparameters['same_inputs'] = False
training_hyperparameters['clip_projection_dim'] = 128
training_hyperparameters['vision_model'] = 'VIT'
training_hyperparameters['use_train_as_val'] = False
training_hyperparameters['dataset'] = ClipDatasets.CONCEPTUAL_CAPTIONS.value
training_hyperparameters['validation_dataset_size'] = 16000
training_hyperparameters['validation_batch_size'] = 16000
training_hyperparameters['use_small_trainloader'] = True
training_hyperparameters['small_train_loader_dataset_size'] = 32
training_hyperparameters['seed'] = 2
training_hyperparameters['train_from_scratch'] = False


training_hyperparameters['continue_from_checkpoint'] = False
training_hyperparameters['train_from_pretrained'] = True
training_hyperparameters['finetune_clip_backbone'] = True
training_hyperparameters['finetune_multi_layer_projection'] = False

training_hyperparameters['cuda_device'] = config_cuda_device
training_hyperparameters['num_workers'] = 12





d32_checkpoints = [
    'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt',
    'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt'
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
    'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt'
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
    'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt'
    'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt',

    'checkpoints/T0.01_Lit_24_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
    'checkpoints/T0.01_Lituniform_align_24_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
    'checkpoints/T0.01_Lituniform_align_xuniform_24_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',

    'checkpoints/T0.01_Lit_44_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
    'checkpoints/T0.01_Lituniform_align_44_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
    'checkpoints/T0.01_Lituniform_align_xuniform_44_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt'

]



def get_gap_stuff(evaluator: Evaluator):
    # ranks = evaluator.get_rank(linalg=False)
   

    return {
        'mean_cosine_similarity': evaluator.get_mean_cosine_similarity(clip_model.get_temperature()),
        'linear_seperability_accuracy': evaluator.get_linear_seperability(),
        'centroid_euclidean_distance': evaluator.get_centroid_euclidean_distance(),

        'val_image_classification_acc': evaluator.get_val_image_classification_acc(return_all=True),

        'get_val_image_retrieval_acc': evaluator.get_val_image_retrieval_acc(return_all=True),

        # 'image_variances': ranks['image_explained_variance_ratios'],
        # 'text_variances': ranks['text_explained_variance_ratios'],

        # 'uniformity_loss': evaluator.get_mscoco_uniformity(),
        # 'alignment_loss': evaluator.get_mscoco_alignment(),
        
    }

def get_zs_stuff(clip_model, evaluator: Evaluator):

    return {

        'food101_zs_acc': evaluator.get_dataset_zero_shot_acc(clip_model, Food101Processor()),
        # 'imagenet_zs_acc': evaluator.get_dataset_zero_shot_acc(clip_model, ImageNet1k()),
        
        # 'dtd_zs_acc': evaluator.get_dataset_zero_shot_acc(clip_model, DTDProcessor()),
        
        # 'caltech101_zs_acc': evaluator.get_dataset_zero_shot_acc(clip_model, Caltech101Processor()),
        
        
        # 'cifar10_zs_acc': evaluator.get_dataset_zero_shot_acc(clip_model, CIFAR10Processor()),
        
        # 'cifar100_zs_acc': evaluator.get_dataset_zero_shot_acc(clip_model, CIFAR100Processor()),



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


wandb.init(config=training_hyperparameters)


# set seed
torch.manual_seed(wandb.config['seed'])
random.seed(wandb.config['seed'])
np.random.seed(wandb.config['seed'])
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"



device = torch.device(config_cuda_device if torch.cuda.is_available() else "cpu")

val_batch_cache_file = 'datasets/conceptual_captions/val_batch_cache_T0.01_Lit_2_scratch_I1C2E1E2_64_val_as_val_16000_conceptual_captions_VIT_pretrained_POST_PAPER.pt'

# mscoco_evaluator = Evaluator(MSCOCOProcessor(), val_batch_cache_file)

with torch.no_grad():


    # evaluator = Evaluator(MSCOCOProcessor(), val_batch_cache_file)
    evaluator = Evaluator(ConceptualCaptionsProcessor(), val_batch_cache_file, load_train_dataset=False)
    # evaluator = Evaluator(MSCOCOProcessor())


    clip_model = ClipAssembler().clip_model.to(device)


    #32D

    # cps_32 = [
    #     'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt',
    #     'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt',
    #     'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt'


    # ]
    # cps_64 = [
    #     'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt',
    #     'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt',
    #     'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt'


    # ]



    concaps_paths = [
        'checkpoints/T0.01_Lit_44_finetune_I1C2E1E2_128_val_as_val_512_conceptual_captions_VIT_pretrained_POST_PAPER.pt',
        'checkpoints/T0.01_Lituniform_align_xuniform_44_finetune_I1C2E1E2_128_val_as_val_512_conceptual_captions_VIT_pretrained_POST_PAPER.pt'
    ]
    # for checkpoint_path in cps_64:
    for checkpoint_path in concaps_paths:

        # checkpoint = torch.load(default_checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model_state_dict = checkpoint['model_state_dict']

        clip_model.load_state_dict(model_state_dict)

        clip_model.half()

        evaluator.set_val_outputs(clip_model, output_loss=False)

        evaluator.set_outputs_to_use('val')



        gap_stuff = get_gap_stuff(evaluator)

        # zs_stuff = get_zs_stuff(clip_model, evaluator)
        with open(f'paper_evals/gap_stuff_{checkpoint_path.split("/")[-1]}_stuff_POST_PAPER.txt', 'w') as f:

            print({
                'checkpoint_path': checkpoint_path,
                'gap_stuff': gap_stuff
                # 'zs_stuff': zs_stuff
            }, file=f)




    # write both checkpoint file and gap stuff to same file
    

    # write checkpoint path to file
  
    
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