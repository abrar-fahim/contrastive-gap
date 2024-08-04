# conceptual captions streaming test

from matplotlib import pyplot as plt


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

from datasets import load_dataset
auth_token = "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"  # Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token
winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]

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



config_cuda_device = 'cuda:5'

training_hyperparameters['temperature'] = 0.01
training_hyperparameters['encoder1_modality'] = 'image'
training_hyperparameters['encoder2_modality'] = 'text'
training_hyperparameters['same_inputs'] = False
training_hyperparameters['clip_projection_dim'] = 128
training_hyperparameters['vision_model'] = 'VIT'
training_hyperparameters['use_train_as_val'] = False
training_hyperparameters['dataset'] = ClipDatasets.CONCEPTUAL_CAPTIONS.value
# training_hyperparameters['validation_dataset_size'] = 16000
# training_hyperparameters['validation_batch_size'] = 16000
training_hyperparameters['validation_dataset_size'] = 24
training_hyperparameters['validation_batch_size'] = 24
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



def get_gap_stuff(evaluator: Evaluator):
    ranks = evaluator.get_rank()
    n_s_buckets = 8

    # group S values into buckets, and get mean of each bucket
    # S shape: (clip_projection_dim, )

    image_S_buckets = torch.chunk(ranks['image_S'], n_s_buckets) 
    text_S_buckets = torch.chunk(ranks['text_S'], n_s_buckets)

    image_pca_variance_ratio_buckets = torch.chunk(ranks['image_explained_variance_ratios'], n_s_buckets)
    text_pca_variance_ratio_buckets = torch.chunk(ranks['text_explained_variance_ratios'], n_s_buckets)

    if len(image_S_buckets) < n_s_buckets:
        image_S_buckets += tuple([torch.tensor([0.0]) for _ in range(n_s_buckets - len(image_S_buckets))])

        image_pca_variance_ratio_buckets += tuple([torch.tensor([0.0]) for _ in range(n_s_buckets - len(image_pca_variance_ratio_buckets))])
    
    if len(text_S_buckets) < n_s_buckets:
        text_S_buckets += tuple([torch.tensor([0.0]) for _ in range(n_s_buckets - len(text_S_buckets))])

        text_pca_variance_ratio_buckets += tuple([torch.tensor([0.0]) for _ in range(n_s_buckets - len(text_pca_variance_ratio_buckets))])

    image_S_bucket_means = [torch.mean(bucket) for bucket in image_S_buckets]
    text_S_bucket_means = [torch.mean(bucket) for bucket in text_S_buckets]

    image_pca_variance_ratio_bucket_sums = [torch.sum(bucket) for bucket in image_pca_variance_ratio_buckets]

    text_pca_variance_ratio_bucket_sums = [torch.sum(bucket) for bucket in text_pca_variance_ratio_buckets]


    return {
        'mean_cosine_similarity': evaluator.get_mean_cosine_similarity(clip_model.get_temperature()),
        'linear_seperability_accuracy': evaluator.get_linear_seperability(),
        'centroid_euclidean_distance': evaluator.get_centroid_euclidean_distance(),

        'val_image_classification_acc': evaluator.get_val_image_classification_acc(return_all=True),

        'get_val_image_retrieval_acc': evaluator.get_val_image_retrieval_acc(return_all=True),
        
        'image_variance0': image_pca_variance_ratio_bucket_sums[0],
        'image_variance1': image_pca_variance_ratio_bucket_sums[1],
        'image_variance2': image_pca_variance_ratio_bucket_sums[2],
        'image_variance3': image_pca_variance_ratio_bucket_sums[3],
        'image_variance4': image_pca_variance_ratio_bucket_sums[4],
        'image_variance5': image_pca_variance_ratio_bucket_sums[5],
        'image_variance6': image_pca_variance_ratio_bucket_sums[6],
        'image_variance7': image_pca_variance_ratio_bucket_sums[7],

        'text_variance0': text_pca_variance_ratio_bucket_sums[0],
        'text_variance1': text_pca_variance_ratio_bucket_sums[1],
        'text_variance2': text_pca_variance_ratio_bucket_sums[2],
        'text_variance3': text_pca_variance_ratio_bucket_sums[3],
        'text_variance4': text_pca_variance_ratio_bucket_sums[4],
        'text_variance5': text_pca_variance_ratio_bucket_sums[5],
        'text_variance6': text_pca_variance_ratio_bucket_sums[6],
        'text_variance7': text_pca_variance_ratio_bucket_sums[7],
    }

def get_zs_stuff(clip_model, evaluator: Evaluator):

    return {
        'imagenet_zs_acc': evaluator.get_dataset_zero_shot_acc(clip_model, ImageNet1k()),

        'dtd_zs_acc': evaluator.get_dataset_zero_shot_acc(clip_model, DTDProcessor()),
        'caltech101_zs_acc': evaluator.get_dataset_zero_shot_acc(clip_model, Caltech101Processor()),
        
        'cifar10_zs_acc': evaluator.get_dataset_zero_shot_acc(clip_model, CIFAR10Processor()),
        'cifar100_zs_acc': evaluator.get_dataset_zero_shot_acc(clip_model, CIFAR100Processor()),




        
      
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


with torch.no_grad():


    # set seed
    torch.manual_seed(wandb.config['seed'])
    random.seed(wandb.config['seed'])
    np.random.seed(wandb.config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"



    device = torch.device(config_cuda_device if torch.cuda.is_available() else "cpu")

    # val_batch_cache_file = 'datasets/mscoco/val_batch_cache_mscoco_full_5k.pt'

    # evaluator = Evaluator(MSCOCOProcessor(), val_batch_cache_file)
    evaluator = Evaluator(MSCOCOProcessor())


    clip_model = ClipAssembler().clip_model.to(device)


    #32D

    # checkpoint_path = 'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt'

    # checkpoint_path = 'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt'

    # checkpoint_path = 'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt'



    # 64D
    # checkpoint_path = 'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt'

    # checkpoint_path = 'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt'

    # checkpoint_path = 'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt'


    # 128D

    # checkpoint_path = 'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt'

    # checkpoint_path = 'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt'

    checkpoint_path = 'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt'



    # checkpoint = torch.load(default_checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_state_dict = checkpoint['model_state_dict']

    clip_model.load_state_dict(model_state_dict)

    # clip_model.half()

    from tqdm import tqdm
    winoground_clip_scores = []
    for example in tqdm(winoground):

        input_i0 = clip_model.encoder1.preprocessor(example["image_0"].convert("RGB")).unsqueeze(0)

        input_i1 = clip_model.encoder1.preprocessor(example["image_1"].convert("RGB")).unsqueeze(0)

        input_c0 = example["caption_0"]
        input_c1 = example["caption_1"]




        # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
        # Note that we could run this example through CLIP as a batch, but I want to drive the point home that we get four independent image-caption scores for each example
        # input_c0_i0 = clip_processor(text=[example["caption_0"]], images=[example["image_0"].convert("RGB")], return_tensors="pt")
        # input_c1_i0 = clip_processor(text=[example["caption_1"]], images=[example["image_0"].convert("RGB")], return_tensors="pt")
        # input_c0_i1 = clip_processor(text=[example["caption_0"]], images=[example["image_1"].convert("RGB")], return_tensors="pt")
        # input_c1_i1 = clip_processor(text=[example["caption_1"]], images=[example["image_1"].convert("RGB")], return_tensors="pt")

        output_c0_i0 = clip_model(input_i0, [input_c0], output_loss=False, return_all=True)
        output_c1_i0 = clip_model(input_i0, [input_c1], output_loss=False, return_all=True)
        output_c0_i1 = clip_model(input_i1, [input_c0], output_loss=False, return_all=True)
        output_c1_i1 = clip_model(input_i1, [input_c1], output_loss=False, return_all=True)


        # output_c0_i0 = clip_model(**input_c0_i0)
        # output_c1_i0 = clip_model(**input_c1_i0)
        # output_c0_i1 = clip_model(**input_c0_i1)
        # output_c1_i1 = clip_model(**input_c1_i1)
        clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
        clip_score_c1_i0 = output_c1_i0.logits_per_image.item()
        clip_score_c0_i1 = output_c0_i1.logits_per_image.item()
        clip_score_c1_i1 = output_c1_i1.logits_per_image.item()
        winoground_clip_scores.append({"id" : example["id"], "c0_i0": clip_score_c0_i0, "c0_i1": clip_score_c0_i1, "c1_i0": clip_score_c1_i0, "c1_i1": clip_score_c1_i1})

    def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(result):
        return image_correct(result) and text_correct(result)

    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    for result in winoground_clip_scores:
        text_correct_count += 1 if text_correct(result) else 0
        image_correct_count += 1 if image_correct(result) else 0
        group_correct_count += 1 if group_correct(result) else 0

    denominator = len(winoground_clip_scores)
    print("text score:", text_correct_count/denominator)
    print("image score:", image_correct_count/denominator)
    print("group score:", group_correct_count/denominator)

    exit()





    evaluator.set_val_outputs(clip_model, output_loss=False)

    gap_stuff = get_gap_stuff(evaluator)



    # write both checkpoint file and gap stuff to same file
    

    # write checkpoint path to file
    # with open(f'paper_evals/{checkpoint_path.split("/")[-1]}_stuff.txt', 'w') as f:

    #     print({
    #         'checkpoint_path': checkpoint_path,
    #         'gap_stuff': gap_stuff
    #     }, file=f)

    
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