import sys
import os
import wandb
import random
import numpy as np
import torch

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
from clips.clip_assembler import ClipAssembler



config_cuda_device = 'cuda:5'

training_hyperparameters['temperature'] = 0.01
training_hyperparameters['encoder1_modality'] = 'image'
training_hyperparameters['encoder2_modality'] = 'text'
training_hyperparameters['same_inputs'] = False
training_hyperparameters['clip_projection_dim'] = 128
training_hyperparameters['vision_model'] = 'VIT'
training_hyperparameters['use_train_as_val'] = False
training_hyperparameters['dataset'] = ClipDatasets.MSCOCO.value
training_hyperparameters['validation_dataset_size'] = 21
training_hyperparameters['validation_batch_size'] = 21
training_hyperparameters['use_small_trainloader'] = False
training_hyperparameters['small_train_loader_dataset_size'] = 32
training_hyperparameters['seed'] = 2
training_hyperparameters['train_from_scratch'] = True
training_hyperparameters['finetune_multi_layer_projection'] = False
training_hyperparameters['cuda_device'] = config_cuda_device





def get_gap_stuff(evaluator: Evaluator):
    ranks = evaluator.get_rank()




    return {
        'mean_cosine_similarity': evaluator.get_mean_cosine_similarity(clip_model.get_temperature()),
        'linear_seperability_accuracy': evaluator.get_linear_seperability(),
        'centroid_euclidean_distance': evaluator.get_centroid_euclidean_distance(),

        'val_image_classification_acc': evaluator.get_val_image_classification_acc(return_all=True),

        'get_val_image_retrieval_acc': evaluator.get_val_image_retrieval_acc(return_all=True),

        'image_variances': ranks['image_explained_variance_ratios'],
        'text_variances': ranks['text_explained_variance_ratios'],

        'uniformity_loss': evaluator.get_mscoco_uniformity(),
        'alignment_loss': evaluator.get_mscoco_alignment(),
    
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

val_batch_cache_file = 'datasets/mscoco/val_batch_cache_mscoco_full_5k.pt'


# read file


fine_grained_correct = json.load(open('datasets/img_cap_metrics/fine_grained/filtered_correct_caps.json'))

fine_grained_incorrect = json.load(open('datasets/img_cap_metrics/fine_grained/filtered_incorrect_caps.json'))

mscoco_processor = MSCOCOProcessor()






with torch.no_grad():


    clip_model = ClipAssembler().clip_model.to(device)



    checkpoints = [
        'checkpoints/T0.01_Lit_44_finetune_I1C2E1E2_128_val_as_val_512_conceptual_captions_VIT_pretrained_POST_PAPER.pt',

        'checkpoints/T0.01_Lituniform_align_xuniform_44_finetune_I1C2E1E2_128_val_as_val_512_conceptual_captions_VIT_pretrained_POST_PAPER.pt'
    ]

    final_results = {
        'default_clip':
        {
            'average_corr_cap_similarity': None,
            'average_incorrect_cap_similarity': None,
            'num_correct': None,
            'num_incorrect': None

        },
        'CUAXU': {
            'average_corr_cap_similarity': None,
            'average_incorrect_cap_similarity': None,
            'num_correct': None,
            'num_incorrect': None
        }
    }

    for checkpoint_path in checkpoints:

        # checkpoint = torch.load(default_checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model_state_dict = checkpoint['model_state_dict']

        clip_model.load_state_dict(model_state_dict)


        run_sum_correct_caption_similarity = 0
        run_sum_incorrect_caption_similarity = 0

        n_captions = 0

        n_correct = 0

        n_incorrect = 0 

        for i, caption_data in tqdm(enumerate(fine_grained_correct)):
            image_id = caption_data['imgid']

            

            image_id = int(image_id)
            # print('image id ', image_id)

            correct_caption = caption_data['caption']

            incorrect_caption = fine_grained_incorrect[i]['caption']

            assert image_id == int(fine_grained_incorrect[i]['imgid']), "Image ids do not match"

            assert int(caption_data['question_id']) == int(fine_grained_incorrect[i]['question_id']), "Question ids do not match"

            # get image

            # image = mscoco_processor.train_dataset._load_image(image_id)
            image = mscoco_processor.val_dataset._load_image(image_id)

            preprocessed_image = mscoco_processor.image_preprocessor(image).unsqueeze(0)

            image_embedding = clip_model.encode_image(preprocessed_image)['embeds'] # shape: (1, 512)


            caption_embeddings = clip_model.encode_text([correct_caption, incorrect_caption])['embeds'] # shape: (2, 512)

            # compute similarity

            similarities = image_embedding @ caption_embeddings.T # shape: (1, 2)

            n_captions += 1

            run_sum_correct_caption_similarity += similarities[0]
            run_sum_incorrect_caption_similarity += similarities[1]
            if similarities[0] > similarities[1]:
                n_correct += 1
            else:
                n_incorrect += 1


            



            # print(f'similarities for {i}, captions: {correct_caption}, {incorrect_caption}: {similarities}')

        if 'xuniform' in checkpoint_path:
            final_results['CUAXU']['average_corr_cap_similarity'] = run_sum_correct_caption_similarity / n_captions
            final_results['CUAXU']['average_incorrect_cap_similarity'] = run_sum_incorrect_caption_similarity / n_captions
            final_results['CUAXU']['n_correct'] = n_correct
            final_results['CUAXU']['n_incorrect'] = n_incorrect

        else:

            final_results['default_clip']['average_corr_cap_similarity'] = run_sum_correct_caption_similarity / n_captions
            final_results['default_clip']['average_incorrect_cap_similarity'] = run_sum_incorrect_caption_similarity / n_captions
            final_results['default_clip']['n_correct'] = n_correct
            final_results['default_clip']['n_incorrect'] = n_incorrect



            


        






        with open(f'file_results/{checkpoint_path.split("/")[-1]}_stuff_FINAL.txt', 'w') as f:

            print(final_results, file=f)

