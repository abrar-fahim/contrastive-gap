import sys
import os
import wandb
import random
import numpy as np
import torch

from torch import Tensor

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

from torch.utils.data import Dataset, DataLoader


class CaptionsDataset(Dataset):

    def __init__(self, mode='fg') -> None:


        super().__init__()

        self.MODES = ['fg', 'diff_fg', 'negation']
        assert mode in self.MODES, 'invalid mode'



        if mode == 'fg':

            self.fine_grained_correct = json.load(open('datasets/img_cap_metrics/fine_grained/filtered_correct_caps.json'))

            self.fine_grained_incorrect = json.load(open('datasets/img_cap_metrics/fine_grained/filtered_incorrect_caps.json'))

            self.jsons = [self.fine_grained_correct, self.fine_grained_incorrect]

            self.caption_types = ['fg_correct', 'fg_incorrect']

        elif mode == 'diff_fg':


            self.diff_fine_grained_gt = json.load(open('datasets/img_cap_metrics/different_fine_grained/vqa_caps.json'))

            self.diff_fine_grained_tag = json.load(open('datasets/img_cap_metrics/different_fine_grained/image_tag_caps.json'))
        
            self.diff_fine_grained_plausible = json.load(open('datasets/img_cap_metrics/different_fine_grained/plausible_caps.json'))

            self.diff_fine_grained_random = json.load(open('datasets/img_cap_metrics/different_fine_grained/random_caps.json'))

            self.caption_types = ['diff_gf_gt', 'diff_fg_tag', 'diff_fg_plausible', 'diff_fg_random']

            self.jsons = [self.diff_fine_grained_gt, self.diff_fine_grained_tag, self.diff_fine_grained_plausible, self.diff_fine_grained_random,  ]

        elif mode == 'negation':


            self.negation = json.load(open('datasets/img_cap_metrics/negation/correct_yes_no.json'))

            self.negation_incorrect = json.load(open('datasets/img_cap_metrics/negation/negated_yes_no.json'))

            self.caption_types = ['negation_correct', 'negation_incorrect']

            self.jsons = [self.negation, self.negation_incorrect]





        self.mscoco_processor: MSCOCOProcessor = MSCOCOProcessor()

    def __len__(self):
        return len(self.fine_grained_correct)
    


    def __getitem__(self, index):
        # return the json entry at index, with the image too

        data = {
            'image_id': None,
            'question_id': None
        }

        for i, json in enumerate(self.jsons):

            if data['image_id'] == None:
                data['image_id'] = int(json[index]['imgid'])
                data['question_id'] = int(json[index]['question_id'])
            else:
                assert int(json[index]['imgid']) == data['image_id'], "Image ids do not match"
                assert int(json[index]['question_id']) == data['question_id'], "Question ids do not match"
                

            data[self.caption_types[i]] = json[index] # EX: data['fg_correct'] = {... (the json element itself)}

        data['image'] = self.mscoco_processor.val_dataset._load_image(data['image_id'])

        data['preprocessed_image'] = self.mscoco_processor.image_preprocessor(data['image'])

        # image = self.mscoco_processor.val_dataset._load_image(image_id)

        # preprocessed_image = self.mscoco_processor.image_preprocessor(image)

        return data




    def collate_fn(self, batch):
        # batch is list of dictionaries



        preprocessed_images = [b['preprocessed_image'] for b in batch]

        preprocessed_images = torch.stack(preprocessed_images, dim=0) # shape: (batch_size, 3, 224, 224) Maybe

        batch_data = {
            'preprocessed_images': preprocessed_images
        }

        for caption_type in self.caption_types:

            batch_data[caption_type] = [b[caption_type]['caption'] for b in batch]

        return batch_data


        

        # correct_captions = [b['correct_caption'] for b in batch]
        # incorrect_captions = [b['incorrect_caption'] for b in batch]

        # diff_fg_gt_captions = [b['diff_fg_gt_caption'] for b in batch]
        # diff_fg_tag_captions = [b['diff_fg_tag_caption'] for b in batch]
        # diff_fg_plausible_captions = [b['diff_fg_plausible_caption'] for b in batch]
        # diff_fg_random_captions = [b['diff_fg_random_caption'] for b in batch]

        # negation_correct_captions = [b['negation_correct_caption'] for b in batch]
        # negation_incorrect_captions = [b['negation_incorrect_caption'] for b in batch]



        # # return preprocessed_images, correct_captions, incorrect_captions
        # return {
        #     'preprocessed_images': preprocessed_images,

        #     'correct_captions':  correct_captions,
        #     'incorrect_captions': incorrect_captions,

        #     'diff_fg_gt_captions': diff_fg_gt_captions,
        #     'diff_fg_tag_captions': diff_fg_tag_captions,
        #     'diff_fg_plausible_captions': diff_fg_plausible_captions,
        #     'diff_fg_random_captions': diff_fg_random_captions,

        #     'negation_correct_captions': negation_correct_captions,
        #     'negation_incorrect_captions': negation_incorrect_captions,
        # }










        




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



captions_dataset = CaptionsDataset()




with torch.no_grad():


    clip_model = ClipAssembler().clip_model.to(device)



    checkpoints = [
        'checkpoints/T0.01_Lit_44_finetune_I1C2E1E2_128_val_as_val_512_conceptual_captions_VIT_pretrained_POST_PAPER.pt',

        'checkpoints/T0.01_Lituniform_align_xuniform_44_finetune_I1C2E1E2_128_val_as_val_512_conceptual_captions_VIT_pretrained_POST_PAPER.pt'
    ]

    # final_results = {
    #     'default_clip':
    #     {
    #         'average_corr_cap_similarity': None,
    #         'average_incorrect_cap_similarity': None,
    #         'n_correct': None,
    #         'n_incorrect': None

    #     },
    #     'CUAXU': {
    #         'average_corr_cap_similarity': None,
    #         'average_incorrect_cap_similarity': None,
    #         'n_correct': None,
    #         'n_incorrect': None
    #     }
    # }

    final_results: dict = {}


    for checkpoint_path in checkpoints:

        checkpoint_result: dict[str, dict] = {}

        # checkpoint = torch.load(default_checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model_state_dict = checkpoint['model_state_dict']

        clip_model.load_state_dict(model_state_dict)


        # run_sum_correct_caption_similarity = 0
        # run_sum_incorrect_caption_similarity = 0

        # run_sum_diff_fg_gt_similarity = 0
        # run_sum_diff_fg_tag_similarity = 0
        # run_sum_diff_fg_plausible_similarity = 0
        # run_sum_diff_fg_random_similarity = 0
        
        # run_sum_negation_correct_similarity = 0
        # run_sum_negation_incorrect_similarity = 0

        # n_captions = 0


        # n_correct_fg = 0
        # n_incorrect_fg = 0 

        # n_negation_correct = 0 
        # n_negation_incorrect = 0 


        

        captions_dataloader = DataLoader(captions_dataset, batch_size=1024, shuffle=False, num_workers=16, collate_fn=captions_dataset.collate_fn)

        counts: dict[str, float] = {}

        similarity_run_sums: dict[str, float] = {}

        n_captions = 0

        for batched_data in tqdm(captions_dataloader):

            preprocessed_images = batched_data['preprocessed_images']

            image_embeddings = clip_model.encode_image(preprocessed_images)['embeds'] # shape: (n,  512)

            caption_embeddings: dict[str, Tensor] = {}

            for caption_type in captions_dataset.caption_types:

                caption_embeddings[caption_type] = clip_model.encode_text(batched_data[caption_type])['embeds']

            similarities: dict[str, Tensor] = {}

            for caption_type in captions_dataset.caption_types:

                similarities[caption_type] = (image_embeddings @ caption_embeddings[caption_type].T).diag().flatten()

            

            if len(captions_dataset.caption_types) == 2:

                for i, caption_type in enumerate(captions_dataset.caption_types):

                    if caption_type in counts:

                        counts[caption_type] += (similarities[caption_type] > similarities[captions_dataset.caption_types[i-1]]).count_nonzero()
                    else:
                         counts[caption_type] = (similarities[caption_type] > similarities[captions_dataset.caption_types[i-1]]).count_nonzero()

            n_captions += preprocessed_images.shape[0]  # same as batch size

            for caption_type in captions_dataset.caption_types:
                
                if caption_type in similarity_run_sums:
                    similarity_run_sums[caption_type] += similarities[caption_type].sum()
                else:
                    similarity_run_sums[caption_type] = similarities[caption_type].sum()









            # correct_captions = batched_data['correct_captions']
            # incorrect_captions = batched_data['incorrect_captions']

            # diff_fg_gt_captions = batched_data['diff_fg_gt_captions']
            # diff_fg_tag_captions = batched_data['diff_fg_tag_captions']
            # diff_fg_plausible_captions = batched_data['diff_fg_plausible_captions']
            # diff_fg_random_captions = batched_data['diff_fg_random_captions']

            # negation_correct_captions = batched_data['negation_correct_captions']
            # negation_incorrect_captions = batched_data['negation_incorrect_captions']

            # image_embeddings = clip_model.encode_image(preprocessed_images)['embeds'] # shape: (n,  512)

            


            # correct_caption_embeddings = clip_model.encode_text(correct_captions)['embeds']
            # incorrect_caption_embeddings = clip_model.encode_text(incorrect_captions)['embeds']

            # diff_fg_gt_caption_embeddings = clip_model.encode_text(diff_fg_gt_captions)['embeds']
            # diff_fg_tag_caption_embeddings = clip_model.encode_text(diff_fg_tag_captions)['embeds']
            # diff_fg_plausible_caption_embeddings = clip_model.encode_text(diff_fg_plausible_captions)['embeds']
            # diff_fg_random_caption_embeddings = clip_model.encode_text(diff_fg_random_captions)['embeds']

            # negation_correct_caption_embeddings = clip_model.encode_text(negation_correct_captions)['embeds']
            # negation_incorrect_caption_embeddings = clip_model.encode_text(negation_incorrect_captions)['embeds']




            # compute similarity

            # correct_similarities = image_embeddings @ correct_caption_embeddings.T # shape: (n, n)
            # incorrect_similarities = image_embeddings @ incorrect_caption_embeddings.T # shape: (n, n)

            # diff_fg_gt_similarities = image_embeddings @ diff_fg_gt_caption_embeddings.T # shape: (n, n)
            # diff_fg_tag_similarities = image_embeddings @ diff_fg_tag_caption_embeddings.T # shape: (n, n)
            # diff_fg_plausible_similarities = image_embeddings @ diff_fg_plausible_caption_embeddings.T # shape: (n, n)
            # diff_fg_random_similarities = image_embeddings @ diff_fg_random_caption_embeddings.T # shape: (n, n)

            # negation_correct_similarities = image_embeddings @ negation_correct_caption_embeddings.T # shape: (n, n)
            # negation_incorrect_similarities = image_embeddings @ negation_incorrect_caption_embeddings.T # shape: (n, n)

            

            # diagonals of the matrices is what we need 

            # correct_similarities = correct_similarities.diag().flatten()
            # incorrect_similarities = incorrect_similarities.diag().flatten()

            # diff_fg_gt_similarities = diff_fg_gt_similarities.diag().flatten()
            # diff_fg_tag_similarities = diff_fg_tag_similarities.diag().flatten()
            # diff_fg_plausible_similarities = diff_fg_plausible_similarities.diag().flatten()
            # diff_fg_random_similarities = diff_fg_random_similarities.diag().flatten()
            
            # negation_correct_similarities = negation_correct_similarities.diag().flatten()
            # negation_incorrect_similarities = negation_incorrect_similarities.diag().flatten()

            # n_correct_fg += (correct_similarities > incorrect_similarities).count_nonzero()
            # n_incorrect_fg += (correct_similarities < incorrect_similarities).count_nonzero()

            # n_negation_correct += (negation_correct_similarities < negation_incorrect_similarities).count_nonzero()
            # n_negation_incorrect += (negation_correct_similarities < negation_incorrect_similarities).count_nonzero()


            # n_captions += len(correct_captions) # same as batch size
            # run_sum_correct_caption_similarity += correct_similarities.sum()
            # run_sum_incorrect_caption_similarity += incorrect_similarities.sum()

            # run_sum_diff_fg_gt_similarity += diff_fg_gt_similarities.sum()
            # run_sum_diff_fg_tag_similarity += diff_fg_tag_similarities.sum()
            # run_sum_diff_fg_plausible_similarity += diff_fg_plausible_similarities.sum()
            # run_sum_diff_fg_random_similarity += diff_fg_random_similarities.sum()

            # run_sum_negation_correct_similarity += negation_correct_similarities.sum()
            # run_sum_negation_incorrect_similarity += negation_incorrect_similarities.sum()

            



            # print(f'similarities for {i}, captions: {correct_caption}, {incorrect_caption}: {similarities}')
                    

        for caption_type in captions_dataset.caption_types:
            checkpoint_result[caption_type] = {}
            checkpoint_result[caption_type]['average'] = similarity_run_sums[caption_type] / n_captions 

            if caption_type in counts:

                checkpoint_result[caption_type]['counts'] = counts[caption_type] / n_captions 

            

        checkpoint_name = 'CUAXU' if 'xuniform' in checkpoint_path else 'clip_default'

        final_results[checkpoint_name] = checkpoint_result

        # if 'xuniform' in checkpoint_path:
            
        #     for caption_type in captions_dataset.caption_types:
        #         final_results['CUAXU'][caption_type]



        #     final_results['CUAXU']['average_correct_cap_similarity'] = run_sum_correct_caption_similarity / n_captions
        #     final_results['CUAXU']['average_incorrect_cap_similarity'] = run_sum_incorrect_caption_similarity / n_captions

        #     final_results['CUAXU']['average_diff_fg_gt_similarity'] = run_sum_diff_fg_gt_similarity / n_captions
        #     final_results['CUAXU']['average_diff_fg_tag_similarity'] = run_sum_diff_fg_tag_similarity / n_captions
        #     final_results['CUAXU']['average_diff_fg_plausible_similarity'] = run_sum_diff_fg_plausible_similarity / n_captions
        #     final_results['CUAXU']['average_diff_fg_random_similarity'] = run_sum_diff_fg_random_similarity / n_captions

        #     final_results['CUAXU']['average_negation_correct_similarity'] = run_sum_negation_correct_similarity / n_captions
        #     final_results['CUAXU']['average_negation_incorrect_similarity'] = run_sum_negation_incorrect_similarity / n_captions


        #     final_results['CUAXU']['n_correct_fg'] = n_correct_fg / n_captions
        #     final_results['CUAXU']['n_incorrect_fg'] = n_incorrect_fg / n_captions

        #     final_results['CUAXU']['n_negation_correct'] = n_negation_correct / n_captions
        #     final_results['CUAXU']['n_negation_incorrect'] = n_negation_incorrect / n_captions



        # else:

        #     final_results['default_clip']['average_corr_cap_similarity'] = run_sum_correct_caption_similarity / n_captions
        #     final_results['default_clip']['average_incorrect_cap_similarity'] = run_sum_incorrect_caption_similarity / n_captions

        #     final_results['default_clip']['average_diff_fg_gt_similarity'] = run_sum_diff_fg_gt_similarity / n_captions
        #     final_results['default_clip']['average_diff_fg_tag_similarity'] = run_sum_diff_fg_tag_similarity / n_captions
        #     final_results['default_clip']['average_diff_fg_plausible_similarity'] = run_sum_diff_fg_plausible_similarity / n_captions
        #     final_results['default_clip']['average_diff_fg_random_similarity'] = run_sum_diff_fg_random_similarity / n_captions

        #     final_results['default_clip']['average_negation_correct_similarity'] = run_sum_negation_correct_similarity / n_captions
        #     final_results['default_clip']['average_negation_incorrect_similarity'] = run_sum_negation_incorrect_similarity / n_captions


        #     final_results['default_clip']['n_correct_fg'] = n_correct_fg / n_captions
        #     final_results['default_clip']['n_incorrect_fg'] = n_incorrect_fg / n_captions

        #     final_results['default_clip']['n_negation_correct'] = n_negation_correct / n_captions
        #     final_results['default_clip']['n_negation_incorrect'] = n_negation_incorrect / n_captions


    with open(f'file_results/{checkpoint_path.split("/")[-1]}_stuff_FINAL.txt', 'w') as f:

        print(final_results, file=f)

