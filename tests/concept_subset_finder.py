import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import training_hyperparameters
from dataset_processors.mscoco_processor import MSCOCOProcessor
from tqdm import tqdm
import torch
import random
from clips.hf_clip import HFClip
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import pickle

# hypers
max_target_captions = 30 # keep top k captions
pos_to_get = ['NNP', 'NNS', 'NN', 'NNPS']
n_captions_to_search = 2048 # number of captions to search for target image
n_input_captions = 10 # number of input captions to consider
# dataset_path = 'datasets/mscoco/linear_eval_dataset/consistency_dataset'
dataset_path = 'datasets/mscoco/linear_eval_dataset/linearity_dataset'


def get_target_image(input_caption, dataset_processor, clip_model):
    '''
    - Need to use default clip_model here
    - Input: input_caption
    - Return: target_image, target_image_caption, target_image_embedding, target_image_caption_embedding
    - Functions of this form always:
        - Alter the input_caption in some way
        - Use altered caption to search for image = target_image
        - Return stuff for that target_image
    '''

    # Find target_caption that has similar structure as caption only with subject changed

    # get subject from caption
    global pos_to_get, max_target_captions, n_captions_to_search
    pos_tags = pos_tag(word_tokenize(input_caption))

    input_caption_subjects = [word.lower() for word, pos in pos_tags if pos in pos_to_get]
    # input_caption_subject = input_caption_subjects[0].lower()

    # get unique subjects
    input_caption_subjects = list(set(input_caption_subjects))

    print('input caption ', input_caption)

    print('input caption subjects ', input_caption_subjects)

    # print('all ', pos_tags)


    val_dataloader = dataset_processor.val_dataloader
    dataset_processor.return_org_imgs_collate_fn = True
    top_k_captions = []
    top_k_images = []
    top_k_image_embeddings = []
    top_k_image_caption_embeddings = []

    top_k_preprocessed_images = []
    top_k_tokenized_captions = []
    
    top_k_bleu_scores = []

    top_k_subjects = []

    n_batches_to_search = n_captions_to_search // training_hyperparameters['validation_batch_size'] + 1




    for batch_i, batch in tqdm(enumerate(val_dataloader)):

        if batch_i >= n_batches_to_search or len(top_k_captions) >= max_target_captions:
            break

        # print('batch_i ', batch_i, ' / ', n_batches_to_search)

        # print('top k captions len ', len(top_k_captions), ' ////')
        

        preprocessed_images, tokenized_captions, images, captions = batch


        batch_outputs = clip_model(preprocessed_images, tokenized_captions, return_all=True)

        # normalize image and text embeds
        normalized_image_embeds = batch_outputs.image_embeds / batch_outputs.image_embeds.norm(dim=-1, keepdim=True)
        normalized_text_embeds = batch_outputs.text_embeds / batch_outputs.text_embeds.norm(dim=-1, keepdim=True)



        for index, caption in enumerate(captions):

            if index >= n_captions_to_search or len(top_k_captions) >= max_target_captions:
                break
            # get subject from caption
            pos_tags = pos_tag(word_tokenize(caption))
            caption_subjects = [word.lower() for word, pos in pos_tags if pos in pos_to_get]

            # get unique subjects
            caption_subjects = list(set(caption_subjects))

            if len(caption_subjects) == 0:
                continue

            append = False
            
            if 'linearity' in dataset_path:

                # check if caption subjects contain any of the input caption subjects
                if any(subject in caption_subjects for subject in input_caption_subjects) and (len(caption_subjects) != len(input_caption_subjects)): # this is linearity, excluding consistency to force image embeddings out of image space
                    append = True

            elif 'consistency' in dataset_path:
                # check if there is exactly same number of subjects in caption and input caption and atleast one subject is same
                if len(caption_subjects) == len(input_caption_subjects) and any(subject in caption_subjects for subject in input_caption_subjects):
                    append = True
                
            if append:
                # add to lists
                top_k_captions.append(caption)
                top_k_images.append(images[index])
                top_k_image_embeddings.append(normalized_image_embeds[index])
                top_k_image_caption_embeddings.append(normalized_text_embeds[index])
                top_k_subjects.append(caption_subjects)
                top_k_preprocessed_images.append(preprocessed_images[index])
                top_k_tokenized_captions.append(tokenized_captions[index])

                pass


    return {
        'top_images': top_k_images,
        'top_captions': top_k_captions,
        'top_image_embeddings': top_k_image_embeddings,
        'top_image_caption_embeddings': top_k_image_caption_embeddings,
        'top_subjects': top_k_subjects,
        'input_caption_subjects': input_caption_subjects,
        'top_preprocessed_images': top_k_preprocessed_images,
        'top_tokenized_captions': top_k_tokenized_captions,
    }




dataset_processor = MSCOCOProcessor(return_org_imgs_collate_fn=True)

clip_model = HFClip()

# read all subjects from text file
# with open('unique_subjects.txt', 'r') as f:
#     all_subjects = f.readlines()

# evaluate_concept_arrangement(dataset_processor, clip_model,all_subjects )




# set seed
torch.manual_seed(42)


dataset_processor.return_org_imgs_collate_fn = True
# get first caption in dataset as input caption for now
# input_caption = next(iter(dataset_processor.train_dataloader))[3][0]

# print('input caption ', input_caption)

# target_image_outputs = get_target_image(input_caption, dataset_processor, clip_model)

# print('input caption ', input_caption)

# print('top captions ', target_image_outputs['top_captions'])

# print('top subjects ', target_image_outputs['top_subjects'])



'''
What I need to save
- input caption
- input image
- input caption subjects
- top k captions
- subjects for each of the top k captions
- top k images
- top k image embeddings
'''

n_batches_to_consider = n_input_captions // training_hyperparameters['validation_batch_size'] + 1

with torch.no_grad():

    for batch_i, batch in tqdm(enumerate(dataset_processor.val_dataloader)):

        if batch_i >= n_batches_to_consider:
            break
        preprocessed_images, tokenized_captions, images, captions = batch

        batch_outputs = clip_model(preprocessed_images, tokenized_captions, return_all=True)

        # normalize image and text embeds
        normalized_image_embeds = batch_outputs.image_embeds / batch_outputs.image_embeds.norm(dim=-1, keepdim=True)
        normalized_text_embeds = batch_outputs.text_embeds / batch_outputs.text_embeds.norm(dim=-1, keepdim=True)

        for index, caption in enumerate(captions):

            if index >= n_input_captions:
                break
            # print magnitude of normalized image embeds
            # print('magnitude of normalized image embeds ', normalized_image_embeds[index].norm())

            print('index ', index , '/ ', len(captions))
            target_image_outputs = get_target_image(caption, dataset_processor, clip_model)

            print('input caption ', caption)
            print('input caption subjects ', target_image_outputs['input_caption_subjects'])
            print('top captions ', target_image_outputs['top_captions'])
            print('top subjects ', target_image_outputs['top_subjects'])

            


            # save stuff to file
            filename = dataset_path
            data = {
                'input_caption': caption,
                'input_tokenized_caption': tokenized_captions[index],
                'input_image': images[index],
                'input_preprocessed_image': preprocessed_images[index],
                'input_caption': captions[index],
                'input_caption_subjects': target_image_outputs['input_caption_subjects'],
                'top_captions': target_image_outputs['top_captions'],
                'top_images': target_image_outputs['top_images'],
                'top_preprocessed_images': target_image_outputs['top_preprocessed_images'],
                'top_tokenized_captions': target_image_outputs['top_tokenized_captions'],
                # 'top_image_embeddings': target_image_outputs['top_image_embeddings'],
                # 'top_image_caption_embeddings': target_image_outputs['top_image_caption_embeddings'],
                'top_subjects': target_image_outputs['top_subjects'],


            }
            with open(filename, 'ab+') as fp:
                pickle.dump(data, fp)

        
    

