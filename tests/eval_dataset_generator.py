import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processors.mscoco_processor import MSCOCOProcessor
from tqdm import tqdm
import torch
import random
from clips.hf_clip import HFClip
from src.utils import evaluate_concept_arrangement
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.translate.bleu_score import sentence_bleu
import pickle

def get_target_image(input_caption, dataset_processor, clip_model):
    '''
    - Input: input_caption
    - Return: target_image, target_image_caption, target_image_embedding, target_image_caption_embedding
    - Functions of this form always:
        - Alter the input_caption in some way
        - Use altered caption to search for image = target_image
        - Return stuff for that target_image
    '''

    # Find target_caption that has similar structure as caption only with subject changed

    # get subject from caption
    pos_to_get = ['NN', 'NNP']
    pos_tags = pos_tag(word_tokenize(input_caption))

    input_caption_subjects = [word.lower() for word, pos in pos_tags if pos in pos_to_get]
    # input_caption_subject = input_caption_subjects[0].lower()

    input_caption_without_subject = input_caption

    print('input caption subjects ', input_caption_subjects)

    for subject in input_caption_subjects:
        # remove subject from caption
        input_caption_without_subject = input_caption_without_subject.replace(subject, '')

    # remove subject from caption
    # input_caption_without_subject = input_caption.replace(input_caption_subject, '')

    print('input caption without subject ', input_caption_without_subject)


    train_dataloader = dataset_processor.train_dataloader
    train_dataloader.return_org_imgs_collate_fn = True

    k = 10 # keep top k captions
    top_k_captions = []
    top_k_images = []
    top_k_image_embeddings = []
    top_k_image_caption_embeddings = []
    
    top_k_bleu_scores = []

    top_k_subjects = []


    for batch in tqdm(train_dataloader):
        preprocessed_images, tokenized_captions, images, captions = batch



        for index, caption in enumerate(captions):
            # get subject from caption
            pos_to_get = ['NNP', 'NN']
            pos_tags = pos_tag(word_tokenize(caption))

            caption_subjects = [word for word, pos in pos_tags if pos in pos_to_get]

            if len(caption_subjects) == 0:
                continue
            # caption_subject = caption_subjects[0].lower()


            # remove subject from caption
            # caption_without_subject = caption.replace(caption_subject, '')

            caption_without_subject = caption

            for subject in caption_subjects:
                # remove subject from caption
                caption_without_subject = caption_without_subject.replace(subject, '')
    

            # check if bleu score between caption_without_subject and input_caption_without_subject is high
            bleu_score = sentence_bleu([input_caption_without_subject], caption_without_subject)

            add_image = False

            if len(top_k_bleu_scores) < k:
                add_image = True
            else:
                min_bleu_score_index = torch.argmin(torch.tensor(top_k_bleu_scores))
                if bleu_score > top_k_bleu_scores[min_bleu_score_index]:
                    add_image = True

            if add_image:
                # get image corresponding to caption
                image = images[index]
                image_embedding = clip_model.encode_image(preprocessed_images[index].unsqueeze(0))
                image_caption_embedding = clip_model.encode_text(clip_model.tokenize_captions([caption]))

                # normalize
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                image_caption_embedding = image_caption_embedding / image_caption_embedding.norm(dim=-1, keepdim=True)

                if len(top_k_bleu_scores) < k:
                    top_k_captions.append(caption)
                    top_k_images.append(image)
                    top_k_image_embeddings.append(image_embedding)
                    top_k_image_caption_embeddings.append(image_caption_embedding)
                    top_k_bleu_scores.append(bleu_score)
                    top_k_subjects.append(caption_subjects)

                else:
                    top_k_captions[min_bleu_score_index] = caption
                    top_k_images[min_bleu_score_index] = image
                    top_k_image_embeddings[min_bleu_score_index] = image_embedding
                    top_k_image_caption_embeddings[min_bleu_score_index] = image_caption_embedding
                    top_k_bleu_scores[min_bleu_score_index] = bleu_score
                    top_k_subjects[min_bleu_score_index] = caption_subjects
    return top_k_images, top_k_captions, top_k_image_embeddings, top_k_image_caption_embeddings, top_k_subjects




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
input_caption = next(iter(dataset_processor.train_dataloader))[3][0]

print('input caption ', input_caption)

top_images, top_captions, _, _, top_k_subjects = get_target_image(input_caption, dataset_processor, clip_model)

print('input caption ', input_caption)

print('top captions ', top_captions)

print('top subjects ', top_k_subjects)

'''
What I need to save
- input caption
- input caption subjects
- top k captions
- subjects for each of the top k captions
- top k images
- top k image embeddings
'''

n_batches = 0

for batch in tqdm(dataset_processor.train_dataloader):

    if n_batches > 0: #doing just one batch for now
        break
    preprocessed_images, tokenized_captions, images, captions = batch

    for index, caption in enumerate(captions):

        print('index ', index , '/ ', len(captions))
        top_images, top_captions, top_image_embeddings, top_image_caption_embeddings, top_k_subjects = get_target_image(caption, dataset_processor, clip_model)

        # save stuff to file
        filename = 'datasets/mscoco/linear_eval_dataset/dataset'
        data = {
            'input_caption': caption,
            'input_caption_subjects': top_k_subjects,
            'top_k_captions': top_captions,
            'top_k_images': top_images,
            'top_k_image_embeddings': top_image_embeddings,
            'top_k_image_caption_embeddings': top_image_caption_embeddings,
        }
        with open(filename, 'a+') as fp:
            pickle.dump(data, fp)
    n_batches = n_batches + 1

        
    

