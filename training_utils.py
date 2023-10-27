import torch
import torch.nn as nn
import torch.nn.functional as F
from clip_parent import ClipParent
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, CLIPTextModel, CLIPVisionModel, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from PIL import Image
import requests


def do_validation(val_dataloader, clip_model):
    with torch.no_grad():
        # get batch from validation set
        (val_imgs, val_captions) = next(iter(val_dataloader))
        outputs = clip_model(val_imgs, val_captions, output_loss=False) # so tha I get cosine similarities directly
        logits_per_image, logits_per_text = outputs # shape of both: ([64, 64])

        # print('logits_per_image ', logits_per_image)

        # print logits per image for first 5 images
        # print('logits_per_image ', logits_per_image[:5, :5])
        cosine_similarities = logits_per_image.diag() # shape: [64]
        # get median cosine similarity
        median_cosine_similarity = torch.median(cosine_similarities)
        print('median cosine similarity ', median_cosine_similarity)

        # get median of elements that are not on the diagonal
        non_similar_median_cosine_similarity = logits_per_image[~torch.eye(logits_per_image.shape[0], dtype=bool)].median()
        print('non_similar_median_cosine_similarity ', non_similar_median_cosine_similarity)

        # print temperature
        print('clip_model.logit_scale ', clip_model.model.logit_scale)

        '''
        - Get text-text similarities
        '''

        # text_encoder_outputs = clip_model.project_image(val_captions) # shape: ([batch_size, 512])

        # # normalize features
        # text_encoder_outputs = text_encoder_outputs / torch.norm(text_encoder_outputs, dim=1, keepdim=True)

        # # cosine similarities between text-text pairs
        # text_text_cosine_similarities = text_encoder_outputs @ text_encoder_outputs.t() # shape: ([batch_size, batch_size])

        # # get median of elements that are in the upper triangle (excluding diagonal!!)
        # median_text_text_cosine_similarity = text_text_cosine_similarities[torch.triu(torch.ones(text_text_cosine_similarities.shape[0], text_text_cosine_similarities.shape[1]), diagonal=1).bool()].median()

        # print('median_text_text_cosine_similarity ', median_text_text_cosine_similarity)

        # '''
        # - Get image-image similarities
        # '''

        # image_encoder_outputs = clip_model.project_image(val_imgs) # shape: ([batch_size, 512])

        # # normalize features
        # image_encoder_outputs = image_encoder_outputs / torch.norm(image_encoder_outputs, dim=1, keepdim=True)

        # # cosine similarities between image-image pairs
        # image_image_cosine_similarities = image_encoder_outputs @ image_encoder_outputs.t()

        # # get median of elements that are not on the diagonal
        # median_image_image_cosine_similarity = image_image_cosine_similarities[~torch.eye(image_image_cosine_similarities.shape[0], dtype=bool)].median()

        # print('median_image_image_cosine_similarity ', median_image_image_cosine_similarity)

