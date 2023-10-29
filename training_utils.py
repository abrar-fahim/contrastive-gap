import torch
import torch.nn as nn
import torch.nn.functional as F
from clip_parent import ClipParent
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, CLIPTextModel, CLIPVisionModel, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


pca = None



def do_validation(val_dataloader, clip_model, index=0):
    with torch.no_grad():
        
        # get batch from validation set
        (val_imgs, val_captions) = next(iter(val_dataloader))

        outputs = clip_model(val_imgs, val_captions, output_loss=False, return_all=True) # so tha I get cosine similarities directly
        logits_per_image = outputs.logits_per_image # shape of both: ([64, 64])

        logits_per_text = outputs.logits_per_text # shape of both: ([64, 64])

        # softmax on logits_per_image
        image_class_probs = F.softmax(logits_per_image, dim=-1) # shape: ([64, 64])

        # calculate accuracy
        # get indices of max values
        image_class_preds = image_class_probs.argmax(dim=-1) # shape: ([64])
        # get indices of correct predictions
        image_class_labels = torch.arange(image_class_probs.shape[0], device=image_class_probs.device) # shape: ([64])

        # calculate accuracy
        image_accuracy = (image_class_preds == image_class_labels).float().mean()

        print('--- ACCURACY STUFF --- ')

        print('image preds ', image_class_preds)
        # print('image labels ', image_class_labels)

        print('image_accuracy ', image_accuracy)

        print('--- IMAGE-TEXT SIMILARITIES --- ')



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
        Check if model predictions are exploding
        (Do this check without the temperature param)
        '''

        # doing it just for images for now
        # image_embeds = outputs.vision_model_output.pooler_output # shape: ([batch_size, 512]), these are before linear projection
        image_embeds = outputs.image_embeds # shape: ([batch_size, 512]), these are after linear projection

        



        print('---IMAGE --- ')

        # find max and min values
        max_value = image_embeds.max()
        min_value = image_embeds.min()
        print('max_value ', max_value)
        print('min_value ', min_value)
        # median
        median_value = image_embeds.median()
        print('median_value ', median_value)

        # text_embeds = outputs.text_model_output.pooler_output # shape: ([batch_size, 512]), these are before linear projection
        text_embeds = outputs.text_embeds # shape: ([batch_size, 512]), these are before linear projection
        print('--- TEXT --- ')

        # find max and min values
        max_value = text_embeds.max()
        min_value = text_embeds.min()
        print('max_value ', max_value)
        print('min_value ', min_value)
        # median
        median_value = text_embeds.median()
        print('median_value ', median_value)

        # save pca plots to file
        # write_pca_plots_to_file(image_embeds, text_embeds, index, 'pca_plots/')


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


def write_pca_plots_to_file(image_projections, text_projections, index, output_dir):
    '''
    write PCA plot coordinates of image and text projections, AFTER the linear projection, to file in output_dir
    image projections shape: [batch_size, 512]
    text projections shape: [batch_size, 512]
    '''

    global pca

    # stack image and text projections
    stacked_projections = torch.cat((image_projections, text_projections), dim=0) # shape: [2*batch_size, 512]

    if pca is None:
        # get PCA
        pca = PCA(n_components=2)
        pca.fit(stacked_projections.cpu().numpy())


    # # get PCA
    # pca = PCA(n_components=2)
    # pca.fit(stacked_projections.cpu().numpy())

    # get PCA coordinates
    pca_coordinates = pca.transform(stacked_projections.cpu().numpy()) # shape: [2*batch_size, 2]

    # get image and text coordinates
    image_coordinates = pca_coordinates[:image_projections.shape[0], :] # shape: [batch_size, 2]
    text_coordinates = pca_coordinates[image_projections.shape[0]:, :] # shape: [batch_size, 2]

    # write to file
    np.save(output_dir + 'image_coordinates_' + str(index) + '.npy', image_coordinates)
    np.save(output_dir + 'text_coordinates_' + str(index) + '.npy', text_coordinates)





def plot_pca_from_file(image_coordinates_file, text_coordinates_file):
    '''
    Plot PCA of image and text projections, AFTER the linear projection
    image projections shape: [batch_size, 512]
    text projections shape: [batch_size, 512]
    '''

    # get image and text coordinates
    image_coordinates = np.load(image_coordinates_file) # shape: [batch_size, 2]
    text_coordinates = np.load(text_coordinates_file) # shape: [batch_size, 2]

    # 

    # plot
    plt.title(f'PCA of image (red) and text (blue) projections, after {image_coordinates_file.split("_")[3].split(".")[0]} pass(es)')
    plt.scatter(image_coordinates[:, 0], image_coordinates[:, 1], c='r')
    plt.scatter(text_coordinates[:, 0], text_coordinates[:, 1], c='b')
    plt.show()
    

def plot_pca_subplots_from_file(dir, start, stop, step):

    # calculate number of subplots
    num_subplots = (stop - start) // step

    # create subplots
    fig, axs = plt.subplots(1, num_subplots)

    # set title
    fig.suptitle('stacked subplots of image (red) and text (blue) projections')

    





    for i in range(start, stop, step):
        # get image and text coordinates
        image_coordinates = np.load(dir + 'image_coordinates_' + str(i) + '.npy') # shape: [batch_size, 2]
        text_coordinates = np.load(dir + 'text_coordinates_' + str(i) + '.npy') # shape: [batch_size, 2]

        # keep axes fixed for all subplots
        axs[i//step].set_xlim(-0.75, 0.75)
        axs[i//step].set_ylim(-0.75, 0.75)

        # plot
        axs[i//step].set_title(f'after {i} pass(es)')
        axs[i//step].scatter(image_coordinates[:, 0], image_coordinates[:, 1], c='r')
        axs[i//step].scatter(text_coordinates[:, 0], text_coordinates[:, 1], c='b')
    
    plt.show()


