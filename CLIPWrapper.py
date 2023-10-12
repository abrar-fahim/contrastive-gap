from PIL import Image
import requests
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import seaborn as sns
import random

from archives.text_images import image_urls, texts, image_names

from similarities import text_image_similarity, text_similarity, image_similarity

import sklearn.decomposition

from sklearn.manifold import TSNE

import clip



class CLIPWrapper:

    # init
    def __init__(self, texts, images, similarity_type='cosine', dim_reduction_technique=None, dims=None):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # device='cpu'

        # self.model, self.processor = clip.load("ViT-B/32", device=device)
        self.texts = texts
        self.images = images

        # set inputs
        self.inputs = self.processor(text=self.texts, images=self.images, return_tensors="pt", padding=True)

        self.outputs = self.model(**self.inputs)

        self.similarity_type = similarity_type

        self.dim_reduction_technique = dim_reduction_technique

        self.dims = dims




    def get_pca_embeddings(self, output='projection'):
        '''
        Returns text and image embeddings after some dimensionality reduction
        Does dimensionality reduction on the common space occupied by both image and text embeddings

        Depends on the dim_reduction_technique attribute

        output: 'encoder', 'projection', or 'both'

        encoder output returns raw text and image embeddings

        projection output returns text and image embeddings after linear CLIP projection layer 
        '''

        n_texts = self.outputs.text_embeds.shape[0]
        n_images = self.outputs.image_embeds.shape[0]


        if self.dim_reduction_technique is None:
            raise ValueError('dim_reduction_technique attribute must be set')


        if output == 'projection':
            text_embeds = self.outputs.text_embeds
            text_embeds = text_embeds.detach().cpu()

            image_embeds = self.outputs.image_embeds
            image_embeds = image_embeds.detach().cpu()

            all_embeds = np.concatenate((text_embeds, image_embeds), axis=0)

        elif output == 'encoder':
            text_encoder_outputs = self.outputs.text_model_output
            text_embeds = text_encoder_outputs.pooler_output.detach().cpu()

            image_encoder_outputs = self.outputs.vision_model_output
            image_embeds = image_encoder_outputs.pooler_output.detach().cpu()

            all_embeds = np.concatenate((text_embeds, image_embeds), axis=0)

        elif output == 'both':
            text_embeds = self.outputs.text_embeds
            text_embeds = text_embeds.detach().cpu()

            image_embeds = self.outputs.image_embeds
            image_embeds = image_embeds.detach().cpu()

            text_encoder_outputs = self.outputs.text_model_output
            text_encoder_outputs = text_encoder_outputs.pooler_output.detach().cpu()

            image_encoder_outputs = self.outputs.vision_model_output
            image_encoder_outputs = image_encoder_outputs.pooler_output.detach().cpu()


            # CAVEAT HERE, FIX LATER
            '''
            to concatenate text_encoder_outputs and image_encoder_outputs, we need to make them the same dimension.
            So, reduce image_encoder_outputs dimension from 768 to 512 by doing PCA on them only
            Doing this changes all_embeds, but since we're only using image embeds when doing some sort of dimensionality reduction, its fine ig

            UPDATE: this wont work unless I have atleast 512 images
            '''

            # do PCA dimensionality reduction on image_encoder_outputs

            # pca = sklearn.decomposition.PCA(n_components=512, random_state=0)

            # image_encoder_outputs_pca = pca.fit_transform(image_encoder_outputs)
           
            # all_embeds = np.concatenate((text_embeds, image_embeds, text_encoder_outputs, image_encoder_outputs_pca), axis=0)

            # problem here is that image_encoder_outputs is 768 dimensional, whereas everything else is 512 dimensional.

            # concat text embeddings with image embeddings          
            # for now, just do ignoring image_encoder_outputs
            all_embeds = np.concatenate((text_embeds, image_embeds, text_encoder_outputs), axis=0)

        else:
            raise ValueError('output must be either projection or encoder')


        # text_embeds shape: (n, 512)

        
        if self.dim_reduction_technique == 'pca':
            # do PCA dimensionality reduction on text embeddings
            pca = sklearn.decomposition.PCA(n_components=self.dims, random_state=0)


            all_embeds_pca = pca.fit_transform(all_embeds)

            # # do PCA on image_encoder outputs now
            # image_encoder_pca = sklearn.decomposition.PCA(n_components=self.dims, random_state=0)

            # image_encoder_outputs_pca = image_encoder_pca.fit_transform(image_encoder_outputs)

            # # do PCA again on all embeds

            # all_embeds_pca = np.concatenate((all_embeds_pca, image_encoder_outputs_pca), axis=0)

            # pca_2 = sklearn.decomposition.PCA(n_components=self.dims, random_state=0)

            # all_embeds_pca = pca_2.fit_transform(all_embeds_pca)

            all_embeds_dim_reduced = all_embeds_pca

            

        elif self.dim_reduction_technique == 'tsne':
            # do TSNE dimensionality reduction on text embeddings
            tsne = TSNE(n_components=self.dims, perplexity=2, random_state=0)

            all_embeds_tsne = tsne.fit_transform(all_embeds)

            all_embeds_dim_reduced = all_embeds_tsne

        if self.dim_reduction_technique is not None:
            # seperate text and image embeddings
            text_embeds = all_embeds_dim_reduced[:text_embeds.shape[0], :]

            image_embeds = all_embeds_dim_reduced[text_embeds.shape[0]:text_embeds.shape[0]+image_embeds.shape[0], :]

            if output == 'both':
                text_encoder_outputs = all_embeds_dim_reduced[n_texts+n_images:2*n_texts+n_images, :]


                # these wont exist now, but they will LATER
                image_encoder_outputs = all_embeds_dim_reduced[2*n_texts+n_images:, :]


        if output == 'projection' or output == 'encoder':
            return text_embeds, image_embeds
        elif output == 'both':
            return text_embeds, image_embeds, text_encoder_outputs, image_encoder_outputs
        
        

    def get_text_embeddings(self, both=False):
        text_embeds = self.outputs.text_embeds
        text_embeds = text_embeds.detach().cpu()

        # text_embeds shape: (n, 512)

        if self.dims is not None:
           
            if both:
                text_embeds = self.get_pca_embeddings(output='both')[0]
            else:
                text_embeds = self.get_pca_embeddings(output='projection')[0]

        return text_embeds # shape: (n,512)
    

    
    def get_image_embeddings(self, both=False):
        image_embeds = self.outputs.image_embeds
        image_embeds = image_embeds.detach().cpu()

        if self.dims is not None:
            if both:
                image_embeds = self.get_pca_embeddings(output='both')[1]
            else:
                image_embeds = self.get_pca_embeddings(output='projection')[1]
        return image_embeds
    

    def get_text_encoder_outputs(self, both=False):
        text_encoder_outputs = self.outputs.text_model_output
        text_embeds = text_encoder_outputs.pooler_output.detach().cpu()

        if self.dims is not None:
           
            if both:
                # using both here for now, reconsider LATER
                text_embeds = self.get_pca_embeddings(output='both')[2]
            else:
                text_embeds = self.get_pca_embeddings(output='encoder')[0]
        return text_embeds

    def get_image_encoder_outputs(self, both=False):
        image_encoder_outputs = self.outputs.vision_model_output
        image_embeds = image_encoder_outputs.pooler_output.detach().cpu()

        if self.dims is not None:
           
            if both:
                # using both here for now, reconsider LATER
                image_embeds = self.get_pca_embeddings(output='both')[3]
            else:
                image_embeds = self.get_pca_embeddings(output='encoder')[1]
        return image_embeds

    def get_logits_per_image_and_probs(self):

        logits_per_image = self.outputs.logits_per_image  # this is the image-text similarity score

        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        return logits_per_image, probs
    
    def get_text_text_similarities(self):

        text_embeds = self.get_text_embeddings()
        
        cosine_similarities = text_similarity(text_embeds, self.similarity_type, self.dims)
        return cosine_similarities
    
    def get_average_text_text_similarity(self):
        text_text_similarities = self.get_text_text_similarities()
        print('text-text similarities ', text_text_similarities)

        n_texts = text_text_similarities.shape[0]

        # extract upper triangle of similarity matrix into a flat vector
        upper_tri = text_text_similarities[np.triu_indices(n_texts, k=1)]

        # take average
        avg_text_similarity = np.mean(np.array(upper_tri))


        return avg_text_similarity
    
    def get_text_image_similarities(self):

        text_embeds = self.get_text_embeddings()
        image_embeds = self.get_image_embeddings()
        cosine_similarities = text_image_similarity(text_embeds, image_embeds, self.similarity_type, self.dims)
        return cosine_similarities
    
    def get_average_text_image_similarity(self):
        text_image_similarities = self.get_text_image_similarities()
        print('text image similarities ', text_image_similarities)

        avg_text_image_similarity = np.mean(np.array(text_image_similarities))

        return avg_text_image_similarity
    
    def get_image_image_similarities(self):
        image_embeds = self.get_image_embeddings()
        cosine_similarities = image_similarity(image_embeds, self.similarity_type, self.dims)
        return cosine_similarities
    
    def get_average_image_image_similarity(self):
        image_image_similarities = self.get_image_image_similarities()
        print('image image similarities ', image_image_similarities)

        n_images = image_image_similarities.shape[0]

        # extract upper triangle of similarity matrix into a flat vector
        upper_tri = image_image_similarities[np.triu_indices(n_images, k=1)]

        # take average
        avg_image_similarity = np.mean(np.array(upper_tri))

        return avg_image_similarity




# if main

if __name__ == '__main__':

    images = []

    for url in image_urls:
        images.append(Image.open(requests.get(url, stream=True).raw))


    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # classes = ["white cat on pink bed", "pink cat on white bed"] # this is error

    texts = ['Two individuals learning to ski along with an instructor', 'Two vases of fresh flowers sit on top of the table.', 'A man holding up a banana in front of him.', 'A blue and white motorcycle with a trunk on the back parked by a curb.', 'close up of a black cat neat a bottle of wine']

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

    outputs = model(**inputs)

    print('outputs ', outputs)

    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    print('logits_per_image ', logits_per_image)
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

    print("Label probs:", probs)


    plt.figure(figsize=(10, 10))

    cosine_similarities = text_similarity(outputs.text_embeds.detach().cpu(), 'cosine_similarity')
    # cosine_similarities = text_image_similarity(outputs, 'cosine')
    # cosine_similarities = image_similarity(outputs, 'cosine')


    # USE FOR IMAGES AND TEXTS 
    # sns.heatmap(cosine_similarities, annot=True, fmt=".4f", linewidths=.5, square=True, xticklabels=image_names, yticklabels=texts, cbar=False)
    # plt.title("Cosine Similarity Between Text and Image Features")

    # USE FOR TEXTS ONLY
    sns.heatmap(cosine_similarities, annot=True, fmt=".2f", linewidths=.5, square=True, xticklabels=texts, yticklabels=texts, cbar=False)
    plt.title("Cosine Similarity of Text Features")

    # FOR IMAGES ONLY
    # sns.heatmap(cosine_similarities, annot=True, fmt=".2f", linewidths=.5, square=True, xticklabels=image_names, yticklabels=image_names, cbar=False)
    # plt.title("Cosine Similarity of Image Features")

    plt.show()

