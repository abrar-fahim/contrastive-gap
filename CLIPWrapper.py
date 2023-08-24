from PIL import Image
import requests
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import seaborn as sns

from text_images import image_urls, texts, image_names

from similarities import text_image_similarity, text_similarity, image_similarity

import sklearn.decomposition



class CLIPWrapper:

    # init
    def __init__(self, texts, images, similarity_type='cosine', pca_dims=None):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.texts = texts
        self.images = images

        # set inputs
        self.inputs = self.processor(text=self.texts, images=self.images, return_tensors="pt", padding=True)

        self.outputs = self.model(**self.inputs)

        self.similarity_type = similarity_type

        self.pca_dims = pca_dims

    def get_pca_embeddings(self):
        '''
        Returns text and image embeddings after PCA dimensionality reduction
        Does PCA dimensionality reduction on the common space occupied by both image and text embeddings
        '''
        text_embeds = self.outputs.text_embeds
        text_embeds = text_embeds.detach().cpu()

        image_embeds = self.outputs.image_embeds
        image_embeds = image_embeds.detach().cpu()

        # text_embeds shape: (n, 512)

        # concat text embeddings with image embeddings

        all_embeds = np.concatenate((text_embeds, image_embeds), axis=0)
        # do PCA dimensionality reduction on text embeddings
        pca = sklearn.decomposition.PCA(n_components=self.pca_dims)


        all_embeds_pca = pca.fit_transform(all_embeds)

        # seperate text and image embeddings
        text_embeds = all_embeds_pca[:text_embeds.shape[0], :]

        image_embeds = all_embeds_pca[text_embeds.shape[0]:, :]

        return text_embeds, image_embeds

        

    def get_text_embeddings(self):
        text_embeds = self.outputs.text_embeds
        text_embeds = text_embeds.detach().cpu()

        

        # text_embeds shape: (n, 512)

        if self.pca_dims is not None:

           text_embeds = self.get_pca_embeddings()[0]

        return text_embeds # shape: (n,512)
    
    def get_image_embeddings(self):
        image_embeds = self.outputs.image_embeds
        image_embeds = image_embeds.detach().cpu()

        if self.pca_dims is not None:
            
           image_embeds = self.get_pca_embeddings()[1]
        return image_embeds

    def get_logits_per_image_and_probs(self):

        logits_per_image = self.outputs.logits_per_image  # this is the image-text similarity score

        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        return logits_per_image, probs
    
    def get_text_text_similarities(self):

        text_embeds = self.get_text_embeddings()
        
        cosine_similarities = text_similarity(text_embeds, self.similarity_type, self.pca_dims)
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
        cosine_similarities = text_image_similarity(text_embeds, image_embeds, self.similarity_type, self.pca_dims)
        return cosine_similarities
    
    def get_average_text_image_similarity(self):
        text_image_similarities = self.get_text_image_similarities()
        print('text image similarities ', text_image_similarities)

        avg_text_image_similarity = np.mean(np.array(text_image_similarities))

        return avg_text_image_similarity
    
    def get_image_image_similarities(self):
        image_embeds = self.get_image_embeddings()
        cosine_similarities = image_similarity(image_embeds, self.similarity_type, self.pca_dims)
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

