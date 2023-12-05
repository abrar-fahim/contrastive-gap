import sys
import os

from tqdm import tqdm
import torch
import torchvision

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_processors.mscoco_processor import MSCOCOProcessor
import torch
import matplotlib.pyplot as plt
import numpy as np
from clips.hf_clip import HFClip

processor = MSCOCOProcessor()

# processor.show_real_images_captions = True

torch.manual_seed(42)


# get first batch
batch = next(iter(processor.train_dataloader))

images, captions, original_images, original_captions = batch


print('caption ', original_captions[0])

# display first image

plt.imshow(original_images[0])
plt.show()

# create enum of different colors
class OPTIONS:
    search_nearest_images = 1
    search_nearest_captions = 2

option = OPTIONS.search_nearest_captions


'''
Get images closest to the first image
'''

clip_model = HFClip()

torch.manual_seed(42) # resetting the batch

batch = next(iter(processor.train_dataloader))
images, captions, original_images, original_captions = batch
outputs = clip_model(images, captions, return_all=True)
image_embeddings = outputs.image_embeds
target_image_embedding = image_embeddings[0]
target_image = original_images[0]

target_caption = original_captions[0]

target_caption_embedding = outputs.text_embeds[0]

n = 10

# top_n_image_embeddings = torch.zeros((0, image_embeddings.shape[1]))

# top_n_caption_embeddings = torch.zeros((0, target_caption_embedding.shape[0]))


top_n_embeddings = torch.zeros((0, image_embeddings.shape[1]))

top_n_images = []

top_n_captions = []

for batch in tqdm(processor.train_dataloader):
    images, captions, original_images, original_captions = batch
    outputs = clip_model(images, captions, return_all=True)
    image_embeddings = outputs.image_embeds

    if option == OPTIONS.search_nearest_images:
        top_n_embeddings = torch.cat((top_n_embeddings, image_embeddings), dim=0)
    elif option == OPTIONS.search_nearest_captions:
        top_n_embeddings = torch.cat((top_n_embeddings, outputs.text_embeds), dim=0)

    top_n_images.extend(original_images)

    top_n_captions.extend(original_captions)


    # compare with target image embedding using cosine similarity
    # cosine_similarities = torch.nn.functional.cosine_similarity(target_image_embedding, top_n_embeddings)

    if option == OPTIONS.search_nearest_images:
        cosine_similarities = torch.matmul(top_n_embeddings, target_image_embedding.unsqueeze(-1)).squeeze(-1)
    elif option == OPTIONS.search_nearest_captions:
        cosine_similarities = torch.matmul(top_n_embeddings, target_caption_embedding.unsqueeze(-1)).squeeze(-1)


    # get top n images
    top_n_image_indices = torch.topk(cosine_similarities, n).indices


    # get top n image embeddings
    top_n_embeddings = top_n_embeddings[top_n_image_indices]


    # get top n images
    top_n_images = [top_n_images[i] for i in top_n_image_indices]

    # get top n captions
    top_n_captions = [top_n_captions[i] for i in top_n_image_indices]

# show top n images and target image
top_n_images.append(target_image)

print('top captions ', top_n_captions)

# show 10 images as a grid
fig = plt.figure(figsize=(10, 10))
columns = 5
rows = 2
frame = plt.gca()



for i in range(1, columns*rows +1):
    img = top_n_images[i-1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    # show cosine similarity
    plt.title(f'{cosine_similarities[i-1]:.2f}')

    # hide axes
    plt.axis('off')
    # show the caption
    # plt.xlabel(top_n_captions[i-1])
    # frame.axes.get_xaxis().set_ticklabels([])
    # frame.axes.get_yaxis().set_ticklabels([])
plt.show()