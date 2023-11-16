from clips.hf_clip import HFClip
import torch

import clip
from clips.openai_clip import OpenAIClip

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, preprocess = clip.load("ViT-B/32", device=device)

# hf clip
hf_clip = HFClip().to(device)

# openai clip
openai_clip = OpenAIClip().to(device)

images_dir = 'arithmetic_images/'



cat_colors = ['brown', 'black', 'ginger', 'white', 'blue', 'car', 'couch']

objects = ['car']

preprocessed_cat_images = []

original_cat_images = []

# get original cat images
for color in cat_colors:
    original_cat_images.append(Image.open(images_dir + f"{color}_cat.jpeg"))


# preprocess cat images
preprocessed_cat_images = [preprocess(Image.open(images_dir + f"{color}_cat.jpeg")).unsqueeze(0).to(device) for color in cat_colors]

# make tensor
preprocessed_cat_images = torch.cat(preprocessed_cat_images, dim=0)

'''
hf clip
'''
# get cat embeddings
hf_clip_cat_embeddings = {
    color: hf_clip.encode_image(preprocessed_cat_images[i].unsqueeze(0)) for i, color in enumerate(cat_colors)
}

open_ai_cat_embeddings = {
    color: openai_clip.encode_image(preprocessed_cat_images[i].unsqueeze(0)) for i, color in enumerate(cat_colors)
}

# get text embeddings for all the colors, in the form of a dictionary
hf_color_embeddings = {
    color: hf_clip.encode_text([color]) for color in cat_colors
}

open_ai_color_embeddings = {
    color: openai_clip.encode_text([color]) for color in cat_colors
}

hf_object_embeddings = {
    object: hf_clip.encode_text([object]) for object in objects
}

open_ai_object_embeddings = {
    object: openai_clip.encode_text([object]) for object in objects
}




def test_cat_color(image, subtract, add):

    # white cat - white color + brown color = brown cat
    hf_clip_pred_brown_cat = hf_clip_cat_embeddings[image] - hf_color_embeddings[subtract] + hf_color_embeddings[add]

    open_ai_pred_brown_cat = open_ai_cat_embeddings[image] - open_ai_color_embeddings[subtract] + open_ai_color_embeddings[add]

    # print all cosine similarities between the predicted brown cat and the cat embeddings

    hf_cosine_similarities = torch.cosine_similarity(hf_clip_pred_brown_cat, torch.cat(list(hf_clip_cat_embeddings.values())), dim=-1)

    open_ai_cosine_similarities = torch.cosine_similarity(open_ai_pred_brown_cat, torch.cat(list(open_ai_cat_embeddings.values())), dim=-1)

    print('hf clip cosine similarities ', hf_cosine_similarities)

    print('open ai cosine similarities ', open_ai_cosine_similarities)



    # search for closest embedding
    hf_closest_cat_embedding = torch.argmax(hf_cosine_similarities)

    open_ai_closest_cat_embedding = torch.argmax(open_ai_cosine_similarities)

    return hf_closest_cat_embedding, open_ai_closest_cat_embedding

    # search for the closest image
    hf_clip_closest_cat_image = original_cat_images[hf_closest_cat_embedding]

    open_ai_closest_cat_image = original_cat_images[open_ai_closest_cat_embedding]

    # display the two images in subplots
    # import matplotlib.pyplot as plt

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(hf_clip_closest_cat_image)
    # ax2.imshow(open_ai_closest_cat_image)
    # plt.show()

def test_cat_object(color):
    # white cat + couch = cat_couch image
    hf_clip_pred_cat_couch = hf_clip_cat_embeddings[color] + hf_object_embeddings['car']

    open_ai_pred_cat_couch = open_ai_cat_embeddings[color] + open_ai_object_embeddings['car']

    # print all cosine similarities between the predicted brown cat and the cat embeddings

    hf_cosine_similarities = torch.cosine_similarity(hf_clip_pred_cat_couch, torch.cat(list(hf_clip_cat_embeddings.values())), dim=-1)

    open_ai_cosine_similarities = torch.cosine_similarity(open_ai_pred_cat_couch, torch.cat(list(open_ai_cat_embeddings.values())), dim=-1)

    print('hf clip cosine similarities ', hf_cosine_similarities)

    print('open ai cosine similarities ', open_ai_cosine_similarities)

    # search for closest embedding
    hf_closest_cat_embedding = torch.argmax(hf_cosine_similarities)

    open_ai_closest_cat_embedding = torch.argmax(open_ai_cosine_similarities)

    # search for the closest image
    hf_clip_closest_cat_image = original_cat_images[hf_closest_cat_embedding]

    open_ai_closest_cat_image = original_cat_images[open_ai_closest_cat_embedding]

    # display the two images in subplots
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(hf_clip_closest_cat_image)
    ax2.imshow(open_ai_closest_cat_image)
    plt.show()

hf_closest_cat_embedding, open_ai_closest_cat_embedding = test_cat_color('white', 'white', 'brown')

# search for the closest image
hf_clip_closest_cat_image = original_cat_images[hf_closest_cat_embedding]

open_ai_closest_cat_image = original_cat_images[open_ai_closest_cat_embedding]

# # display the two images in subplots
# import matplotlib.pyplot as plt

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(hf_clip_closest_cat_image)
# ax2.imshow(open_ai_closest_cat_image)
# plt.show()



# test all color combinations

hf_corrects = 0
open_ai_corrects = 0
total = 0
for image in cat_colors:
    for add in cat_colors:
        if image == add:
            continue

        hf_closest_cat_embedding, open_ai_closest_cat_embedding = test_cat_color(image, image, add)

        # check if hf closest embedding matches the index
        # print('hf closest cat embedding ', hf_closest_cat_embedding)
        # print('open ai closest cat embedding ', open_ai_closest_cat_embedding)

        if hf_closest_cat_embedding == cat_colors.index(add):
            hf_corrects += 1

        if open_ai_closest_cat_embedding == cat_colors.index(add):
            open_ai_corrects += 1
        
        total += 1

            
print('hf corrects ', hf_corrects / total)
print('open ai corrects ', open_ai_corrects / total)

test_cat_object('white')
test_cat_object('black')
test_cat_object('brown')
test_cat_object('ginger')
test_cat_object('blue')




'''
do the arithmetic
'''














