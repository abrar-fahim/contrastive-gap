from clips.hf_clip import HFClip
import torch

import clip
from clips.openai_clip import OpenAIClip

device = torch.device(training_hyperparameters['cuda_device'] if torch.cuda.is_available() else "cpu")

# model, preprocess = clip.load("ViT-B/32", device=device)

# openai_clip = OpenAIClip().to(device)

openai_clip = HFClip().to(device)

labels = ['cat', 'dog']

colors = ['brown', 'black', 'white', 'blue']

colored_labels = [f"{color} {label}" for color in colors for label in labels]

openai_clip_embeddings = {
    label: openai_clip.encode_text([label]) for label in labels
}

# add color embeddings to the dictionary
for color in colors:
    openai_clip_embeddings[color] = openai_clip.encode_text([color])

# add colored_label embeddings to the dictionary
for color in colors:
    for label in labels:
        openai_clip_embeddings[f"{color} {label}"] = openai_clip.encode_text([f"{color} {label}"])

# do arithmetic

def add_and_check(color, label, return_similarity=False):
    '''
    returns True if the predicted color label is the same as the actual color label
    '''

    predicted_color_label = openai_clip_embeddings[color] + openai_clip_embeddings[label]

    # predicted_color_label = predicted_color_label / predicted_color_label.norm(dim=-1, keepdim=True)

    cosine_similarities = {}

    for colored_label in colored_labels:
        cosine_similarities[colored_label] = torch.cosine_similarity(predicted_color_label, openai_clip_embeddings[colored_label], dim=-1).item()

    # iterate over keys and values of cosine_similarities
    max_cosine_similarity = -1
    for key, value in cosine_similarities.items():
        if value > max_cosine_similarity:
            max_cosine_similarity = value
            predicted_color_label = key

    if return_similarity:
        return cosine_similarities
    
    if predicted_color_label == f"{color} {label}":
        return True
    return False
        


for color in colors:
    for label in labels:
        print(f"color: {color}, label: {label}, predicted color label: {add_and_check(color, label)}")

# plot bar graph of cosine similarities for blue cat
import matplotlib.pyplot as plt

cosine_similarities = add_and_check('blue', 'cat', return_similarity=True)

plt.bar(cosine_similarities.keys(), cosine_similarities.values())
plt.xticks(rotation=90)
plt.show()





