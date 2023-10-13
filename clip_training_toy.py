'''
- Setup image encoder
- Setup text encoder
- use nonlinear projection layer
- Setup toy dataset
- Setup training loop
- Train model on toy dataset using minibatches
- Test to see if model can get high cosine similarities between images and captions of same concept
- This works as sanity check to test algorithm in general

'''

'''
- Setup toy dataset, MSCOCO for now
'''

import json

import numpy as np

# load dataset

import torchvision.datasets as dset
import torchvision.transforms as transforms
cap = dset.CocoCaptions(root = './datasets/mscoco/val2014',
                        annFile = 'datasets/mscoco/annotations/captions_val2014.json',
                        transform=transforms.PILToTensor())

print('Number of samples: ', len(cap))
img, target = cap[3] # load 4th sample

# display image

# import matplotlib.pyplot as plt
# plt.imshow( img.permute(1, 2, 0)  )
# plt.show()


print("Image Size: ", img.size())
print(target)



exit()

import torch

from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor
from PIL import Image
import requests

from transformers import AutoImageProcessor, ViTModel
import torch
from datasets import load_dataset

# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]

'''
- Setup Image Encoder
'''



url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
image_encoder = ViTModel.from_pretrained('google/vit-large-patch16-224')
# classification_model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
inputs = feature_extractor(images=image, return_tensors="pt")

print('inputs ', inputs['pixel_values'].shape)
# outputs = model(**inputs)

outputs = image_encoder(**inputs)


pooler_output = outputs.pooler_output # ([1, 1024])

print('pooler_output ', pooler_output.shape)


'''
- Setup Text Encoder
'''

from transformers import AutoTokenizer, GPT2Model, GPT2Config
import torch

configuration = GPT2Config(summary_type="cls_index")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

inputs = tokenizer("Hello my dog is cute.", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

print('last_hidden_states ', last_hidden_states.shape)

eos_representation = last_hidden_states[:, -1, :] # ([1, 768])
print('eos_representation ', eos_representation.shape)

'''
- Setup nonlinear projection layer
'''

import torch.nn as nn





# print('outputs ', outputs)
# logits = outputs.logits
# # model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])


