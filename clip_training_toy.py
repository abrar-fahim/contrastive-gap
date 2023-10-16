'''
- Setup image encoder
- Setup text encoder
- use nonlinear projection layer
- Setup toy dataset
- Setup loss fn
- Setup training loop

- Train model on toy dataset using minibatches
- Test to see if model can get high cosine similarities between images and captions of same concept
- This works as sanity check to test algorithm in general

'''

'''
- Setup toy dataset, MSCOCO for now
'''

from my_clip import MyClip, MyClipLoss
from torch.utils.data import DataLoader
import torch.optim as optim
import torch

import clip
# load dataset

import os

import torchvision.datasets as dset

start_new = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print('device ', device)

model, preprocess = clip.load("ViT-B/32", device=device)
train_dataset = dset.CocoCaptions(root = './datasets/mscoco/val2014',
                        annFile = 'datasets/mscoco/annotations/captions_val2014.json',
                        # transform=[transforms.PILToTensor()])
                        transform=preprocess,
                        )

print('Number of samples: ', len(train_dataset))


img, target = train_dataset[0] # load 4th sample

# check if there are any images with <5 captions
# for i, (image, cap) in enumerate(train_dataset):
#     # print progress
#     if i % 1000 == 0:
#         print('progress: ', i, '/', len(train_dataset))
#     if len(cap) > 5:
#         print('image with more than 5 captions: ', i)
#         print('cap ', cap)

# there are some images with > 5 captions
# display image

# import matplotlib.pyplot as plt
# plt.imshow( img.permute(1, 2, 0)  )
# plt.show()








clip_model = MyClip().to(device)

# print parameters that are trainable
for name, param in clip_model.named_parameters():
    if param.requires_grad:
        print(name)


'''
- Setup training loop
'''

# setup adamW optimizer

optimizer = optim.AdamW(clip_model.parameters(), lr=1e-2)

# setup loss function
clip_loss = MyClipLoss()

def collate_fn(batch):
    '''
    batch is a list of tuples?
    each tuple is of the form (image, caption)
    image is a tensor of shape [3, 224, 224]
    caption is a tuple of strings
    '''

    imgs, captions = zip(*batch)

    # keep only first caption for each image
    captions = [caption[0] for caption in captions]
    return (torch.stack(imgs), captions)



# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

n_epochs = 3

# set seed
torch.manual_seed(42)


clip_model.train()




'''
checkpointing stuff
'''





model_path = 'checkpoints/my_clip_checkpoint.pt'

if os.path.exists(model_path) and not start_new:

    # load checkpoint
    checkpoint = torch.load(model_path)
    clip_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    train_dataloader = checkpoint['train_dataloader']
    i = checkpoint['dataloader_enumerator_index']

else:
    epoch = 0
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    i = 0


# training loop
while epoch < n_epochs:


    running_loss = 0.0

    
    for (img, caption) in train_dataloader:

        # zero the parameter gradients
        optimizer.zero_grad()

        # caption WAS a list of tuples, where first tuple corresponds to first captions of all the images in the batch

        # caption is now a list of 64 strings 

        # forward + backward + optimize
        outputs = clip_model(img, caption)
        loss = clip_loss(*outputs)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2 == 1:    # print every 2 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2))
            running_loss = 0.0

        # save model 
        if i % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': clip_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'train_dataloader': train_dataloader,
                'dataloader_enumerator_index': i,
                }, model_path)
        i += 1



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


