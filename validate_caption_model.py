from training_utils import do_validation
import torch
from my_clip import MyClip, MyClipLoss
from grad_cache_wrapper import GradCacheWrapper
from openai_clip import OpenAIClip
from hf_clip import HFClip
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch
from training_utils import do_validation
import clip
import os
import torchvision.datasets as dset
import matplotlib.pyplot as plt
from PIL import Image
import requests

# set seed
torch.manual_seed(42)

training_hyperparameters = {
    'batch_size': 16,
    'grad_cache': False,
    'grad_cache_multiplier': 32,
    'n_epochs': 1,
    'lr': 1e-5,
    'weight_decay': 0.2,
    'model_path': 'checkpoints/my_clip_checkpoint.pt',
    'validation_dataset_size': 256,
    'validation_batch_size': 256,
    'do_checkpointing': False,
    'start_new': False,
    'use_small_trainloader': True,
    'small_train_loader_batch_size': 256,
    'small_train_loader_dataset_size': 10000,
    }


def collate_fn(batch):
    '''
    batch is a list of tuples?
    each tuple is of the form (image, caption)
    image is a tensor of shape [3, 224, 224]
    caption is a tuple of strings
    '''

    imgs, og_captions = zip(*batch)

    # keep only first caption for each image
    captions = [caption[0] for caption in og_captions]

    # caption2 = [caption[0] for caption in og_captions]
    # return (caption2, captions)
    return (torch.stack(imgs), captions)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print('device ', device)

model, preprocess = clip.load("ViT-B/32", device=device)




train_dataset = dset.CocoCaptions(root = './datasets/mscoco/val2014',
                        annFile = 'datasets/mscoco/annotations/captions_val2014.json',
                        # transform=[transforms.PILToTensor()])
                        transform=preprocess,
)

clip_model = HFClip().to(device)

subset_indices = torch.randint(0, len(train_dataset) , (training_hyperparameters['small_train_loader_dataset_size'],)) # always defined and exists, but only used when small training loader is used, and we're not loading from checkpoint at start

train_data_subset = Subset(train_dataset, subset_indices)

train_dataloader = DataLoader(train_data_subset, batch_size=training_hyperparameters['small_train_loader_batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)

dataloader = train_dataloader


# get 100 indices that are not in train_data_subset
val_indices = torch.randint(0, len(train_dataset) , (training_hyperparameters['validation_dataset_size'],))
j = 0
while j < training_hyperparameters['validation_dataset_size']:
    while val_indices[j] in subset_indices:
        val_indices[j] = torch.randint(0, len(train_dataset) , (1,))
    j += 1
print('j ', j)

val_data_subset = Subset(train_dataset, val_indices)

val_dataloader = DataLoader(val_data_subset, batch_size=training_hyperparameters['validation_batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)



for (imgs, captions) in dataloader:

    clip_model.eval()


    do_validation(val_dataloader, clip_model, index=0, captioning_model=True)

    break