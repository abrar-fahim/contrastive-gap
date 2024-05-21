


uniformity = -3.4139-2.1259-3.0493-2.5984-2.5751

centroid_dist=0.5068+0.7397+0.6281+0.5222+0.5610

print('centroid dist ', centroid_dist / 5)

print('uniformity ', uniformity / 5)

exit()




# conceptual captions streaming test

from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.iter import HttpReader, LineReader
import torchdata.datapipes as dp
import aiohttp
from PIL import Image
import io
from typing import Optional
from typing import List
from typing import Sequence, Tuple
import asyncio
from typing import Generator
import torch
import matplotlib.pyplot as plt

import sys
import os
import wandb
import random
import numpy as np
from torchdata.datapipes.iter import FileLister, FileOpener




# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 

from src.config import training_hyperparameters
from src.my_ce_loss import MyCrossEntropyLoss
# from dataset_processors.food101_processor import Food101Processor

# wandb.init(config=training_hyperparameters)

from tqdm import tqdm






image_embeds = torch.randn(5, 3, dtype=torch.float32)

text_embeds = torch.randn(5, 3, dtype=torch.float32)

text_embeds = torch.zeros(5, 3, dtype=torch.float32)

text_embeds = image_embeds + 1
print('image embeds ', image_embeds)

print('text embeds ', text_embeds)



pairwise_image_dirs = image_embeds.unsqueeze(1) - image_embeds

pairwise_text_dirs = text_embeds.unsqueeze(1) - text_embeds

loss = (pairwise_image_dirs - pairwise_text_dirs).square().mean()

print('loss ', loss)
exit()


print('image embeds shape ', image_embeds.shape)

print("image embeds unsqueeze shape ", image_embeds.unsqueeze(0).shape)

dists = torch.cdist(image_embeds.unsqueeze(0), text_embeds.unsqueeze(0))[0]

print('dists ', dists.shape)

# create torch matrix with ones everywhere except the diagonal

ones = torch.ones((len(image_embeds), len(text_embeds))).to('cpu').tril(diagonal = -1)

ones += torch.ones((len(image_embeds), len(text_embeds))).to('cpu').triu(diagonal = 1)

print('ones ', ones)


selected_elements = torch.masked_select(dists, ones == 1)

print('masked select ', torch.masked_select(dists, ones == 1).square().shape)


# logits_per_image = torch.randn(64, 10)



# class_labels = torch.randint(0, 9, (64,))

exit()







uniform_loss = torch.masked_select(torch.cdist(image_embeds.unsqueeze(0), text_embeds.unsqueeze(0))[0], torch.ones((len(image_embeds), len(text_embeds))).to('cpu').tril(diagonal = -1) == 1).square().mul(-2).exp().mean().log()

print('uniform loss ', uniform_loss)