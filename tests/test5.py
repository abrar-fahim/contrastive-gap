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






image_embeds = torch.randn(10, 512, dtype=torch.float32)

text_embeds = torch.randn(10, 512, dtype=torch.float32)

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