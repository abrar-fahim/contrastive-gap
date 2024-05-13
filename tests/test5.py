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






image_embeds = torch.randint(0, 10, (10, 512), dtype=torch.float32)

text_embeds = torch.randint(0, 10, (10, 512), dtype=torch.float32)


# logits_per_image = torch.randn(64, 10)



# class_labels = torch.randint(0, 9, (64,))

class_labels = torch.arange(0, 10).repeat(2)

# make logits_per_image according to class_labels
logits_per_image = torch.zeros((len(class_labels), 10))
for i, label in enumerate(class_labels):
    logits_per_image[i][label] = 1.0

print('logits per image ', logits_per_image)

print('labels ', class_labels)
# contrastive_loss = MyCrossEntropyLoss()
contrastive_loss = torch.nn.CrossEntropyLoss()

loss = contrastive_loss(logits_per_image, class_labels)

topk = 1
ranks = logits_per_image.topk(topk, 1)[1].T

predictions = ranks == class_labels
print('predictions ', predictions)

top1_corrects = torch.sum(torch.any(predictions[:1], dim = 0)).item() 

print('top1 corrects ', top1_corrects)


print('loss ', loss)

exit()







uniform_loss = torch.masked_select(torch.cdist(image_embeds.unsqueeze(0), text_embeds.unsqueeze(0))[0], torch.ones((len(image_embeds), len(text_embeds))).to('cpu').tril(diagonal = -1) == 1).square().mul(-2).exp().mean().log()

print('uniform loss ', uniform_loss)