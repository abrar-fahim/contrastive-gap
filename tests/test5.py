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
# from dataset_processors.food101_processor import Food101Processor

# wandb.init(config=training_hyperparameters)

from tqdm import tqdm



image_embeds = torch.randint(0, 10, (10, 3), dtype=torch.float32)

text_embeds = torch.randint(0, 10, (10, 3), dtype=torch.float32)

uniform_loss = torch.masked_select(torch.cdist(image_embeds.unsqueeze(0), text_embeds.unsqueeze(0))[0], torch.ones((len(image_embeds), len(text_embeds))).to('cpu').tril(diagonal = -1) == 1).square().mul(-2).exp().mean().log()

print('uniform loss ', uniform_loss)