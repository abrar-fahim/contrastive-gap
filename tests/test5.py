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
from dataset_processors.food101_processor import Food101Processor

wandb.init(config=training_hyperparameters)

from tqdm import tqdm

food101processor = Food101Processor()

# testing to see if I can stream dataset from zip file directly

for (imgs, lbls) in food101processor.train_dataset:
    print(lbls)