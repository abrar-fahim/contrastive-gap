
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
import time

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 

from src.config import training_hyperparameters



wandb.init(config=training_hyperparameters)

torch.manual_seed(wandb.config['seed'])


from tqdm import tqdm
from dataset_processors.voc_processor import VocProcessor

from dataset_processors.conceptual_captions_processor import ConceptualCaptionsProcessor





voc_processor = VocProcessor().load_val_dataset()
