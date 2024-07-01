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



# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 

from src.config import training_hyperparameters



wandb.init(config=training_hyperparameters)

torch.manual_seed(wandb.config['seed'])


from tqdm import tqdm
from dataset_processors.mscoco_processor import MSCOCOProcessor
from dataset_processors.conceptual_captions_processor import ConceptualCaptionsProcessor

# processor = ConceptualCaptionsProcessor()
processor = MSCOCOProcessor()
batch_size = wandb.config['validation_batch_size']

collate_fn = processor.collate_fn
# val_dataloader = torch.utils.data.DataLoader(processor.val_data_pipe, batch_size=batch_size, collate_fn=collate_fn, generator=torch.Generator().manual_seed(wandb.config['seed']))
val_dataloader = torch.utils.data.DataLoader(processor.val_dataset, batch_size=batch_size, collate_fn=collate_fn, generator=torch.Generator().manual_seed(wandb.config['seed']))

# train_dataloader = torch.utils.data.DataLoader(processor.train_dataset, batch_size=batch_size, collate_fn=collate_fn, generator=torch.Generator().manual_seed(wandb.config['seed']))





print('len dataset ', len(processor.train_dataset))
# image = processor.train_dataset[393226]

image = processor.train_dataset._load_image(131089)

image.save('./test.jpeg')

print('image ', image)

exit()





# for image, caption in tqdm(data_pipe):
for image, caption in tqdm(processor.train_dataloader):
# for image, caption in tqdm(processor.val_data_pipe):
# for image, caption in tqdm(val_dataloader):

    # check for repeats in caption

    if len(caption) != len(set(caption)):
        print('-- REPEATSSSS IN CAPTIONNNN')
        continue
    else:
        print('No repeats in caption')

    # for i, cap in enumerate(caption):


    #     print(cap)

    #     # display first image
    #     plt.imshow(image[i].permute(1, 2, 0))
    #     plt.show()
        
    print(caption[10])
    # display first image
    plt.imshow(image[10].permute(1, 2, 0))
    plt.show()



    # if image == None:
    #     print('Image is None')
    #     continue


    
    # print(f"{caption[0]}")
    # print(f"Image size: {image.shape}")
    # # display first image
    # plt.imshow(image[0].permute(1, 2, 0))
    # plt.show()

    
    # Don't do anything here.  We just want to test the loading speed.
    pass


# for i, (image, caption) in enumerate(data_pipe):
#     if i >= 10:
#         break
#     print(f"Caption {i + 1}: {caption}")
#     print(f"Image size: {image.size}")