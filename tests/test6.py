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
from torchdata.datapipes.iter import FileLister, FileOpener, Decompressor

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 

from src.config import training_hyperparameters

wandb.init(config=training_hyperparameters)

from tqdm import tqdm


# testing to see if I can stream dataset from zip file directly

TSV_URLS = {
    'train': './datasets/food101/food-101.tar.gz',
    # 'val': './datasets/conceptual_captions/GCC-1.1.0-Validation.tsv'
    # 'train': 'https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250'
}

async def async_get_image(
    session: aiohttp.ClientSession, url: str
) -> Optional[Image.Image]:
    try:
        resp = await session.get(url)
        image_bytes = await resp.read()
        return Image.open(io.BytesIO(image_bytes))
    except Exception:
        # If an exception occurs, such as a timeout, invalid URL, etc, just
        # return None, and the caller can handle skipping this
        return None
    
async def async_batch_get_images(
    urls: Sequence[str], timeout: float = 1.0
) -> List[Optional[Image.Image]]:
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        return await asyncio.gather(*[async_get_image(session, url) for url in urls])


def _datapipe_from_tsv_url(
    tsv_url: str, buffer_size: int = 128
) -> IterDataPipe[Tuple[Image.Image, str]]:
    # pipe = HttpReader([tsv_url])
    # pipe = LineReader(pipe, return_path=False)

    # source_dp = IterableWrapper([(tsv_url, io.StringIO(text1)), ("file2", io.StringIO(text2))])


    pipe = dp.iter.FileOpener([tsv_url], mode='r')

    pipe = LineReader(pipe, return_path=False)
    # # use pipe to read from local file
    # pipe = LineReader(pipe, return_path=True)
    # LineReader downloads raw bytes.  Decode them to strings, then split.
    pipe = pipe.map(lambda line: line.split("\t"))

    return ParallelSampleLoader(pipe, buffer_size=buffer_size)

def conceptual_captions_3m(
    split: str = "train", buffer_size: int = 128
) -> IterDataPipe[Tuple[Image.Image, str]]:
    return _datapipe_from_tsv_url(tsv_url=TSV_URLS[split], buffer_size=buffer_size)

class ParallelSampleLoader(IterDataPipe):
    def __init__(
        self, dp: IterDataPipe[Tuple[str, str]], buffer_size: int = 128
    ) -> None:
        super().__init__()
        self.dp = dp
        self.buffer_size = buffer_size

        self.device = torch.device("cpu")


    def __iter__(self) -> Generator[Tuple[Image.Image, str], None, None]:
        pipe: IterDataPipe[List[Tuple[str, str]]] = self.dp.batch(self.buffer_size)
        for batch in pipe:
            # The batch is a list of tuples, where the first element is the
            # caption, and the second element is the URL of the image.

            # print('batch: ', batch)
            captions = [x[0] for x in batch]
            image_urls = [x[1] for x in batch]
            images = asyncio.run(async_batch_get_images(image_urls))

            for image, caption in zip(images, captions):
                if image is not None:
                    yield image, caption



# pipe = conceptual_captions_3m(split='train', buffer_size=128)

# for (img, cap) in tqdm(pipe):

#     pass





zip_data_pipe = FileOpener([TSV_URLS['train']], mode="b")

zip_loader_dp = Decompressor(zip_data_pipe, file_type="tar")

for _, stream in zip_loader_dp:
    print(stream.getnames())