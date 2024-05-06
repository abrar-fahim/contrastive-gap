from datasets import load_dataset
import clip
import torch
from src.config import *
import random
from torch.utils.data import DataLoader, Subset
from src.utils import  get_checkpoint_path
from dataset_processors.dataset_processor_parent import DatasetProcessorParent
from clips.hf_clip import HFClip
import numpy as np
from collections import OrderedDict
import torchdata.datapipes as dp

from torchvision.transforms import v2
from torchvision.transforms.functional import resized_crop


import aiohttp
from PIL import Image
import io
from typing import Optional
from torchdata.datapipes.iter import IterDataPipe
# import Generator
from typing import Generator, List, Tuple, Sequence
import asyncio
from torchdata.datapipes.iter import HttpReader, LineReader

TSV_URLS = {
    'train': './datasets/conceptual_captions/Train-GCC-training.tsv',
    'val': './datasets/conceptual_captions/GCC-1.1.0-Validation.tsv'
    # 'train': 'https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250'
}

class ConceptualCaptionsProcessor(DatasetProcessorParent):

    

    def __init__(self, return_org_imgs_collate_fn=False, return_only_captions=False) -> None:

        self.train_data_pipe: IterDataPipe = None
        self.val_data_pipe: IterDataPipe = None

        self.train_dataset = None
        self.train_dataset = None
        self.train_dataloader = None
        self.train_subset_indices = None
        self.val_dataset = None
        self.val_dataloader = None
        self.show_real_images_captions=False
        self.return_org_imgs_collate_fn = return_org_imgs_collate_fn
        self.return_only_captions = return_only_captions

        self.val_tokenized_captions = None

        self.use_cached_tokenized_captions = False

        self.device = torch.device(config_cuda_device if torch.cuda.is_available() else "cpu")

        self.encoder1_modality = wandb.config['encoder1_modality']
        self.encoder2_modality = wandb.config['encoder2_modality']
        self.same_inputs = wandb.config['same_inputs']
        self.same_encoder = wandb.config['same_encoder']
        self.second_caption_offset = wandb.config['second_caption_offset']

        _, self.image_preprocessor = clip.load(wandb.config['openai_clip_model'], device=self.device)

        # set seed
        assert torch.initial_seed() == wandb.config['seed'], "Seed not set properly"
        # random.seed(wandb.config['seed'])
    # np.random.seed(wandb.config['seed'])


        if not self.same_inputs and self.encoder1_modality == self.encoder2_modality == 'image':
            self.same_image_transforms = v2.Compose([
                # v2.RandomResizedCrop(size=(224, 224), scale=(0.2, 0.5), antialias=True),
                
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                # v2.ToDtype(torch.float32, scale=True),
            ])



        # always need to first load train then load val dataset. Fix this confusing requirement later
        self.load_train_dataset()
        self.load_val_dataset()


    def collate_fn(self, batch):
        '''
        batch is a list of tuples?
        each tuple is of the form (image, caption)
        image is a jpeg image
        caption is a tuple of strings
        '''


        imgs: list
        og_captions: list

        imgs, og_captions = zip(*batch)

        try:


            imgs = tuple(self.image_preprocessor(img.convert("RGBA")) for img in imgs)
            # imgs = tuple(self.image_preprocessor(img) for img in imgs)

        except Exception as e:

            print('Exception in collate_fn: ', e)

            return (None, None)





        # keep only first caption for each image
        # captions = [caption[0] for caption in og_captions]

        captions = og_captions
        # remove repeats in captions and imgs

        org_len = len(captions)
        # get indices of unique captions
        
        # unique_captions = list(set(captions))
        unique_captions = list(OrderedDict.fromkeys(captions))
        unique_captions_indices = [captions.index(caption) for caption in unique_captions]

        # get unique imgs
        imgs = [imgs[i] for i in unique_captions_indices]

        # count repeats
        n_repeats = org_len - len(unique_captions)
        # print('n_repeats: ', n_repeats)

        captions = unique_captions

        og_captions = [og_captions[i] for i in unique_captions_indices]

        if self.encoder1_modality == 'text':
            outputs1 = captions

        elif self.encoder1_modality == 'image':

            # outputs1 = imgs

            

            preprocessed_images = torch.stack(imgs)

            outputs1 = preprocessed_images


        if self.encoder2_modality == 'text':
            if self.same_inputs:
                # outputs2 = [caption[0] for caption in og_captions]
                outputs2 = captions
            else:
                # outputs2 = [caption[1] for caption in og_captions]
                outputs2 = captions

                if self.encoder1_modality == 'text':
                    raise NotImplementedError("There is only one caption to sample from in conceptual captions")
            

        elif self.encoder2_modality == 'image':
            if self.same_inputs:
                
                outputs2 = outputs1
            else:
                if self.encoder1_modality == "image":
                    # images should be augmented somehow

                    # imgs2 = [self.same_image_transforms(img) for img in imgs]
                    imgs2 = [resized_crop(img, size=(224, 224), top=50, left=50, height=100, width=100, antialias=True) for img in imgs]

                    

                    assert len(imgs2) == len(outputs1), f"outputs2 {len(imgs2)} and outputs1 {len(imgs2)} are not same length"

                    # preprocessed_images = tuple(self.image_preprocessor(img) for img in imgs2)

                    preprocessed_images = torch.stack(imgs2)

                    outputs2 = preprocessed_images

                    # ensure that outputs2 and outputs1 are same type
                    assert type(outputs2[0]) == type(outputs1[0]), f"outputs2[0] {type(imgs2[0])} and outputs1[0] {type(outputs1[0])} are not same type"




                else:
                    # outputs2 = imgs

                    # preprocessed_images = tuple(self.image_preprocessor(img) for img in imgs)



                    preprocessed_images = torch.stack(imgs)

                    outputs2 = preprocessed_images




                
        if wandb.config['mismatched_pairs']:

            if self.encoder2_modality == 'text':

                assert type(outputs2) == list, f"encoder2 is texts, but outputs2 is not a list: its  {type(outputs2)}"
                # shuffle outputs2
                # outputs2 = random.sample(outputs2, len(outputs2))

                # shift outputs2 by 1
                outputs2 = [outputs2[-1]] + outputs2[:-1]

            elif self.encoder2_modality == 'image':

                assert type(outputs2) == torch.Tensor, f"encoder2 is images, but outputs2 is not a tensor: its {type(outputs2)}"

                # shuffle outputs2
                # outputs2 = outputs2[torch.randperm(outputs2.size()[0])]


                # shift outputs2 by 1
                outputs2 = torch.roll(outputs2, shifts=1, dims=0)

        return (outputs1, outputs2)



        

        if self.text_only:

            if self.same_inputs:

                # if self.second_caption_offset:
                    # add a constant string to each caption
                    # captions_2 = ['A picture of ' + caption[0] for caption in og_captions]

                    # shuffle the letters in each of captions_2
                    # captions_2 = [' '.join(random.sample(caption[0].split(), len(caption[0].split()))) for caption in og_captions]
                # else:
            
                captions2 = [caption[0] for caption in og_captions]
            else:
                captions2 = [caption[1] for caption in og_captions]



        
        if self.return_only_captions:
            return captions


        if clip_caption_model_train_hyperparameters['show_real_images']:
            # return (torch.stack(imgs), captions)
            return (imgs, captions)    
        
        if self.show_real_images_captions:
            return (imgs, captions)

        if self.text_only:
            return (captions2, captions) # since dataloader is imgs, captions format
        
        # stacked_images = stacked_images.to(device)

        return (imgs, captions)

        if self.return_org_imgs_collate_fn:

            preprocessed_imgs = tuple(self.preprocess(img) for img in imgs)

            stacked_preprocessed_images = torch.stack(preprocessed_imgs)


            return (stacked_preprocessed_images, tokenized_captions, imgs, captions)
        
        
        stacked_images = torch.stack(imgs) 
        return (stacked_images, tokenized_captions)


    def get_num_batches(self) -> int:

        return 3318333 // wandb.config['batch_size']
        return len(self.train_dataloader)
    
    @staticmethod
    def seed_dataloader_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)




    def load_train_dataset(self):

        self.train_data_pipe = conceptual_captions_3m(split="train", buffer_size=256)

        batch_size = wandb.config['batch_size']


        self.train_dataloader = DataLoader(self.train_data_pipe, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=wandb.config['num_workers'], worker_init_fn=self.seed_dataloader_worker, generator=torch.Generator().manual_seed(wandb.config['seed']))

    def load_val_dataset(self):

        self.val_data_pipe = conceptual_captions_3m(split="val")

        # val_indices = torch.randint(0, 15840 , (wandb.config['validation_dataset_size'],))

        val_indices = torch.arange(0, wandb.config['validation_dataset_size'])
        val_data_subset = Subset(self.val_data_pipe, val_indices)



        # no need val dataloader as I'm creating it in do_validation in utils

        # val_dataloader = DataLoader(val_data_subset, batch_size=wandb.config['validation_batch_size'], shuffle=True, collate_fn=self.collate_fn, num_workers=wandb.config['num_workers'], worker_init_fn=self.seed_dataloader_worker)


        # set class variables
        self.val_dataset = val_data_subset
        # self.val_dataloader = val_dataloader

    def print_dataset_stats(self):

        print()
        print('--- TRAIN DATASET STATS ---')
        print()

        print('no of train samples: ', 3318333)

        print()
        print('--- VAL DATASET STATS ---')
        print()


        print('no of val samples: ', 15840)



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

def package_images_captions(batch):
    # The batch is a list of tuples, where the first element is the
    # caption, and the second element is the URL of the image.

    # print('batch: ', batch)
    captions = [x[0] for x in batch]
    image_urls = [x[1] for x in batch]
    images = asyncio.run(async_batch_get_images(image_urls))

    for image, caption in zip(images, captions):
        if image is not None:
            yield image, caption
def _datapipe_from_tsv_url(
    tsv_url: str, buffer_size: int = 256
) -> IterDataPipe[Tuple[Image.Image, str]]:
    # pipe = HttpReader([tsv_url])
    # pipe = LineReader(pipe, return_path=False)

    # source_dp = IterableWrapper([(tsv_url, io.StringIO(text1)), ("file2", io.StringIO(text2))])

    datapipe = (
        dp.iter.FileOpener([tsv_url], mode='r')
        .readlines(return_path=False)
        .shuffle()
        .sharding_filter()
        .map(lambda line: line.split("\t"))
        .batch(buffer_size)
        
        
        # .map(lambda x: package_images_captions(x))
    )

    # pipe = pipe.sharding_filter()

    # pipe = LineReader(pipe, return_path=False)
    # # # use pipe to read from local file
    # # pipe = LineReader(pipe, return_path=True)
    # # LineReader downloads raw bytes.  Decode them to strings, then split.

    
    # pipe = pipe.map(lambda line: line.split("\t"))

    return ParallelSampleLoader(datapipe)
    # return datapipe

def conceptual_captions_3m(
    split: str = "train", buffer_size: int = 256
) -> IterDataPipe[Tuple[Image.Image, str]]:
    return _datapipe_from_tsv_url(tsv_url=TSV_URLS[split], buffer_size=buffer_size)




class ParallelSampleLoader(IterDataPipe):
    def __init__(
        self, dp: IterDataPipe[Tuple[str, str]]
    ) -> None:
        super().__init__()
        self.dp = dp

    def __iter__(self) -> Generator[Tuple[Image.Image, str], None, None]:
        # pipe: IterDataPipe[List[Tuple[str, str]]] = self.dp.batch(self.buffer_size)
        pipe = self.dp
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