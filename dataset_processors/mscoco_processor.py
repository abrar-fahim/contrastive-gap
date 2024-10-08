import torchvision.datasets as dset
import clip
import torch
from src.config import *
import random
from torch.utils.data import DataLoader, Subset
from src.utils import  get_checkpoint_path
from dataset_processors.dataset_processor_parent import DatasetProcessorParent
import numpy as np
from collections import OrderedDict

from torchvision.transforms import v2
from torchvision.transforms.functional import resized_crop





class MSCOCOProcessor(DatasetProcessorParent):

    def __init__(self, return_org_imgs_collate_fn=False, return_only_captions=False, root='.') -> None:
        self.train_dataset = None
        self.full_train_dataset = None
        self.train_dataloader: DataLoader = None
        self.train_subset_indices = None
        self.val_dataset = None
        self.val_dataloader = None
        self.show_real_images_captions=False
        self.return_org_imgs_collate_fn = return_org_imgs_collate_fn
        self.return_only_captions = return_only_captions

        self.val_tokenized_captions = None

        self.use_cached_tokenized_captions = False

        self.root = root

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
        image is a tensor of shape [3, 224, 224]
        caption is a tuple of strings
        '''


        imgs: list
        og_captions: list

        imgs, og_captions = zip(*batch)


        # keep only first caption for each image
        captions = [caption[0] for caption in og_captions]
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

            # preprocessed_images = tuple(self.image_preprocessor(img) for img in imgs)

            preprocessed_images = torch.stack(imgs)

            outputs1 = preprocessed_images


        if self.encoder2_modality == 'text':
            if self.same_inputs:
                outputs2 = [caption[0] for caption in og_captions]
            else:
                outputs2 = [caption[1] for caption in og_captions]

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
        return len(self.train_dataloader)
    
    @staticmethod
    def seed_dataloader_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    
    def load_train_dataset(self):

        self.full_train_dataset = dset.CocoCaptions(
            root = f'{self.root}/datasets/mscoco2/train2017',
            # root = './datasets/mscoco/train2014',
            # root = '/Volumes/SanDisk Extreme SSD Media/clipverse/mscoco_train/train2014',
            annFile = f'{self.root}/datasets/mscoco2/annotations/captions_train2017.json',
            # annFile = 'datasets/mscoco/annotations/captions_train2014.json',
            transform=self.image_preprocessor,
        )


        dataset_to_use = None

        if wandb.config['use_small_trainloader']:

            '''
            Prepare subset of training dataset
            '''
            # subset_indices = torch.randint(0, len(self.train_dataset) , (wandb.config['small_train_loader_dataset_size'],))

            # prepare subset_indices WITHOUT replacement
            subset_indices = torch.randperm(len(self.full_train_dataset))[:wandb.config['small_train_loader_dataset_size']] 
            train_data_subset = Subset(self.full_train_dataset, subset_indices)
            dataset_to_use = train_data_subset

        else:
                
            dataset_to_use = self.full_train_dataset 

        self.train_dataloader = DataLoader(dataset_to_use, shuffle=False, collate_fn=self.collate_fn, num_workers=wandb.config['num_workers'], worker_init_fn=self.seed_dataloader_worker, generator=torch.Generator().manual_seed(wandb.config['seed']), persistent_workers=True, prefetch_factor=4,
        batch_sampler=RepeatSampler(torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(dataset_to_use), batch_size=wandb.config['batch_size'], drop_last=False)))

        self.train_dataset = dataset_to_use
        


   
    def load_val_dataset(self):
        val_dataset = dset.CocoCaptions(
            root = f'{self.root}/datasets/mscoco2/val2017',
            # root = './datasets/mscoco/val2014',
            # root = '/Volumes/SanDisk Extreme SSD Media/clipverse/mscoco copy/val2014',
            annFile= f'{self.root}/datasets/mscoco2/annotations/captions_val2017.json',
            # annFile= './datasets/mscoco/annotations/captions_val2014.json',
            transform=self.image_preprocessor,
        )

        # generate random indices

        use_subset = False

        val_dataset_to_use = None

        if use_subset:

            val_indices = torch.randperm(len(val_dataset))[:wandb.config['validation_dataset_size']]

            val_dataset_to_use = Subset(val_dataset, val_indices)

        else:

            val_dataset_to_use = val_dataset

        self.val_dataset = val_dataset_to_use # using only one validation batch for now. change to use entire val batch LATER





    def print_dataset_stats(self):

        print()
        print('--- TRAIN DATASET STATS ---')
        print()

        print('no of train samples: ', len(self.train_dataset))

        print()
        print('--- VAL DATASET STATS ---')
        print()


        print('no of val samples: ', len(self.val_dataset))

class RepeatSampler(torch.utils.data.Sampler):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

    def __len__(self):
        return len(self.sampler)

    