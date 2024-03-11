import torchvision.datasets as dset
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

from torchvision.transforms import v2
class MSCOCOProcessor(DatasetProcessorParent):

    def __init__(self, return_org_imgs_collate_fn=False, return_only_captions=False) -> None:
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

        self.encoder1_modality = training_hyperparameters['encoder1_modality']
        self.encoder2_modality = training_hyperparameters['encoder2_modality']
        self.same_inputs = training_hyperparameters['same_inputs']
        self.same_encoder = training_hyperparameters['same_encoder']
        self.second_caption_offset = training_hyperparameters['second_caption_offset']

        # set seed
        assert torch.initial_seed() == training_hyperparameters['seed'], "Seed not set properly"
        # random.seed(training_hyperparameters['seed'])
        # np.random.seed(training_hyperparameters['seed'])


        if self.same_inputs and self.encoder1_modality == 'image':
            self.same_image_transforms = v2.Compose([
                v2.RandomResizedCrop(size=(224, 224), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                v2.ToDtype(torch.float32, scale=True),
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
            outputs1 = imgs


        if self.encoder2_modality == 'text':
            if self.same_inputs:
                outputs2 = [caption[0] for caption in og_captions]
            else:
                outputs2 = [caption[1] for caption in og_captions]

        elif self.encoder2_modality == 'image':
            if self.same_inputs:
                
                outputs2 = imgs
            else:
                if self.encoder1_modality == "image":
                    # images should be augmented somehow

                    outputs2 = self.same_image_transforms(imgs)

                    # ensure that outputs2 and outputs1 are same type
                    assert type(outputs2[0]) == type(outputs1[0]), f"outputs2[0] {type(outputs2[0])} and outputs1[0] {type(outputs1[0])} are not same type"

                    assert len(outputs2) == len(outputs1), f"outputs2 {len(outputs2)} and outputs1 {len(outputs2)} are not same length"


                else:
                    outputs2 = imgs




                


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

    
    @staticmethod
    def seed_dataloader_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def load_train_dataset(self):


        
        '''
        comparing with training hyperparameters and not self.show_real_images_captions, because:
        - training hyperparameters stays constant throghout
        - self.show_real_images_captions is set to True in the middle of do_validation in validate.py
        '''

        if training_hyperparameters['show_incorrect_images'] or self.return_org_imgs_collate_fn: 
            # no preprocess here, instead have it in collate fn
            train_dataset = dset.CocoCaptions(root = './datasets/mscoco/val2014',
            annFile = 'datasets/mscoco/annotations/captions_val2014.json',
            # transform=[transforms.PILToTensor()])
            # transform=self.preprocess, # transforming in collate instead
            )
        else:
            train_dataset = dset.CocoCaptions(root = './datasets/mscoco/val2014',
            annFile = 'datasets/mscoco/annotations/captions_val2014.json',
            # transform=[transforms.PILToTensor()])
            # transform=self.preprocess,
            )



        subset_indices = torch.randint(0, len(train_dataset) , (training_hyperparameters['small_train_loader_dataset_size'],)) 

        # no problems upto here



        # always defined and exists, but only used when small training loader is used, and we're not loading from checkpoint at start

        dataset_to_use = None
        batch_size = None

        checkpoint_path = get_checkpoint_path()

        # NOT SAVING DATALOADER IN CHECKPOINT, so load dataloader normally

        # if os.path.exists(checkpoint_path) and training_hyperparameters['continue_from_checkpoint'] and training_hyperparameters['do_checkpointing']:

        #     '''
        #     Load from checkpoint``
        #     '''
        #     print('Loading dataloader from checkpoint...')

        #     checkpoint = torch.load(checkpoint_path)
        #     self.train_dataloader = checkpoint['train_dataloader']
        #     # keep self.train_dataset same as in init, since it doesnt matter
        #     return

        
        '''
        Not loading from checkpoint, so prepare new dataloader
        '''
        if training_hyperparameters['use_small_trainloader']:

            '''
            Prepare subset of training dataset
            '''
            train_data_subset = Subset(train_dataset, subset_indices)
            dataset_to_use = train_data_subset
            batch_size = training_hyperparameters['small_train_loader_batch_size']

        else:

            dataset_to_use = train_dataset
            batch_size = training_hyperparameters['batch_size']
        
        # set class variables
        self.train_dataset = dataset_to_use
        self.train_dataloader = DataLoader(dataset_to_use, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=training_hyperparameters['num_workers'], worker_init_fn=self.seed_dataloader_worker, generator=torch.Generator().manual_seed(training_hyperparameters['seed']))
        self.train_subset_indices = subset_indices

    def load_val_dataset(self):

        val_indices = torch.randint(0, len(self.train_dataset) , (training_hyperparameters['validation_dataset_size'],))

        # make sure that the validation indices are not in the training indices
        j = 0
        while j < training_hyperparameters['validation_dataset_size']:
            while val_indices[j] in self.train_subset_indices:
                val_indices[j] = torch.randint(0, len(self.train_dataset) , (1,))
            j += 1


        val_data_subset = Subset(self.train_dataset, val_indices)



        # no need val dataloader as I'm creating it in do_validation in utils

        # val_dataloader = DataLoader(val_data_subset, batch_size=training_hyperparameters['validation_batch_size'], shuffle=True, collate_fn=self.collate_fn, num_workers=training_hyperparameters['num_workers'], worker_init_fn=self.seed_dataloader_worker)


        # set class variables
        self.val_dataset = val_data_subset
        # self.val_dataloader = val_dataloader

    def print_dataset_stats(self):

        print()
        print('--- TRAIN DATASET STATS ---')
        print()

        print('no of train samples: ', len(self.train_dataset))

        print()
        print('--- VAL DATASET STATS ---')
        print()


        print('no of val samples: ', len(self.val_dataset))





        
        

        
        
        
        