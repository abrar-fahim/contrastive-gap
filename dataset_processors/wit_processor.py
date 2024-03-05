import clip
import torch
from src.config import *
import random
from torch.utils.data import DataLoader, Subset
from src.utils import get_checkpoint_path
from dataset_processors.dataset_processor_parent import DatasetProcessorParent
import os
import webdataset as wds
from clips.hf_clip import HFClip
import numpy as np

class WITProcessor(DatasetProcessorParent):

    def __init__(self) -> None:
        self.train_dataset = None
        self.device = training_hyperparameters['cuda_device'] if torch.cuda.is_available() else "cpu"
        _, self.preprocess = clip.load(training_hyperparameters['openai_clip_model'], device=self.device)

        self.train_dataset = None
        self.train_dataloader = None
        self.val_dataset = None
        self.val_dataloader = None

        # root_dir = '/Volumes/SanDisk Extreme SSD Media/UofA/Research/dataset/400m/laion400m-data/'
        self.root_dir = './datasets/400m/laion400m-data/'
        self.tar_count = 0
        corrupt_files = ['00000.tar','00002.tar', '00004.tar', '00007.tar', '00006.tar', '00008.tar', '00009.tar', '00010.tar', '00011.tar', '00012.tar', '00013.tar',  '00014.tar', '00015.tar']

        # count number of .tar files in root_dir
        for root, dirs, files in os.walk(self.root_dir):
            for filename in files:
                if filename.endswith('.tar') and filename not in corrupt_files:
                    tar_count += 1

        # setup 80/20 split
        train_tar_count = int(0.8 * tar_count)
        val_tar_count = tar_count - train_tar_count

        train_paths = []
        val_paths = []

        tar_index = 0

        for root, dirs, files in os.walk(self.root_dir): 
            for filename in files:
                if filename.endswith('.tar') and filename not in corrupt_files:
                    if tar_index < train_tar_count:
                        train_paths.append(os.path.join(root, filename))
                    else:
                        val_paths.append(os.path.join(root, filename))
                    tar_index += 1

        self.train_paths = train_paths
        self.val_paths = val_paths
        self.train_tar_count = train_tar_count
        self.val_tar_count = val_tar_count

        self.torch_generator = torch.Generator()
        self.torch_generator.manual_seed(training_hyperparameters['seed'])

        


        # set seed
        torch.manual_seed(training_hyperparameters['seed'])
        random.seed(training_hyperparameters['seed'])


        pass

    @staticmethod
    def collate_fn(batch):
        '''
        batch is a list of tuples?
        each tuple is of the form (image, caption)
        image is a tensor of shape [3, 224, 224]
        caption is a tuple of strings
        '''

        imgs, og_captions = zip(*batch)

        captions = list(og_captions)

        
        # tokenize captions and return tokens directly
        tokenized_captions = HFClip.static_tokenize_captions(captions)

        if clip_caption_model_train_hyperparameters['show_real_images']:
            # return (torch.stack(imgs), captions)
            return (imgs, captions)     
        
        return (torch.stack(imgs), tokenized_captions)

    def json_to_caption(json):
        return json['caption']
    
    @staticmethod
    def seed_dataloader_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def load_train_dataset(self):

        self.train_dataset = wds.WebDataset(self.train_paths).shuffle(1000, rng=random).decode("pill").to_tuple("jpg;png", "json").map_tuple(self.preprocess, self.json_to_caption).with_length(9000 * len(self.train_paths))

        if training_hyperparameters['use_small_trainloader']:
            batch_size = training_hyperparameters['small_train_loader_batch_size']
        else:
            batch_size = training_hyperparameters['batch_size']
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, num_workers=training_hyperparameters['num_workers'], collate_fn=self.collate_fn, worker_init_fn=self.seed_dataloader_worker)



    def load_val_dataset(self):
        
        self.val_dataset = wds.WebDataset(self.val_paths).shuffle(1000, rng=random).decode("pill").to_tuple("jpg;png", "json").map_tuple(self.preprocess, self.json_to_caption).with_length(9000 * len(self.val_paths))

        self.val_dataloader = DataLoader(self.val_dataset, batch_size=training_hyperparameters['validation_batch_size'], collate_fn=self.collate_fn, num_workers=training_hyperparameters['num_workers'], worker_init_fn=self.seed_dataloader_worker)

    def print_dataset_stats(self):
        print()
        print('--- TRAIN DATASET STATS ---')
        print()


        print('Number of train tar files: ', len(self.train_paths))
        print('no of train samples: ', len(self.train_paths) * 9000)

        print()
        print('--- VAL DATASET STATS ---')
        print()


        print('Number of val tar files: ', len(self.val_paths))
        print('no of val samples: ', len(self.val_paths) * 9000)

    




        


