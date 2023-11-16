import torchvision.datasets as dset
import clip
import torch
from src.config import *
import random
from torch.utils.data import DataLoader, Subset
from src.utils import collate_fn, get_checkpoint_path
from dataset_processors.dataset_processor_parent import DatasetProcessorParent
import os

class MSCOCOProcessor(DatasetProcessorParent):

    def __init__(self) -> None:
        self.train_dataset = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.preprocess = clip.load(training_hyperparameters['openai_clip_model'], device=self.device)

        self.train_dataset = None
        self.train_dataloader = None
        self.train_subset_indices = None
        self.val_dataset = None
        self.val_dataloader = None

        # set seed
        torch.manual_seed(42)
        random.seed(42)

        self.load_train_dataset()
        self.load_val_dataset()

    def load_train_dataset(self):
        train_dataset = dset.CocoCaptions(root = './datasets/mscoco/val2014',
        annFile = 'datasets/mscoco/annotations/captions_val2014.json',
        # transform=[transforms.PILToTensor()])
        transform=self.preprocess,
        )

        subset_indices = torch.randint(0, len(train_dataset) , (training_hyperparameters['small_train_loader_dataset_size'],)) 
        # always defined and exists, but only used when small training loader is used, and we're not loading from checkpoint at start

        dataset_to_use = None
        batch_size = None

        checkpoint_path = get_checkpoint_path()

        if os.path.exists(checkpoint_path) and training_hyperparameters['continue_from_checkpoint'] and training_hyperparameters['do_checkpointing']:

            '''
            Load from checkpoint
            '''
            print('Loading dataloader from checkpoint...')

            checkpoint = torch.load(checkpoint_path)
            self.train_dataloader = checkpoint['train_dataloader']
            # keep self.train_dataset same as in init, since it doesnt matter
            return

        
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
        self.train_dataloader = DataLoader(dataset_to_use, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
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
        val_dataloader = DataLoader(val_data_subset, batch_size=training_hyperparameters['validation_batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)

        # set class variables
        self.val_dataset = val_data_subset
        self.val_dataloader = val_dataloader

    def print_dataset_stats(self):

        print()
        print('--- TRAIN DATASET STATS ---')
        print()

        print('no of train samples: ', len(self.train_dataset))

        print()
        print('--- VAL DATASET STATS ---')
        print()


        print('no of val samples: ', len(self.val_dataset))





        
        

        
        
        
        