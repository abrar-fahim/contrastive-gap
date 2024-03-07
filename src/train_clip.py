'''
- Setup image encoder
- Setup text encoder
- use nonlinear projection layer
- Setup toy dataset
- Setup loss fn
- Setup training loop

- Train model on toy dataset using minibatches
- Test to see if model can get high cosine similarities between images and captions of same concept
- This works as sanity check to test algorithm in general

'''

'''
- Setup toy dataset, MSCOCO for now
'''

import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clips.hf_clip import HFClip
import torch.optim as optim
import torch
from src.utils import do_validation, get_checkpoint_path, init_stats_csv_file, generate_csv_file_name
import os
import random
import wandb
from src.config import *
from dataset_processors.mscoco_processor import MSCOCOProcessor
from dataset_processors.wit_processor import WITProcessor
from dataset_processors.cifar10_processor import CIFAR10Processor
from trainer import Trainer, GradCacheTrainer
from clips.clip_assembler import ClipAssembler
import numpy as np






def main():

    

    # set seed
    torch.manual_seed(training_hyperparameters['seed'])
    random.seed(training_hyperparameters['seed'])
    np.random.seed(training_hyperparameters['seed'])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"



    device = torch.device(training_hyperparameters['cuda_device'] if torch.cuda.is_available() else "cpu")

    print('device ', device)

    dataset_processor = None

    if training_hyperparameters['dataset'] == ClipDatasets.MSCOCO:
        dataset_processor = MSCOCOProcessor()
    elif training_hyperparameters['dataset'] == ClipDatasets.WIT400:
        dataset_processor = WITProcessor()


    cifar_dataset_processor = CIFAR10Processor()

    # clip_model = MyClip().to(device)
    # clip_model = OpenAIClip().to(device)


    clip_model = ClipAssembler().clip_model.to(device)

    # clip_model = HFClip().to(device)


    '''
    - Setup training loop
    '''



    n_epochs = training_hyperparameters['n_epochs']

    clip_model.train()


    '''
    checkpointing stuff
    '''

    print('continuting from checkpoint ', training_hyperparameters['continue_from_checkpoint'])

    print('training from scratch ', training_hyperparameters['train_from_scratch'])

    checkpoint_path = get_checkpoint_path()


    i_loaded_from_checkpoint = False

    
    

    if os.path.exists(checkpoint_path) and training_hyperparameters['continue_from_checkpoint'] and training_hyperparameters['do_checkpointing']:

        # setup adamW optimizer

        optimizer = optim.AdamW(clip_model.parameters(), lr=training_hyperparameters['lr'], weight_decay=training_hyperparameters['weight_decay'])

        print()
        print('--- CONTINUING FROM CHECKPOINT ---')
        print()


        # load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        clip_model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        i = checkpoint['dataloader_enumerator_index']
        i_loaded_from_checkpoint = True

    else:
        '''
        Goes here if I dont need to load from checkpoint
        '''

        print()
        print('--- NOT LOADING FROM CHECKPOINT---')
        print()
        epoch = 0
        i = 0

        if not training_hyperparameters['train_from_scratch']:

            clip_model.set_weights('default') # because CLIP loads from latest checkpoint in init for inference

        # setup adamW optimizer

        optimizer = optim.AdamW(clip_model.parameters(), lr=training_hyperparameters['lr'], weight_decay=training_hyperparameters['weight_decay'])


    dataset_processor.print_dataset_stats()
    cifar_dataset_processor.print_dataset_stats()

    cifar_dataset_processor = None

    '''
    create csv file
    '''

    if training_hyperparameters['save_losses'] and not training_hyperparameters['continue_from_checkpoint']:
        # only create new csv file if we're not continuing from checkpoint
        init_stats_csv_file(clip_model)


    # setup trainer



    wandb.init(
        project="clipverse", 
        # track hyperparameters and run metadata
        config=training_hyperparameters,
        name=generate_csv_file_name(clip_model)
    )

    if training_hyperparameters['grad_cache']:
        trainer = GradCacheTrainer(dataset_processor, wandb)
    else:
        trainer = Trainer(dataset_processor, wandb)


    print()
    print(f'--- VALIDATING BEFORE TRAINING BEGINS ---')
    print()

    clip_model.eval()

    # do_validation(dataset_processor, clip_model, index=i, epoch=epoch, captioning_model=False, val_dataset_processor=cifar_dataset_processor)
    # do_validation(dataset_processor, clip_model, index=i, epoch=epoch, captioning_model=False)

    clip_model.train()

    # training loop
    while epoch < n_epochs:


        if not i_loaded_from_checkpoint:
            i = 0

        trainer.train_one_epoch(clip_model, optimizer, i=i, epoch=epoch, save_every=training_hyperparameters['save_every'], val_dataset_processor=cifar_dataset_processor)

        i_loaded_from_checkpoint = False
        epoch += 1

    clip_model.train()

    print('--- TRAINING COMPLETE ---')



    # delete validation batch cache
    mscoco_batch_file_path = f"datasets/mscoco/val_batch_cache_{training_hyperparameters['seed']}.pt"
    mscoco_train_dataset_batch_file_path = f"datasets/mscoco/train_batch_cache_{training_hyperparameters['seed']}.pt"

    if os.path.exists(mscoco_batch_file_path):
        os.remove(mscoco_batch_file_path)
        print(f'removed {mscoco_batch_file_path}')
    else:
        print(f'{mscoco_batch_file_path} does not exist')
    
    if os.path.exists(mscoco_train_dataset_batch_file_path):
        os.remove(mscoco_train_dataset_batch_file_path)
        print(f'removed {mscoco_train_dataset_batch_file_path}')
    else:
        print(f'{mscoco_train_dataset_batch_file_path} does not exist')

    
    



if __name__ == '__main__':
    main()