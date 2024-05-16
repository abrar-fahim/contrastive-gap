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
from src.utils import get_checkpoint_path, init_stats_csv_file, generate_csv_file_name, cleanup_after_training

# from src.validate import do_validation
from src.evaluator import Evaluator
import os
import random
import wandb
from src.config import *
from dataset_processors.mscoco_processor import MSCOCOProcessor
from dataset_processors.wit_processor import WITProcessor
from dataset_processors.cifar10_processor import CIFAR10Processor
from dataset_processors.conceptual_captions_processor import ConceptualCaptionsProcessor
from trainer import Trainer, GradCacheTrainer
from clips.clip_assembler import ClipAssembler
import numpy as np

from torch.cuda.amp import GradScaler

from src.scheduler import cosine_scheduler

# import torch exponential scheduler
from torch.optim.lr_scheduler import ExponentialLR





def delete_checkpoint_file(checkpoint_path):
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f'removed {checkpoint_path}')
    else:
        print(f'{checkpoint_path} does not exist')



def main():

    

    if wandb.run == None: # so that wandb doesnt reset config in case this run is part of a sweep
        wandb.init(
            project="clipverse", 
            # track hyperparameters and run metadata
            config=training_hyperparameters,
            # name=generate_csv_file_name(clip_model)
        )


    

    # set seed
    torch.manual_seed(wandb.config['seed'])
    random.seed(wandb.config['seed'])
    np.random.seed(wandb.config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"


        
    device = torch.device(config_cuda_device if torch.cuda.is_available() else "cpu")

    print('device ', device)

    dataset_processor = None

    if wandb.config['dataset'] == ClipDatasets.MSCOCO.value:
        dataset_processor = MSCOCOProcessor()
    elif wandb.config['dataset'] == ClipDatasets.WIT400.value:
        dataset_processor = WITProcessor()
    elif wandb.config['dataset'] == ClipDatasets.CONCEPTUAL_CAPTIONS.value:
        dataset_processor = ConceptualCaptionsProcessor()


    # cifar_dataset_processor = CIFAR10Processor()

    # clip_model = MyClip().to(device)
    # clip_model = OpenAIClip().to(device)

    checkpoint_path = get_checkpoint_path()

    if wandb.config['train_from_scratch']:
        # delete checkpoint file if it exists
        delete_checkpoint_file(checkpoint_path)


    clip_model = ClipAssembler().clip_model.to(device)

    # clip_model = HFClip().to(device)


    '''
    - Setup training loop
    '''



    n_epochs = wandb.config['n_epochs']

    clip_model.train()


    '''
    checkpointing stuff
    '''

    print('continuting from checkpoint ', wandb.config['continue_from_checkpoint'])

    print('training from scratch ', wandb.config['train_from_scratch'])


    i_loaded_from_checkpoint = False

    # setup adamW optimizer

    optimizer = optim.AdamW(clip_model.parameters(), lr=wandb.config['lr'], weight_decay=wandb.config['weight_decay'], betas=(0.9, 0.99))

    scaler = GradScaler()

    n_steps = dataset_processor.get_num_batches() * n_epochs

    # scheduler = cosine_scheduler(optimizer, wandb.config['lr'], wandb.config['n_warmup_steps'], n_steps)

    scheduler = ExponentialLR(optimizer, gamma=0.9)


    
    

    if os.path.exists(checkpoint_path) and wandb.config['continue_from_checkpoint'] and wandb.config['do_checkpointing']:

        

        

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


        if not wandb.config['train_from_scratch']:
            # guess I can delete checkpoint file here too. 
            delete_checkpoint_file(checkpoint_path)

            # clip_model.set_weights('default') # because CLIP loads from latest checkpoint in init for inference


    dataset_processor.print_dataset_stats()
    # cifar_dataset_processor = None

    '''
    create csv file
    '''

    if wandb.config['save_losses'] and not wandb.config['continue_from_checkpoint']:
        # only create new csv file if we're not continuing from checkpoint
        init_stats_csv_file(clip_model)





    # setup trainer
        


        

    




    '''
    Setup evaluator
    '''

    evaluator = Evaluator(dataset_processor)

    if wandb.config['grad_cache']:
        trainer = GradCacheTrainer(dataset_processor, evaluator)
    else:
        trainer = Trainer(dataset_processor, evaluator)





    # print()
    # print(f'--- VALIDATING BEFORE TRAINING BEGINS ---')
    # print()

    

    clip_model.eval()

    if wandb.config['W_layer_gap'] >= 0:
        print()
        print(f'--- SETTING W ---')
        print()

        print('W gap ', wandb.config['W_layer_gap'])
        # set W
        W = trainer.calculateW(clip_model)
        clip_model.setW(W)


    # evaluator.evaluate_model(clip_model, epoch=epoch, index=i)


    # do_validation(dataset_processor, clip_model, index=i, epoch=epoch, captioning_model=False, val_dataset_processor=cifar_dataset_processor)
    # do_validation(dataset_processor, clip_model, index=i, epoch=epoch, captioning_model=False)


   

 

    clip_model.train()

    # training loop
    while epoch < n_epochs:


        if not i_loaded_from_checkpoint:
            i = 0

        epoch = trainer.train_one_epoch(clip_model, optimizer, scaler=scaler,  scheduler=scheduler ,i=i, epoch=epoch, save_every=wandb.config['save_every'])

        # returns epoch because trainer can run multiple epochs on its own for efficiency

        

        i_loaded_from_checkpoint = False
        epoch += 1

        if wandb.config['use_scheduler']:
            scheduler.step()


    clip_model.train()

    print('--- TRAINING COMPLETE ---')



    # delete validation batch cache
    cleanup_after_training()

    
    



if __name__ == '__main__':
    main()