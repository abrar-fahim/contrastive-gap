'''
Prepare config details for evaluating CLIP on the most common settings
'''


def prepare_config(clip_projection_dim = 128, cuda_device='cuda', seed=2, wandb_enabled=True):

    '''
    Prepare config details for evaluating CLIP on the most common settings


    -------


    >>> prepare_config(clip_projection_dim = 128, cuda_device='cuda', seed=2)

    >>> checkpoint = torch.load(checkpoint_path, map_location=device)

    >>> clip_model = ClipAssembler().clip_model.to(device)

    >>> model_state_dict = checkpoint['model_state_dict']
    >>> clip_model.load_state_dict(model_state_dict)



    '''

    import sys
    import os
    import wandb
    import random
    import numpy as np

    import torch

    # add parent directory to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    # add sibling directory to path 
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 

    from src.config import training_hyperparameters, ClipDatasets

    # from src.config import *

    # dataset stuff
    # training_hyperparameters['dataset'] = ClipDatasets.CONCEPTUAL_CAPTIONS.value
    # training_hyperparameters['validation_dataset_size'] = 16000
    # training_hyperparameters['validation_batch_size'] = 16000
    # training_hyperparameters['use_small_trainloader'] = True
    # training_hyperparameters['small_train_loader_dataset_size'] = 32
    # training_hyperparameters['use_train_as_val'] = False


    
    training_hyperparameters['validation_dataset_size'] = 5000
    training_hyperparameters['validation_batch_size'] = 5000
    training_hyperparameters['dataset'] = ClipDatasets.MSCOCO.value

    config_cuda_device = cuda_device

    training_hyperparameters['temperature'] = 0.01
    training_hyperparameters['encoder1_modality'] = 'image'
    training_hyperparameters['encoder2_modality'] = 'text'
    training_hyperparameters['same_inputs'] = False
    training_hyperparameters['clip_projection_dim'] = clip_projection_dim
    training_hyperparameters['vision_model'] = 'VIT'
    


    training_hyperparameters['seed'] = seed
    training_hyperparameters['train_from_scratch'] = False


    training_hyperparameters['continue_from_checkpoint'] = False
    training_hyperparameters['train_from_pretrained'] = True
    training_hyperparameters['finetune_clip_backbone'] = True
    training_hyperparameters['finetune_multi_layer_projection'] = False

    training_hyperparameters['cuda_device'] = config_cuda_device
    training_hyperparameters['num_workers'] = 12

    if wandb_enabled:

        wandb.init(config=training_hyperparameters)
    else:
        wandb.init(config=training_hyperparameters, mode="disabled")


    # set seed
    torch.manual_seed(wandb.config['seed'])
    random.seed(wandb.config['seed'])
    np.random.seed(wandb.config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"



    device = torch.device(config_cuda_device if torch.cuda.is_available() else "cpu")



def prepare_default_clip_config(cuda_device='cuda', seed=2, wandb_enabled=True):

    '''
    Prepare config details for evaluating CLIP on the most common settings


    -------


    >>> prepare_config(clip_projection_dim = 128, cuda_device='cuda', seed=2)

    >>> checkpoint = torch.load(checkpoint_path, map_location=device)

    >>> clip_model = ClipAssembler().clip_model.to(device)

    >>> model_state_dict = checkpoint['model_state_dict']
    >>> clip_model.load_state_dict(model_state_dict)



    '''

    import sys
    import os
    import wandb
    import random
    import numpy as np

    import torch

    # add parent directory to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    # add sibling directory to path 
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 

    from src.config import training_hyperparameters, ClipDatasets

    # from src.config import *

    # dataset stuff
    # training_hyperparameters['dataset'] = ClipDatasets.CONCEPTUAL_CAPTIONS.value
    # training_hyperparameters['validation_dataset_size'] = 16000
    # training_hyperparameters['validation_batch_size'] = 16000
    # training_hyperparameters['use_small_trainloader'] = True
    # training_hyperparameters['small_train_loader_dataset_size'] = 32
    # training_hyperparameters['use_train_as_val'] = False


    
    training_hyperparameters['validation_dataset_size'] = 5000
    training_hyperparameters['validation_batch_size'] = 5000
    training_hyperparameters['dataset'] = ClipDatasets.MSCOCO.value

    config_cuda_device = cuda_device

    training_hyperparameters['temperature'] = 0.01
    training_hyperparameters['encoder1_modality'] = 'image'
    training_hyperparameters['encoder2_modality'] = 'text'
    training_hyperparameters['same_inputs'] = False
    training_hyperparameters['clip_projection_dim'] = 512
    training_hyperparameters['vision_model'] = 'VIT'
    


    training_hyperparameters['seed'] = seed
    training_hyperparameters['train_from_scratch'] = False


    training_hyperparameters['continue_from_checkpoint'] = False
    training_hyperparameters['train_from_pretrained'] = True
    training_hyperparameters['finetune_clip_backbone'] = False
    training_hyperparameters['finetune_multi_layer_projection'] = False

    training_hyperparameters['cuda_device'] = config_cuda_device
    training_hyperparameters['num_workers'] = 12

    if wandb_enabled:

        wandb.init(config=training_hyperparameters)
    else:
        wandb.init(config=training_hyperparameters, mode="disabled")


    # set seed
    torch.manual_seed(wandb.config['seed'])
    random.seed(wandb.config['seed'])
    np.random.seed(wandb.config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"



    device = torch.device(config_cuda_device if torch.cuda.is_available() else "cpu")
