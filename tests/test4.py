import torch

import sys
import os
import wandb
import random
import numpy as np

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 

sys.path.append('/remote/cirrus-home/afahim2/tmp/clip-project/clipverse')



from src.config import training_hyperparameters

from src.config import *

from clips.clip_assembler import ClipAssembler

def load_ft_clip(model_name:str = 'abc', pretrained: str = 'mscoco_default', device='cuda', cache_dir=None):

    config_cuda_device = 'cuda'

    training_hyperparameters['temperature'] = 0.01
    training_hyperparameters['encoder1_modality'] = 'image'
    training_hyperparameters['encoder2_modality'] = 'text'
    training_hyperparameters['same_inputs'] = False
    training_hyperparameters['clip_projection_dim'] = 128
    training_hyperparameters['vision_model'] = 'VIT'
    training_hyperparameters['seed'] = 2
    training_hyperparameters['train_from_scratch'] = False


    training_hyperparameters['continue_from_checkpoint'] = False
    training_hyperparameters['train_from_pretrained'] = True
    training_hyperparameters['finetune_clip_backbone'] = True
    training_hyperparameters['finetune_multi_layer_projection'] = False

    training_hyperparameters['cuda_device'] = config_cuda_device
    training_hyperparameters['num_workers'] = 12



    wandb.init(config=training_hyperparameters)


    # set seed
    torch.manual_seed(wandb.config['seed'])
    random.seed(wandb.config['seed'])
    np.random.seed(wandb.config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    device = torch.device(config_cuda_device if torch.cuda.is_available() else "cpu")

    clip_assembler = ClipAssembler()

    clip_model = clip_assembler.clip_model.to(device)

    # tokenizer = clip_model.get_text_encoder().tokenizer

    

    tokenizer = clip_assembler.openai_clip_tokenizer

    print(' -- UPDATED -- ')

    transform = clip_model.get_image_encoder().preprocessor

    concaps_paths = [
        '/remote/cirrus-home/afahim2/tmp/clip-project/clipverse/checkpoints/T0.01_Lit_44_finetune_I1C2E1E2_128_val_as_val_512_conceptual_captions_VIT_pretrained_POST_PAPER.pt',
        '/remote/cirrus-home/afahim2/tmp/clip-project/clipverse/checkpoints/T0.01_Lituniform_align_xuniform_44_finetune_I1C2E1E2_128_val_as_val_512_conceptual_captions_VIT_pretrained_POST_PAPER.pt'
    ]

    mscoco_paths_128d = {
        'mscoco_default':  '/remote/cirrus-home/afahim2/tmp/clip-project/clipverse/checkpoints/T0.01_Lit_44_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
        'mscoco_cua':  '/remote/cirrus-home/afahim2/tmp/clip-project/clipverse/checkpoints/T0.01_Lituniform_align_44_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt',
        'mscoco_cuaxu': '/remote/cirrus-home/afahim2/tmp/clip-project/clipverse/checkpoints/T0.01_Lituniform_align_xuniform_44_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL10.pt'

    }


    checkpoint_path = mscoco_paths_128d[pretrained]


    # checkpoint = torch.load(default_checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_state_dict = checkpoint['model_state_dict']

    clip_model.load_state_dict(model_state_dict)

    text = ['a photo of a cat', 'a photo of a dog and a cat']

    hf_tokenizer  = clip_model.get_text_encoder().tokenizer

    clip_tokenizer = clip_assembler.openai_clip_tokenizer

    hf_tokenized_text = hf_tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    clip_tokenized_text = clip_tokenizer(text)

    print(f'HF Tokenized Text: {hf_tokenized_text}')
    print(f'Clip Tokenized Text: {clip_tokenized_text}')

    openai_clip_attn_mask = clip_tokenized_text.ne(0).int()

    print(f'OpenAI Clip Attention Mask: {openai_clip_attn_mask}')

    return (clip_model, transform, tokenizer)



load_ft_clip()