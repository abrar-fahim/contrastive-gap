import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_clip

import src.config as config

import wandb
import copy

from src.utils import generate_csv_file_name, cleanup_after_training



def set_hypers():
    config.training_hyperparameters['seed'] = wandb.config.seed
    config.training_hyperparameters['temperature'] = wandb.config.temperature
    config.training_hyperparameters['intra_modality_temperature'] = wandb.config.temperature
    config.training_hyperparameters['intra_modality_loss'] = wandb.config.intra_modality_loss
    config.training_hyperparameters['rsa_loss'] = wandb.config.rsa_loss
    config.training_hyperparameters['pearson_loss'] = wandb.config.pearson_loss
    config.training_hyperparameters['lr'] = wandb.config.lr
    config.training_hyperparameters['encoder1_modality'] = wandb.config.encoder1_modality
    config.training_hyperparameters['encoder2_modality'] = wandb.config.encoder2_modality
    config.training_hyperparameters['second_caption_offset'] = wandb.config.second_caption_offset
    config.training_hyperparameters['same_encoder'] = wandb.config.same_encoder
    config.training_hyperparameters['same_inputs'] = wandb.config.same_inputs
    config.training_hyperparameters['one_encoder'] = wandb.config.one_encoder


    # reload config
    # importlib.reload(config)
    # importlib.reload(train_clip)





default_configs = [
    {
        # Default
        'encoder1_modality': 'image',
        'encoder2_modality': 'text',
        'same_encoder': False,
        'same_inputs': False,
        'second_caption_offset': False,
        'one_encoder': False,
    },

]


# 5 configs for image only
image_configs = [
    
    # # image only
    # {
    #     'encoder1_modality': 'image',
    #     'encoder2_modality': 'image',
    #     'same_encoder': False,
    #     'same_inputs': False,
    #     'second_caption_offset': False,
    #     'one_encoder': False,
    # },
    # # image only, same encoder
    # {
    #     'encoder1_modality': 'image',
    #     'encoder2_modality': 'image',
    #     'same_encoder': True,
    #     'same_inputs': False,
    #     'second_caption_offset': False,
    #     'one_encoder': False,
    # },
    # image only, same inputs
    {
        'encoder1_modality': 'image',
        'encoder2_modality': 'image',
        'same_encoder': False,
        'same_inputs': True,
        'second_caption_offset': False,
        'one_encoder': False,
    },
    # # image only, same inputs, same encoder
    # {
    #     'encoder1_modality': 'image',
    #     'encoder2_modality': 'image',
    #     'same_encoder': True,
    #     'same_inputs': True,
    #     'second_caption_offset': False,
    #     'one_encoder': False,
    # },
    # # image only, one encoder
    # {
    #     'encoder1_modality': 'image',
    #     'encoder2_modality': 'image',
    #     'same_encoder': False,
    #     'same_inputs': False,
    #     'second_caption_offset': False,
    #     'one_encoder': True,
    # },

]

# 6 configs for text only
text_configs = []

for cfg in copy.deepcopy(image_configs):
    cfg['encoder1_modality'] = 'text'
    cfg['encoder2_modality'] = 'text'
    text_configs.append(cfg)

text_configs.append({
    'encoder1_modality': 'text',
    'encoder2_modality': 'text',
    'same_encoder': True,
    'same_inputs': True,
    'second_caption_offset': True, # use GPT2 tokenizer for second caption
    'one_encoder': False,

})

text_configs.append({
    'encoder1_modality': 'text',
    'encoder2_modality': 'text',
    'same_encoder': False,
    'same_inputs': False,
    'second_caption_offset': False, 
    'one_encoder': False,

})

text_configs.append({
    'encoder1_modality': 'text',
    'encoder2_modality': 'text',
    'same_encoder': True,
    'same_inputs': False,
    'second_caption_offset': False, 
    'one_encoder': False,
})



# so total 12 configs?

def wandb_config_valid(config):

    # all_configs = image_configs
    all_configs = default_configs + image_configs + text_configs

    # compare keys in config with keys in sweep_configuration
    for sweep_config in all_configs:
        if all(sweep_config[key] == config[key] for key in sweep_config):
            return True
        
    return False
                
        
        
            

def main():
    wandb.init()

    print('wandb config ', wandb.config)

    if not wandb_config_valid(wandb.config):
        print('wandb config not valid')
        wandb.finish()
        return

        
    



    set_hypers()

    # in case train_clip.py throws error, we can still finish the run
    try:
        # do training
        train_clip.main()
        wandb.finish() 
    except Exception as e:
        print('Exception in training ', e)
        cleanup_after_training()
        wandb.finish()
        # delete cache batches
        return 



    # do training
    # train_clip.main()
    # wandb.finish() 

# if main
if __name__ == "__main__":




    sweep_configuration = {
        "method": "grid",
        # "method": "random",
        "name": "Default CLIP with Common projection layer 2 layers, no RELU at end",
        "metric": {"goal": "maximize", "name": "val_image_classification_accuracy"},
        "parameters": {
            "temperature": {"values": [0.01]},
            # "intra_modality_loss": {"values": [True, False]},
            "intra_modality_loss": {"values": [False]},
            "rsa_loss": {"values": [False]},
            "pearson_loss": {"values": [False]},
            "training_hyperparameters": {"values": [config.training_hyperparameters]}, # just to keep track of hypers used for this sweep.
            "encoder1_modality": {"values": ["image"]},
            "encoder2_modality": {"values": ["text"]},
            "same_encoder": {"values": [False]},
            "same_inputs": {"values": [False]},
            'second_caption_offset': {'values': [False]},
            'one_encoder': {'values': [False]},
            'common_projection_layer': {'values': [True]},

            # "lr": {"max": 7e-5, "min": 1e-6},
            "lr": {'values': [0.000015]}, # 1.5e-5, optimized for 0.01 temp
            # "lr": {'values': [1e-6, 1e-5, 5e-5, 1e-4 ]}, # 1.5e-5, optimized for 0.01 temp
            # 'seed': {'values': [42, 10, 100]},
            'seed': {'values': [2]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="clipverse")

    print()
    print('--- SWEEP ID ---')
    print(sweep_id)
    print()


    # wandb.agent(sweep_id='nrjuh2de', function=main, project="clipverse")
    wandb.agent(sweep_id=sweep_id, function=main, project="clipverse")
 

