from enum import Enum
class ClipModels(Enum):
    DEFAULT = "clip_default"
    FINETUNED = 'clip_finetuned'
    FINETUNED_TEMP = 'clip_finetuned_temp'

selected_clip_model = ClipModels.FINETUNED_TEMP


training_hyperparameters = {
    'batch_size': 16,
    'grad_cache': False,
    'grad_cache_multiplier': 32,
    'n_epochs': 2,
    'lr': 1e-5,
    'weight_decay': 0.2,
    'model_path': 'checkpoints/my_clip_checkpoint.pt',
    'validation_dataset_size': 256,
    'validation_batch_size': 256,
    'do_checkpointing': True,
    'start_new': False,
    'use_small_trainloader': True,
    'small_train_loader_batch_size': 256,
    'small_train_loader_dataset_size': 10000,
    }


clip_caption_model_train_hyperparameters = {
    'batch_size': 40,
    'n_epochs': 50,
    'lr': 2e-5
}

