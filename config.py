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
    'batch_size': 128,
    'n_epochs': 50,
    'lr': 2e-5,
    'train_from_scratch': True,
    'continue_train_from_prev_checkpoint': True
}


clip_caption_model_weight_paths = {
    "og_mscoco": "caption_checkpoints/coco_weights.pt", # this is default

    "finetuned_caption": "caption_checkpoints/finetuned_clip_coco_prefix-009.pt",

    "finetuned_caption_temp": "caption_checkpoints/finetuned_temp_clip_coco_prefix-049_notfromscratch.pt"

}
