from enum import Enum
class ClipModels(Enum):
    DEFAULT = "clip_default"
    FINETUNED = 'clip_finetuned'
    FINETUNED_TEMP = 'clip_finetuned_temp'

class ClipCaptionModelMapping(Enum):
    MLP = 'mlp'
    TRANSFORMER = 'transformer'

class OpenAIClipPretrainedModels(Enum):
    VIT = "ViT-B/32",
    RN50 = "RN50x4",

class HFClipPretrainedModels(Enum):
    VIT = "openai/clip-vit-base-patch32",
    RN50 = "mlunar/clip-variants-resnet-50x4",


selected_clip_model = ClipModels.FINETUNED_TEMP
# selected_clip_model = ClipModels.DEFAULT


training_hyperparameters = {
    'batch_size': 16,
    'grad_cache': False,
    'grad_cache_multiplier': 32,
    'n_epochs': 1,
    'lr': 1e-5,
    'weight_decay': 0.2,
    'model_path': 'checkpoints/my_clip_checkpoint.pt',
    'validation_dataset_size': 256,
    'validation_batch_size': 256,
    'do_checkpointing': True,
    'start_new': True,
    'use_small_trainloader': True,
    'small_train_loader_batch_size': 256,
    'small_train_loader_dataset_size': 10000,
    'openai_clip_model': OpenAIClipPretrainedModels.VIT.value[0],
    'hf_clip_model': HFClipPretrainedModels.VIT.value[0],
    }


clip_caption_prev_checkpoint_epoch = 10

clip_caption_model_train_hyperparameters = {
    'batch_size': 150,
    'n_epochs': 500,
    'save_every': 5,
    'lr': 2e-5,
    'dataset_size': 30000,
    'train_from_scratch': False,
    'continue_train_from_prev_checkpoint': True,
    'prev_checkpoint_epoch': clip_caption_prev_checkpoint_epoch,
    'model_config': ClipCaptionModelMapping.MLP,
    'only_prefix': True, # whether to train just MLP, or MLP + gpt
    'show_real_images': True,
}


clip_caption_model_weight_paths = {
    "og_mscoco": "caption_checkpoints/coco_weights.pt", # this is default
    # "og_mscoco": "caption_checkpoints/default_clip_coco_prefix-360_DEFAULT.pt", # this is default

    # "finetuned_caption": "caption_checkpoints/finetuned_clip_coco_prefix-009.pt",
    

    'finetuned_caption_temp': f"caption_checkpoints/finetuned_temp_clip_coco_prefix-{clip_caption_prev_checkpoint_epoch:03d}_FINETUNED_TEMP.pt",

}




clip_caption_transformer_model_weight_paths = {
    # 'og_mscoco': 'caption_checkpoints/transformer_coco_weights.pt',
    'finetuned_caption_temp': f"caption_checkpoints/transformer_finetuned_temp_clip_coco_prefix-{clip_caption_prev_checkpoint_epoch:03d}.pt",
}
