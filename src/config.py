from enum import Enum
import wandb
class ClipModels(Enum):
    DEFAULT = "clip_default"
    FINETUNED = 'clip_finetuned'
    FINETUNED_TEMP = 'clip_finetuned_temp'
    WARM = 'clip_warm'

class ClipCaptionModelMapping(Enum):
    MLP = 'mlp'
    TRANSFORMER = 'transformer'

class OpenAIClipPretrainedModels(str, Enum):
    VIT = "ViT-B/32",
    RN50 = "RN50x4",

class HFClipPretrainedModels(str, Enum):
    VIT = "openai/clip-vit-base-patch32",
    RN50 = "mlunar/clip-variants-resnet-50x4",

class ClipDatasets(str, Enum):
    MSCOCO = 'mscoco',
    WIT400 = 'wit400'

# selected_clip_model = ClipModels.WARM
selected_clip_model = ClipModels.FINETUNED_TEMP
# selected_clip_model = CliModels.DEFAULT
# selected_clip_model = ClipModels.FINETUNED

'''
    1. Training CLIP
'''

config_cuda_device = 'cuda:0'

training_hyperparameters = {

    # hardware settings
    'cuda_device': 'cuda:0', # SET index of GPU
    'host': 'cirrus', # SET 'local' or 'cirrus' # CHANGE IN LOCAL



    'seed': 2,
    'selected_clip_model': selected_clip_model.value,
    'dataset': ClipDatasets.MSCOCO.value,
    'batch_size': 128, 
    'grad_cache': False,
    'grad_cache_multiplier': 16,
    'n_epochs': 4000, # SET 12 for scratch, (6 for finetune?)
    'max_steps': None, # SET or None, in which case each epoch goes through all the data
    'lr': 1.5e-5,
    'use_scheduler': True,
    'n_warmup_steps': 100,
    'vision_model': 'VIT', # RN50 or VIT



    'temperature': 0.01,
    'intra_modality_temperature': 0.01,
    'weight_decay': 0.2,
    'validation_dataset_size': 32, # SET
    'validation_batch_size': 32, # SET
    'cifar_batch_size': 128,
    'use_cached_val_batch': True, 
    'do_checkpointing': True,
    'continue_from_checkpoint': False, # False means don't loads weights from previous checkpoint
    'train_from_scratch': True, # this randomly initializes weights
    
    'use_small_trainloader': True, # this is ignored when using WIT400
    'small_train_loader_batch_size': 32, # SET
    # 'small_train_loader_dataset_size': 35000, # 30000
    'small_train_loader_dataset_size': 32, # SO that I'm only training a single batch
    'num_workers': 0,
    'loss_weights': {
        'image_to_text_weight': 0.5,
        'text_to_image_weight': 0.5,
    },


    # these are set by wandb sweep

    # encoder modalities
    'encoder1_modality': 'image', # SET # can be 'image' or 'text'
    'encoder2_modality': 'image', # SET

    # encoder configs
    'same_encoder': False, # SET # ONLY WORKS FOR text_only=True
    'same_inputs': False, # SET # ONLY WORKS FOR text_only=True
    'second_caption_offset': False, # SET # ONLY WORKS FOR text encoders
    'one_encoder': True, # SET # modality depends on text_only or image_only
    'mismatched_pairs': False, # SET 
    'common_projection_layer': False, # SET

    'W_layer_gap': -1, # SET. This controls modality gap at start. 0 means no gap, 1 means full gap. -1 means no W layer


    # loss factors
    'intra_modality_loss': False, 
    'rsa_loss': False,
    'pearson_loss': False,
    'scaled_denominator': False, # SET


    # validation batch stuff
    'train_only_one_batch': False,
    'use_train_as_val': True, # SET


    # Saving embeds and encoder hidden states
    'save_encoder_hidden_states': False, 
    'n_embeds_to_save': 256, 
    # which clip model
    'openai_clip_model': OpenAIClipPretrainedModels.VIT.value,
    'hf_clip_model': HFClipPretrainedModels.VIT.value,


    # Evaluator settings
    'visualize_embeddings': False, # CHANGE IN LOCAL
    'save_every': 25, # CHANGE IN LOCAL
    
    'save_losses': False,
    'csv_path': 'stats/',
    'loss_file_name_template': 'Ttemp_Wiweight_tweight_loss_seed_trainmode_captionencoder', # can have name, temp, iweight, tweight, loss as of now,
    'show_incorrect_images': False,
    }



'''
2. Training CLIP caption model
'''


clip_caption_prev_checkpoint_epoch = 3

clip_caption_model_train_hyperparameters = {
    'batch_size': 150,
    'n_epochs': 10,
    'save_every': 1,
    'lr': 2e-5,
    'dataset_size': 30000,
    'train_from_scratch': False,
    'continue_train_from_prev_checkpoint': False,
    'prev_checkpoint_epoch': clip_caption_prev_checkpoint_epoch,
    'model_config': ClipCaptionModelMapping.MLP,
    'only_prefix': True, # whether to train just MLP, or MLP + gpt
    'show_real_images': False,
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