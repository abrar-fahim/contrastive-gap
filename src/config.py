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
    WIT400 = 'wit400',
    CONCEPTUAL_CAPTIONS = 'conceptual_captions',

# selected_clip_model = ClipModels.WARM
selected_clip_model = ClipModels.FINETUNED_TEMP
# selected_clip_model = CliModels.DEFAULT
# selected_clip_model = ClipModels.FINETUNED

'''
    1. Training CLIP
'''

config_cuda_device = 'cuda:3'

training_hyperparameters = {

    # hardware settings
    'cuda_device': 'cuda:3', # SET index of GPU
        # 'cuda_device': 'cpu', # SET index of GPU
        'host': 'cirrus', # SET 'local' or 'cirrus' # CHANGE IN LOCAL
        'seed': 2,
        'selected_clip_model': selected_clip_model.value,
        'dataset': ClipDatasets.MSCOCO.value,
        # 'dataset': ClipDatasets.CONCEPTUAL_CAPTIONS.value,
    'batch_size': 256, 
    'grad_cache': False,
    'grad_cache_multiplier': 16,
    'n_epochs': 64, # SET 12 for scratch, (6 for finetune?)
    # 'n_epochs': 10000, # SET 12 for scratch, (6 for finetune?)
    'max_steps': None, # SET or None, in which case each epoch goes through all the data
    'lr': 5e-4,
    'use_scheduler': True,
    'n_warmup_steps': 10000,
    'vision_model': 'VIT', # RN50 or VIT
    # 'vision_model': 'RN50', # RN50 or VIT



    'temperature': 0.07,
    'intra_modality_temperature': 0.01,
    'weight_decay': 0.1, # LARGER weight decay means MORE regularization
    'validation_dataset_size': 2048, # SET
    'validation_batch_size': 2048, # SET
    'cifar_batch_size': 128,
    
    'do_checkpointing': True,
    'continue_from_checkpoint': False, # False means don't loads weights from previous checkpoint
    'train_from_scratch': True, # this randomly initializes weights
    'train_from_pretrained': False,
    
    'use_small_trainloader': False, # this is ignored when using WIT400
    # 'small_train_loader_batch_size': 6, # SET
    # 'small_train_loader_dataset_size': 35000, # 30000
    # 'small_train_loader_dataset_size': 80000, # when using training set
    # 'small_train_loader_dataset_size': 6, # SO that I'm only training a single batch
    'num_workers': 24,
    'loss_weights': {
        'image_to_text_weight': 0.5,
        'text_to_image_weight': 0.5,
    },


    # these are set by wandb sweep

    # Architecture settings
    'encoder1_modality': 'image', # SET # can be 'image' or 'text'
    'encoder2_modality': 'text', # SET
    'same_encoder': False, # SET # ONLY WORKS FOR text_only=True
    'one_encoder': False, # SET # modality depends on text_only or image_only
    'common_projection_layer': False, # SET
    'W_layer_gap': -1, # SET. This controls modality gap at start. 0 means no gap, 1 means full gap. -1 means no W layer
    'shared_transformer_layers': False , # SET\
    'clip_projection_dim': 512, # SET # this is the size of the projection layer
    'finetune_multi_layer_projection': False, # SET

    # encoder configs
   
    'same_inputs': False, # SET # ONLY WORKS FOR text_only=True
    'second_caption_offset': False, # SET # ONLY WORKS FOR text encoders
    'mismatched_pairs': False, # SET 

    # loss factors
    'intra_modality_loss': False, 
    'rsa_loss': False,
    'pearson_loss': False,
    'scaled_denominator': False, # SET
    'svd_loss': False,
    'uniformity_loss': False,
    'alignment_loss': False,

    # validation batch stuff
    'train_only_one_batch': False,
    'use_train_as_val': True, # SET
    'use_cached_val_batch': True, 
    'cifar10_acc': True,
    'delete_val_batch_first': False,


    # Saving embeds and encoder hidden states
    'save_encoder_hidden_states': False, 
    'n_embeds_to_save': 256, 
    # which clip model
    'openai_clip_model': OpenAIClipPretrainedModels.VIT.value,
    'hf_clip_model': HFClipPretrainedModels.VIT.value,


    # Evaluator settings
    'visualize_embeddings': False, # CHANGE IN LOCAL
    'save_every': 200, # CHANGE IN LOCAL
    
    'save_losses': False,
    'csv_path': 'stats/',
    'loss_file_name_template': 'Ttemp_loss_seed_trainmode_captionencoder_dim_val_bsize_dataset_vmodel', # can have name, temp, iweight, tweight, loss as of now,
    'show_incorrect_images': False,
}