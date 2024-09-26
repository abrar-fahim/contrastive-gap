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
    FLICKR30K = 'flickr30k'

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
    'cuda_device': config_cuda_device, # SET index of GPU
    # 'cuda_device': 'cpu', # SET index of GPU
    'host': 'cirrus', 
    'seed': 44,
    'selected_clip_model': selected_clip_model.value, # clip_finetuned_temp
    'dataset': ClipDatasets.MSCOCO.value,
    'batch_size': 64, 
    'grad_cache': False,
    'grad_cache_multiplier': 16,
    'n_epochs': 9, 
    'max_steps': -1, # SET or -1, in which case each epoch goes through all the data
    'lr': 1e-6,
    'use_scheduler': 'none', # can be EXP or COSINE or NONE
    'schedule_every': 400, # num steps, NOT epochs
    'n_warmup_steps': 10000,
    'vision_model': 'VIT', # RN50 or VIT
    # 'vision_model': 'RN50', # RN50 or VIT
    



    'temperature': 0.01,
    'learnable_temperature': False,
    'intra_modality_temperature': 0.01,
    'weight_decay': 0.1, 
    'validation_dataset_size': 512, # SET
    'validation_batch_size': 512, # SET
    'cifar_batch_size': 128,
    

    # TRAINING SETTINGS
    'train_from_scratch': {'values': [False]}, # this randomly initializes weights
    'continue_from_checkpoint': {'values': [False]} , # False means don't loads weights from previous checkpoint
    'train_from_pretrained': {'values': [True]}, # FOR FINE-TUNING
    'finetune_clip_backbone': {'values': [True]}, # FOR FINE-TUNING
    'finetune_multi_layer_projection': {'values': [False]},
    'do_checkpointing': True,
    'save_every': 200,

    
    'use_small_trainloader': False, # this is ignored when using WIT400
    'small_train_loader_batch_size': 6, # SET
    'num_workers': 16,
    'zero_shot_acc_num_workers': 4,

    'i_t_loss_weight': 0.5,
    't_i_loss_weight': 0.5,
    # 'loss_weights': {
    #     'image_to_text_weight': 0.5,
    #     'text_to_image_weight': 0.5,
    # },


    # Architecture settings
    'encoder1_modality': 'image', # SET # can be 'image' or 'text'
    'encoder2_modality': 'text', # SET
    'same_encoder': False, # SET # ONLY WORKS FOR text_only=True
    'one_encoder': False, # SET # modality depends on text_only or image_only
    'common_projection_layer': False, # SET
    'W_layer_gap': -1, # SET. This controls modality gap at start. 0 means no gap, 1 means full gap. -1 means no W layer
    'shared_transformer_layers': False , # SET\

    
    'clip_projection_dim': 128, # SET # this is the size of the projection layer




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
    'cross_uniformity_loss': False, # ONLY IMPLEMENTED AS PART OF UNIFORM+ALIGN+XUNIFORM    
    'alignment_loss': False,
    'remove_contrastive_loss': False, # ONLY IMPLEMENTED AS PART OF UNIFORM+ALIGN+XUNIFORM-CONTRASTIVE
    'cyclip_loss': False,
    'uniform_cyclic_loss': False,
    'cyclic_direction_loss': False,
    'cosine_uniformity_loss': False,
    'cosine_align_loss': False,
    'simclr_loss': False,
    # validation batch stuff
    'train_only_one_batch': False,
    'use_train_as_val': False, # SET
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
    

    'save_losses': False,
    'csv_path': 'stats/',
    'loss_file_name_template': 'Ttemp_loss_seed_trainmode_captionencoder_dim_val_bsize_dataset_vmodel_pretrained_POST_PAPER', # can have name, temp, iweight, tweight, loss as of now,
    # 'loss_file_name_template': 'Ttemp_loss_seed_trainmode_captionencoder_dim_val_bsize_dataset_vmodel_pretrained_FINAL2', # can have name, temp, iweight, tweight, loss as of now,
    'show_incorrect_images': False,
}