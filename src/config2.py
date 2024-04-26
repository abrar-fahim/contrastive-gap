

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