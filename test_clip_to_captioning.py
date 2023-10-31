import clip_training_toy, clip_caption_train, validate_caption_model

'''
1. clip_training_toy writes to checkpoints/

2. clip_caption_train reads from checkpoints/my_clip_checkpoint.pt and writes to caption_checkpoints/

3. validate_caption_model reads from caption_checkpoints/my_clip_checkpoint.pt
'''

# first train clip model
clip_training_toy.main()

# then train clip caption model on the trained clip model above
clip_caption_train.main()

# then, validate the caption model
validate_caption_model.main()

