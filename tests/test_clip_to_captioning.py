from src.utils import collate_fn
import src.train_clip as train_clip, clip_caption_train as clip_caption_train, clip_caption.validate_caption_model as validate_caption_model
import src.config as config
import importlib



'''

1. clip_training_toy writes to checkpoints/

2. clip_caption_train reads from checkpoints/my_clip_checkpoint.pt and writes to caption_checkpoints/

3. validate_caption_model reads from caption_checkpoints/my_clip_checkpoint.pt
'''
print()
print(' --- TRAINING CLIP --- ')
print()

# first train clip model
# clip_training_toy.main()

print()
print(' --- TRAINING CLIP CAPTION MODEL --- ')
print()


# then train clip caption model on the trained clip model above
# clip_caption_train.main()

print()
print(' --- VALIDATING CLIP CAPTION MODEL --- ')
print()


# then, validate the caption model
validate_caption_model.main()


# Now, train caption model on DEFAULT clip from scratch


# config.selected_clip_model = config.ClipModels.DEFAULT



# print()
# print(' --- TRAINING CLIP --- ')
# print()

# # first train clip model
# # clip_training_toy.main()

# print()
# print(' --- TRAINING CLIP CAPTION MODEL --- ')
# print()


# # then train clip caption model on the trained clip model above
# clip_caption_train.main()

# print()
# print(' --- VALIDATING CLIP CAPTION MODEL --- ')
# print()


# # then, validate the caption model
# # validate_caption_model.main()