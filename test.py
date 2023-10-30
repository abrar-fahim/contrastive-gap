
from training_utils import *

from clip_caption_train import load_model, ClipCaptionModel

model = ClipCaptionModel()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.load_state_dict(torch.load('caption_checkpoints/coco_prefix-009.py', map_location=device))

