import numpy as np
import torch

from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, CLIPTextModel, CLIPVisionModel, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", )

# print model parameters that have requires_grad=True
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# print value of logit_scale parameter in model
print('model.logit_scale ', model.logit_scale)