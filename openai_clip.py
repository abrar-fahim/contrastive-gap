import torch
from torch import nn
from transformers import ViTModel
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
import numpy as np
import torch.nn.functional as F
from clip_parent import ClipParent
import clip


class OpenAIClip(ClipParent):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)

    def encode_image(self, image):
        return self.model.encode_image(image)

    def encode_text(self, text):
        # assuming raw captions input, so need to tokenize and stuff
        tokenized_captions = self.tokenize_text(text)
        return self.model.encode_text(tokenized_captions)
    
    def tokenize_text(self, text):
        tokenized_captions = torch.cat([clip.tokenize(c) for c in text]).to(self.device)
        return tokenized_captions


    def forward(self, preprocessed_images, captions, scale=True):

        tokenized_captions = self.tokenize_text(captions)


        logits_per_image, logits_per_text = self.model(preprocessed_images, tokenized_captions)

        return logits_per_image, logits_per_text