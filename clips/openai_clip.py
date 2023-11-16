import torch
from torch import nn
from transformers import ViTModel
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
import numpy as np
import torch.nn.functional as F
from clips.clip_parent import ClipParent
import clip


class OpenAIClip(ClipParent):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # self.model = self.model.to(self.device)

        for param in self.model.parameters():
            param.requires_grad = True


        # set logit scale param requires grad to False
        # self.model.logit_scale.requires_grad = False


        
        # set transformer params to False
        # for param in self.model.transformer.parameters():
        #     param.requires_grad = False

        # set visual transformer params to False
        for param in self.model.visual.parameters():
            param.requires_grad = False


    def encode_image(self, preprocessed_images):
        preprocessed_images = preprocessed_images.to(self.device)
        return self.model.encode_image(preprocessed_images)

    def encode_text(self, captions):
        # assuming raw captions input, so need to tokenize and stuff
        tokenized_captions = self.tokenize_text(captions)
        return self.model.encode_text(tokenized_captions)
    
    def tokenize_text(self, text):
        tokenized_captions = torch.cat([clip.tokenize(c) for c in text]).to(self.device)
        tokenized_captions = tokenized_captions.to(self.device)
        return tokenized_captions


    def forward(self, preprocessed_images, captions, scale=True):

        tokenized_captions = self.tokenize_text(captions)

        tokenized_captions = tokenized_captions.to(self.device)

        preprocessed_images = preprocessed_images.to(self.device)

        # print('preprocessed_images ', preprocessed_images)

        # print('tokenized_captions ', tokenized_captions)

        logits_per_image, logits_per_text = self.model(preprocessed_images, tokenized_captions)

        # label_probabilities = softmax(logits, axis=1)

        # logits_per_image, logits_per_text = logits_per_image / 100, logits_per_text / 100

        

        # return logits_per_image.softmax(dim=-1), logits_per_text.softmax(dim=-1)
        return logits_per_image, logits_per_text