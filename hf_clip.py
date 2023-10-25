import torch
from torch import nn
from transformers import ViTModel
import numpy as np
import torch.nn.functional as F
from clip_parent import ClipParent
import clip
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, CLIPTextModel, CLIPVisionModel
from PIL import Image
import requests


class HFClip(ClipParent):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)


        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

        self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    
        self.model = self.model.to(self.device)

    def encode_image(self, preprocessed_images):

        preprocessed_images = preprocessed_images.to(self.device)

        outputs = self.vision_model(pixel_values=preprocessed_images)

        last_hidden_states = outputs.last_hidden_state
        pooled_output = outputs.pooler_output # the last image encoder output just before linear projection. shape: ([batch_size, 512])

        return pooled_output
    
    def encode_text(self, captions):
        # assuming raw captions input, so need to tokenize and stuff
        tokenized_captions = self.tokenizer(captions, padding=True, return_tensors="pt")

        tokenized_captions = tokenized_captions.to(self.device)

        outputs = self.text_model(**tokenized_captions)

        last_hidden_states = outputs.last_hidden_state
        pooled_output = outputs.pooler_output # pooled (EOS token) states, text encoding just before CLIP's linear projection. shape: ([batch_size, 512])

        return pooled_output
    
    def forward(self, preprocessed_images, captions, scale=True):

        # inputs = self.processor(text=['captions', 'hello'], images=image, return_tensors="pt", padding=True)

        preprocessed_images = preprocessed_images.to(self.device)

        caption_inputs = self.tokenizer(captions, padding=True, return_tensors="pt")

        # image_inputs = self.processor(text=captions, return_tensors="pt", padding=True)

        

        outputs = self.model(input_ids=caption_inputs['input_ids'].to(self.device), attention_mask=caption_inputs['attention_mask'].to(self.device), pixel_values=preprocessed_images)

        logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text

        logits_per_image, logits_per_text = logits_per_image / 100, logits_per_text / 100

        return logits_per_image, logits_per_text