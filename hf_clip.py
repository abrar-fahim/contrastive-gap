import torch
from torch import nn
from transformers import ViTModel
import numpy as np
import torch.nn.functional as F
from clip_parent import ClipParent
import clip
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests


class HFClip(ClipParent):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)


        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
        self.model = self.model.to(self.device)

    def encode_image(self, image):
        image = image.to(self.device)
        return self.model.encode_image(image)
    
    def encode_text(self, text):
        # assuming raw captions input, so need to tokenize and stuff
        tokenized_captions = self.tokenize_text(text)
        return self.model.encode_text(tokenized_captions)
    
    def forward(self, preprocessed_images, captions, scale=True):

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        # inputs = self.processor(text=['captions', 'hello'], images=image, return_tensors="pt", padding=True)

        preprocessed_images = preprocessed_images.to(self.device)


        inputs = self.processor(text=captions, images=image, return_tensors="pt", padding=True)

        

        outputs = self.model(input_ids=inputs['input_ids'].to(self.device), attention_mask=inputs['attention_mask'].to(self.device), pixel_values=preprocessed_images)

        logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text

        logits_per_image, logits_per_text = logits_per_image / 100, logits_per_text / 100

        return logits_per_image, logits_per_text