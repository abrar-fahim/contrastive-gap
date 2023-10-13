import torch
from torch import nn


from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor
from PIL import Image
import requests

from transformers import AutoImageProcessor, ViTModel
import torch
from transformers import AutoTokenizer, GPT2Model, GPT2Config
from datasets import load_dataset

class ImageProjector(nn.Module):
    
        def __init__(self):
            super().__init__()
    
            self.image_projector = nn.Linear(1024, 512)
            nn.init.xavier_uniform_(self.image_projector.weight)
    
        def forward(self, image):
            return self.image_projector(image)
        

class TextProjector(nn.Module):
        
            def __init__(self):
                super().__init__()
        
                self.text_projector = nn.Linear(768, 512)
                nn.init.xavier_uniform_(self.text_projector.weight)
        
            def forward(self, text):
                return self.text_projector(text)

class MyClip(nn.Module):

    # init
    def __init__(self):

        super().__init__()

        self.image_feature_extractor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
        self.image_encoder = ViTModel.from_pretrained('google/vit-large-patch16-224')

        gpt_configuration = GPT2Config(summary_type="cls_index")

        self.text_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.text_encoder = GPT2Model.from_pretrained("gpt2")

        '''
        setup learnable parameters
        '''

        self.image_projector = ImageProjector()
        self.text_projector = TextProjector()

        # initialize weights
        nn.init.xavier_uniform_(self.image_projector.weight)
        nn.init.xavier_uniform_(self.text_projector.weight)

        '''
        freeze parameters for image encoder and text encoder
        '''
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        '''
        Maybe unfreeze the pooling layer of the image encoder
        '''
        for param in self.image_encoder.pooler.parameters():
            param.requires_grad = True
        



    def encode_image(self, image):
        inputs = self.image_feature_extractor(images=image, return_tensors="pt")
        outputs = self.image_encoder(**inputs)
        pooler_output = outputs.pooler_output # shape: ([1, 1024])
        return pooler_output


    def encode_text(self, text):
        inputs = self.text_tokenizer(text, return_tensors="pt")
        outputs = self.text_encoder(**inputs)
        last_hidden_states = outputs.last_hidden_state
        eos_representation = last_hidden_states[:, -1, :] #  shape: ([1, 768])
        return eos_representation


    def project_text(self, text):
        return self.text_projector(text)

    def project_image(self, image):
        return self.image_projector(image)
    
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = self.project_image(image_features)
        text_features = self.project_text(text_features)

        # normalize features
        image_features = image_features / torch.norm(image)
        image_features = image_features / torch.norm(text)

        # from clip
        logits_per_image = image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text



