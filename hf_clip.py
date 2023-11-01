import torch
from torch import nn
from transformers import ViTModel
import numpy as np
import torch.nn.functional as F
from clip_parent import ClipParent
import clip
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, CLIPTextModel, CLIPVisionModel, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from PIL import Image
import requests
from enum import Enum

from config import *



class HFClip(ClipParent):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)



        # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", )
        self.model = CLIPModel.from_pretrained(training_hyperparameters['hf_clip_model'], )
        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.tokenizer = AutoTokenizer.from_pretrained(training_hyperparameters['hf_clip_model'])

        # self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

        # self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

        # self.vision_model_with_projection = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

        # self.text_model_with_projection = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

        if selected_clip_model == ClipModels.FINETUNED_TEMP:
            # set temperature to zero
            self.model.logit_scale = torch.nn.Parameter(torch.zeros(1, requires_grad=False, device=self.device))

            self.model.logit_scale.requires_grad = False
        

        if not training_hyperparameters['start_new']:
            if selected_clip_model == ClipModels.FINETUNED_TEMP:
                self.load_state_dict(torch.load('checkpoints/my_clip_checkpoint_finetuned_temp.pt', map_location=self.device)['model_state_dict'])
            elif selected_clip_model == ClipModels.FINETUNED:
                self.load_state_dict(torch.load('checkpoints/my_clip_checkpoint_finetuned.pt', map_location=self.device)['model_state_dict'])
            elif selected_clip_model == ClipModels.DEFAULT:
                pass # no need to load, since hfclip already preloads the default model


    def encode_image(self, preprocessed_images):

        preprocessed_images = preprocessed_images.to(self.device)

        # make garbage captions
        captions = torch.ones(preprocessed_images.shape[0], 10, dtype=torch.long, device=self.device)

        # outputs = self.vision_model(pixel_values=preprocessed_images)

        # last_hidden_states = outputs.last_hidden_state
        # pooled_output = outputs.pooler_output # the last image encoder output just before linear projection. shape: ([batch_size, 512])

        # outputs = self.model(pixel_values=preprocessed_images, input_ids=captions)

        image_features = self.model.get_image_features(pixel_values=preprocessed_images)

        # outputs = self.vision_model_with_projection(pixel_values=preprocessed_images)

        # return pooled_output AFTER projection
        return image_features
    
    # def project_image(self, preprocessed_images):
    #     # print("projecting image")
    #     preprocessed_images = preprocessed_images.to(self.device)

    #     outputs = self.vision_model_with_projection(pixel_values=preprocessed_images)

    #     return outputs.image_embeds
    
    def tokenize_captions(self, captions):
        tokenized_captions = self.tokenizer(captions, padding=True, return_tensors="pt")

        tokenized_captions = tokenized_captions.to(self.device)

        return tokenized_captions
    
    def encode_text(self, captions):
        # # assuming raw captions input, so need to tokenize and stuff
        # tokenized_captions = self.tokenizer(captions, padding=True, return_tensors="pt")

        # tokenized_captions = tokenized_captions.to(self.device)

        # outputs = self.text_model(**tokenized_captions)

        # last_hidden_states = outputs.last_hidden_state
        # pooled_output = outputs.pooler_output # pooled (EOS token) states, text encoding just before CLIP's linear projection. shape: ([batch_size, 512])

        # return pooled_output

        # assuming raw captions input, so need to tokenize and stuff
        tokenized_captions = self.tokenizer(captions, padding=True, return_tensors="pt")

        tokenized_captions = tokenized_captions.to(self.device)

        outputs = self.text_model_with_projection(**tokenized_captions)

        return outputs.text_embeds
    
    
    
    # def project_text(self, captions):
    #     # assuming raw captions input, so need to tokenize and stuff
    #     tokenized_captions = self.tokenizer(captions, padding=True, return_tensors="pt")

    #     tokenized_captions = tokenized_captions.to(self.device)

    #     outputs = self.text_model_with_projection(**tokenized_captions)

    #     return outputs.text_embeds
    
    def forward(self, preprocessed_images, captions, output_loss=True, return_all=False):

        # inputs = self.processor(text=['captions', 'hello'], images=image, return_tensors="pt", padding=True)

        preprocessed_images = preprocessed_images.to(self.device)

        tokenized_captions = self.tokenize_captions(captions)




        outputs = self.model(input_ids=tokenized_captions['input_ids'].to(self.device), attention_mask=tokenized_captions['attention_mask'].to(self.device), pixel_values=preprocessed_images, return_loss=output_loss)



        

        if return_all:
            return outputs
        
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        

        if output_loss:
            loss = outputs.loss
            return logits_per_image, logits_per_text, loss
        else:
            return logits_per_image, logits_per_text

    def forward_1(self, preprocessed_images, captions, scale=False):

        # inputs = self.processor(text=['captions', 'hello'], images=image, return_tensors="pt", padding=True)

        preprocessed_images = preprocessed_images.to(self.device)

        encoded_images = self.encode_image(preprocessed_images)

        encoded_captions = self.encode_text(captions)

         # normalize features
        image_features = encoded_images / torch.norm(encoded_images, dim=1, keepdim=True)
        text_features = encoded_captions / torch.norm(encoded_captions, dim=1, keepdim=True)

        if scale:
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
        else:
            logits_per_image = image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
    

    def forward_old(self, preprocessed_images, captions, scale=True):

        # inputs = self.processor(text=['captions', 'hello'], images=image, return_tensors="pt", padding=True)

        preprocessed_images = preprocessed_images.to(self.device)

        caption_inputs = self.tokenizer(captions, padding=True, return_tensors="pt")

        # image_inputs = self.processor(text=captions, return_tensors="pt", padding=True)

        

        outputs = self.model(input_ids=caption_inputs['input_ids'].to(self.device), attention_mask=caption_inputs['attention_mask'].to(self.device), pixel_values=preprocessed_images)

        logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text

        logits_per_image, logits_per_text = logits_per_image / 100, logits_per_text / 100

        return logits_per_image, logits_per_text
    
