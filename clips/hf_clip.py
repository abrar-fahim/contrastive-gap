import torch
import numpy as np
from clips.clip_parent import ClipParent
from transformers import  CLIPModel, AutoTokenizer
from src.utils import get_checkpoint_path

from src.config import *
import os




class HFClip(ClipParent):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.tokenizer = AutoTokenizer.from_pretrained(training_hyperparameters['hf_clip_model'])
       

        self.temperature = 0.01 # this is default temp

        self.reset_weights_to_default() # loads clip model and sets the logit scale param 

        '''
        load CLIP from respective checkpoint regardless of training mode
        clip training toy and training loop will handle loading from scratch or loading from checkpoint
        '''

        checkpoint_path = get_checkpoint_path()

        # check if checkpoint path exists
        if os.path.exists(checkpoint_path):
            loaded_checkpoint = torch.load(checkpoint_path, map_location=self.device)

        
            # this only makes sense if we're loading from a checkpoint
            if not selected_clip_model == ClipModels.DEFAULT:
                self.load_state_dict(loaded_checkpoint['model_state_dict'])

        # if path doesnt exist, it means we're starting from pretrained model anyway

        print()
        print('--- HF CLIP MODEL ---')
        print()

        print('selected clip model ', selected_clip_model.name)
        # print('logit scale: ', self.model.logit_scale)
        print('temperature (T): ', self.temperature)
  

        print()

        # no need to load state dict for default, since it's the same as the pretrained model


    def reset_weights_to_random(self):
        self.model.init_weights()
        # self model is the only thing here thats trainable anyway

    def reset_weights_to_default(self):
        self.model = CLIPModel.from_pretrained(training_hyperparameters['hf_clip_model'], )

        # set model parameters requires_grad to True
        for param in self.model.parameters():
            param.requires_grad = True

        if selected_clip_model == ClipModels.FINETUNED_TEMP or selected_clip_model == ClipModels.WARM:

            self.temperature = training_hyperparameters['temperature']

            self.model.logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / self.temperature), requires_grad=False, device=self.device))

            self.model.logit_scale.requires_grad = False

        self.to(self.device)
        


    def encode_image(self, preprocessed_images):

        preprocessed_images = preprocessed_images.to(self.device)


        image_features = self.model.get_image_features(pixel_values=preprocessed_images)

        # return pooled_output AFTER projection
        return image_features


    def tokenize_captions(self, captions):
        tokenized_captions = self.tokenizer(captions, padding=True, return_tensors="pt", truncation=True, max_length=77)

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
        # tokenized_captions = self.tokenizer(captions, padding=True, return_tensors="pt")

        # tokenized_captions = tokenized_captions.to(self.device)

        tokenized_captions = self.tokenize_captions(captions)

        text_features = self.model.get_text_features(**tokenized_captions)

        return text_features
    
    
  
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
    
