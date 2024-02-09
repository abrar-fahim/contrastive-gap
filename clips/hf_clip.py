import torch
import numpy as np
from clips.clip_parent import ClipParent
from transformers import CLIPModel, AutoTokenizer, CLIPConfig, CLIPTextConfig, CLIPTextModelWithProjection
from src.utils import get_checkpoint_path
from torch.functional import F

from transformers.models.clip.modeling_clip import CLIPOutput

from src.config import *
import os
import copy


 

class HFClip(ClipParent):

    tokenizer = AutoTokenizer.from_pretrained(training_hyperparameters['hf_clip_model'])

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print('CLIP device ', self.device)


        self.tokenizer = AutoTokenizer.from_pretrained(training_hyperparameters['hf_clip_model'])
       

        self.temperature = 0.01 # this is default temp

        if training_hyperparameters['text_only']:
            self.set_weights('random')
        else:
            self.set_weights('default') # loads clip model and sets the logit scale param 

        '''
        load CLIP from respective checkpoint regardless of training mode
        clip training toy and training loop will handle loading from scratch or loading from checkpoint
        '''

        checkpoint_path = get_checkpoint_path()

        print('check point path for CLIP model ', checkpoint_path)

        # check if checkpoint path exists
        if os.path.exists(checkpoint_path):
            loaded_checkpoint = torch.load(checkpoint_path, map_location=self.device)

        
            # this only makes sense if we're loading from a checkpoint
            if not selected_clip_model == ClipModels.DEFAULT:
                self.load_state_dict(loaded_checkpoint['model_state_dict'])
                print('loaded clip model from checkpoint ', checkpoint_path)

            else:
                print('CLIP model not loaded from checkpoint')

        else:
            print('CLIP model not loaded from checkpoint')

        # if path doesnt exist, it means we're starting from pretrained model anyway

        self.loss = torch.nn.CrossEntropyLoss()

        print()
        print('--- HF CLIP MODEL ---')
        print()

        print('selected clip model ', selected_clip_model.name)

        if training_hyperparameters['text_only']:
            # calculate temperature from logit scale and assert that its the same as temp
            assert np.isclose(self.temperature, 1 / self.logit_scale.exp().item())
        else:
            # calculate temperature from logit scale and assert that its the same as temp
            assert np.isclose(self.temperature, 1 / self.model.logit_scale.exp().item())
        # print('logit scale: ', self.model.logit_scale)
        print('temperature (T): ', self.temperature)

        if training_hyperparameters['text_only']:
            print('CLIP running in text only mode')
            # self.logit_scale = self.model.logit_scale.clone() # because upto here, self.model does exist.

        print()

        # no need to load state dict for default, since it's the same as the pretrained model


    def set_weights(self, state='default'):
        if state == 'default':
            print('-- LOADING DEFAULT CLIP MODEL --')
            self.model = CLIPModel.from_pretrained(training_hyperparameters['hf_clip_model'], )
        elif state == 'random':

            if not training_hyperparameters['text_only']:
            
                print('-- LOADING CLIP MODEL WITH RANDOM WEIGHTS FROM SCRATCH --')
                '''
                These are from https://huggingface.co/docs/transformers/v4.36.1/en/model_doc/clip#transformers.CLIPConfig
                '''
                # Initializing a CLIPConfig with openai/clip-vit-base-patch32 style configuration
                configuration = CLIPConfig()

                # Initializing a CLIPModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
                self.model = CLIPModel(configuration)
                self.model.init_weights()
                # set model parameters requires_grad to True
                for param in self.model.parameters():
                    param.requires_grad = True
            else:
                print('CLIP running in text only mode')
                configuration = CLIPTextConfig()
                self.model = None

                if training_hyperparameters['same_encoder']:
                    print('CLIP running in same encoder mode')
                    self.text_model1 = CLIPTextModelWithProjection(configuration)
                    self.text_model2 = copy.deepcopy(self.text_model1)

                else:
                    print('CLIP running in different encoder mode')
                    self.text_model1 = CLIPTextModelWithProjection(configuration)    
                    self.text_model2 = CLIPTextModelWithProjection(configuration)

                self.text_model1.init_weights()
                self.text_model2.init_weights()
                for param in self.text_model1.parameters():
                    param.requires_grad = True
                for param in self.text_model2.parameters():
                    param.requires_grad = True

        if selected_clip_model == ClipModels.FINETUNED_TEMP or selected_clip_model == ClipModels.WARM:

            self.temperature = training_hyperparameters['temperature']
            self.intra_modality_temperature = training_hyperparameters['intra_modality_temperature']

            self.intra_modality_logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / self.intra_modality_temperature), requires_grad=False, device=self.device)) # not self.model since clip_model doesn't have intra_modality_logit_scale

            self.intra_modality_logit_scale.requires_grad = False
            
            if training_hyperparameters['text_only']:
                self.logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / self.temperature), requires_grad=False, device=self.device))
                self.logit_scale.requires_grad = False
            else:

                self.model.logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / self.temperature), requires_grad=False, device=self.device))

                self.model.logit_scale.requires_grad = False
        
        self.to(self.device)
        


    def encode_image(self, preprocessed_images):

        preprocessed_images = preprocessed_images.to(self.device)

        if training_hyperparameters['text_only']:
            # in this case, "preprocessed_images" is actually tokenized text
            image_features = self.text_model2(**preprocessed_images).text_embeds
        else:
            image_features = self.model.get_image_features(pixel_values=preprocessed_images)

        # return pooled_output AFTER projection
        return image_features
    
    @staticmethod
    def static_tokenize_captions(captions):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenized_captions = HFClip.tokenizer(captions, padding=True, return_tensors="pt", truncation=True, max_length=77)

        # tokenized_captions = tokenized_captions.to(device)

        return tokenized_captions


    def tokenize_captions(self, captions):
        tokenized_captions = self.tokenizer(captions, padding=True, return_tensors="pt", truncation=True, max_length=77)

        tokenized_captions = tokenized_captions.to(self.device)

        return tokenized_captions
    
    def encode_text(self, tokenized_captions):
        '''
        Returns pooled_output AFTER projection
        '''
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

        # tokenized_captions = self.tokenize_captions(captions)

        if training_hyperparameters['text_only']:
            text_features = self.text_model1(**tokenized_captions).text_embeds
        else:
            text_features = self.model.get_text_features(**tokenized_captions)

        return text_features
    
    
  
    def forward(self, preprocessed_images, captions, output_loss=True, return_all=False, output_intra_modality_loss=False):

        '''
        outputs = CLIPOutput(
            loss=loss,
            logits_per_image= logits_per_image,
            logits_per_text= logits_per_text,
            text_embeds= text_embeds,
            image_embeds= image_embeds,
        )
        '''

        # inputs = self.processor(text=['captions', 'hello'], images=image, return_tensors="pt", padding=True)

        # tokenized_captions = self.tokenize_captions(captions)

        if not training_hyperparameters['text_only']:


            tokenized_captions = captions.to(self.device)
            preprocessed_images = preprocessed_images.to(self.device)

            outputs = self.model(input_ids=tokenized_captions['input_ids'], attention_mask=tokenized_captions['attention_mask'], pixel_values=preprocessed_images, return_loss=output_loss)\
            
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # normalized features
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
        else:
            # in this case, "preprocessed_images" are actually captions
            tokenized_captions1 = preprocessed_images.to(self.device)
            tokenized_captions2 = captions.to(self.device)

            outputs1 = self.text_model1(**tokenized_captions1)
            outputs2 = self.text_model2(**tokenized_captions2)

            image_embeds = outputs1.text_embeds
            text_embeds = outputs2.text_embeds

            # normalized features
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            logits_per_image = image_embeds @ text_embeds.t() * self.logit_scale.exp() # logit_scale.exp() is 1 / temperature, so 100 for 0.01
            logits_per_text = text_embeds @ image_embeds.t() * self.logit_scale.exp()

        # this is exactly the same code (although I wrote it) as huggingface clip's loss as in https://github.dev/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py

        

        labels = torch.arange(logits_per_image.shape[0]).to(self.device)

        image_weight = training_hyperparameters['loss_weights']['image_to_text_weight']
        text_weight = training_hyperparameters['loss_weights']['text_to_image_weight']

        loss = 0

        if output_loss == True:

            if training_hyperparameters['intra_modality_loss']:
                # find cosine similarities between image embeddings themselves
                scaled_image_image_similarity = image_embeds @ image_embeds.t() * self.intra_modality_logit_scale.exp()

                # find cosine similarities between text embeddings themselves
                scaled_text_text_similarity = text_embeds @ text_embeds.t() * self.intra_modality_logit_scale.exp()

                intra_modality_loss = self.loss(scaled_image_image_similarity, labels) * image_weight + self.loss(scaled_text_text_similarity, labels) * text_weight

                # print('intra loss: ,', intra_modality_loss)
            if training_hyperparameters['rsa_loss']:
                
                text_text_cosine_similarities = text_embeds @ text_embeds.t()
                image_image_cosine_similarities = image_embeds @ image_embeds.t()

                # i can make intra-modality cosine sims and inter modality cosine sims as similiar as possible

                inter_image_rsa = F.cosine_similarity(logits_per_image.reshape(1, -1), image_image_cosine_similarities.reshape(1, -1))
                inter_text_rsa = F.cosine_similarity(logits_per_text.reshape(1, -1), text_text_cosine_similarities.reshape(1, -1))

                rsa_loss = -(inter_image_rsa + inter_text_rsa) / 2

            if training_hyperparameters['pearson_loss']:

                '''
                ACTUAL PEARSON CORRELATION IN RSA LOSS
                '''

                text_text_cosine_similarities = text_embeds @ text_embeds.t()
                image_image_cosine_similarities = image_embeds @ image_embeds.t()

                image_text_RSM = logits_per_image[torch.tril(torch.ones(logits_per_image.shape[0], logits_per_image.shape[1]), diagonal=-1).bool()] # shape: (k)

                text_RSM = text_text_cosine_similarities[torch.tril(torch.ones(text_text_cosine_similarities.shape[0], text_text_cosine_similarities.shape[1]), diagonal=-1).bool()]    # shape: (k)

                image_RSM = image_image_cosine_similarities[torch.tril(torch.ones(image_image_cosine_similarities.shape[0], image_image_cosine_similarities.shape[1]), diagonal=-1).bool()]   # shape: (k)
                
                # stack image_text RSM and text RSM 
                # and then calculate pearson correlation
                # then calculate loss
                stacked_RSM_text = torch.stack([image_text_RSM, text_RSM], dim=0) # shape: (2, k)
                stacked_RSM_image = torch.stack([image_text_RSM, image_RSM], dim=0) # shape: (2, k)

                text_corr = torch.corrcoef(stacked_RSM_text) # shape: (2, 2)
                image_corr = torch.corrcoef(stacked_RSM_image) # shape: (2, 2)

                pearson_rsa_loss = -(text_corr[0, 1] + image_corr[0, 1]) / 2


            inter_modality_loss = self.loss(logits_per_image, labels) * image_weight + self.loss(logits_per_text, labels) * text_weight 

            if training_hyperparameters['intra_modality_loss']:
                loss = (intra_modality_loss + inter_modality_loss) / 2
            elif training_hyperparameters['rsa_loss']:
                loss = inter_modality_loss + rsa_loss
            elif training_hyperparameters['pearson_loss']:
                loss = inter_modality_loss + pearson_rsa_loss
                
            else:
                loss = inter_modality_loss

            if output_intra_modality_loss:
                loss = {
                    'inter_modality': inter_modality_loss.item(),
                    'rsa': rsa_loss.item() if training_hyperparameters['rsa_loss'] else -100,
                    'intra_modality': intra_modality_loss.item() if training_hyperparameters['intra_modality_loss'] else -100,
                    'pearson_rsa': pearson_rsa_loss.item() if training_hyperparameters['pearson_loss'] else -100,
                    'total': loss.item(),
                }

        outputs = CLIPOutput(
            loss=loss,
            logits_per_image= logits_per_image,
            logits_per_text= logits_per_text,
            text_embeds= text_embeds,
            image_embeds= image_embeds,
        )


        if return_all:
            return outputs
        
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
        if output_loss:
            
            return logits_per_image, logits_per_text, loss
        else:
            return logits_per_image, logits_per_text

