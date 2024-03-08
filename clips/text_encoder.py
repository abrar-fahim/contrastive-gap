import torch
from transformers import CLIPTextConfig, CLIPTextModelWithProjection

from transformers.models.clip.modeling_clip import CLIPOutput

from src.config import *

from clips.encoder import Encoder


 
class TextEncoder(Encoder):


    # init
    def __init__(self, tokenizer, CLIPTextConfig, from_pretrained=False, name='Untitled Text Encoder'):
        '''
        Set CLIPTextConfig with appropriate vocab size if using diff tokenizers
        '''
        super().__init__()

        self.tokenizer = tokenizer

        self.name = name

        self.device = torch.device(training_hyperparameters['cuda_device'] if torch.cuda.is_available() else "cpu")

        if from_pretrained:
            print()
            print(f" --- Initializing {name} from pretrained model ---")
            print()
            self.text_model = CLIPTextModelWithProjection.from_pretrained(training_hyperparameters['hf_clip_model']).to(self.device)

        else:
            print()
            print(f" --- Initializing {name} from scratch --- ")
            print()
            
            self.text_model = CLIPTextModelWithProjection(CLIPTextConfig).to(self.device)
            self.text_model.init_weights()
            
            

        for param in self.text_model.parameters():
            param.requires_grad = True


    def forward(self, captions):

        tokenized_captions = self.tokenize_captions(captions)

        text_features = self.text_model(**tokenized_captions).text_embeds
        del tokenized_captions

        return text_features

    def tokenize_captions(self, captions):
        return self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt").to(self.device)
    

    def reset_weights_to_init(self):

        print()
        print(f" --- Initializing {self.name} from scratch --- ")
        print()
        self.text_model.init_weights()




        