import torch
import numpy as np
from clips.clip_parent import ClipParent
from transformers import CLIPModel, AutoTokenizer, CLIPConfig, CLIPTextConfig, CLIPTextModelWithProjection,GPT2Tokenizer, CLIPVisionConfig
from src.utils import get_checkpoint_path
from torch.functional import F

from transformers.models.clip.modeling_clip import CLIPOutput

from src.config import *
import clip

from encoder import Encoder
from text_encoder import TextEncoder
from image_encoder import ImageEncoder
import copy




class ClipAssembler():

    def __init__(self) -> None:
       
        '''
        Setting tokenizers
        '''

        self.validate_config()
        self.device = training_hyperparameters['cuda_device'] if torch.cuda.is_available() else "cpu"

        self.clip_tokenizer = AutoTokenizer.from_pretrained(training_hyperparameters['hf_clip_model'])

        if training_hyperparameters['second_caption_offset']:
            self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
            self.gpt_tokenizer.pad_token = self.tokenizer.pad_token
            self.gpt_tokenizer.eos_token = self.tokenizer.eos_token

        '''
        Setting image preprocessors
        '''

        _, self.image_preprocessor = clip.load(training_hyperparameters['openai_clip_model'], device=self.device)


        '''
        Setting text configs
        '''

        if training_hyperparameters['second_caption_offset']:
            self.clip_text_config = CLIPTextConfig(vocab_size=self.gpt_tokenizer.vocab_size)
        else:
            self.clip_text_config = CLIPTextConfig()


        '''
        Setting vision configs
        '''

        self.clip_vision_config = CLIPVisionConfig()

        '''
        Setting Encoders
        '''

        if training_hyperparameters['text_only']:
            print()
            print("--- TEXT ONLY MODE --- ")
            print()
            self.encoder1 = TextEncoder(self.clip_tokenizer, self.clip_text_config, from_pretrained=training_hyperparameters['train_from_scratch'], name='CLIP Text Encoder')
        else:
            print()
            print("--- IMAGE + TEXT MODE --- ")
            print()
            self.encoder1 = ImageEncoder(self.image_preprocessor, self.clip_vision_config,from_pretrained=training_hyperparameters['train_from_scratch'], name='CLIP Image Encoder')

        if training_hyperparameters['same_encoder']:
            print()
            print("--- Initializing text encoders to be SAME AT INIT --- ")
            print()
            self.encoder2 = copy.deepcopy(self.encoder1)
        else:
            self.encoder2 = TextEncoder(self.clip_tokenizer, self.clip_text_config, from_pretrained=training_hyperparameters['train_from_scratch'], name='Text Encoder with GPT2 tokenizer')

        
        if training_hyperparameters['second_caption_offset']:
            print()
            print("--- Setting Second Text Encoder to have GPT tokenizer --- ")
            self.encoder2.tokenizer = self.gpt_tokenizer


        '''
        Setting CLIP model
        '''

        self.clip_model = HFClip(self.encoder1.model, self.encoder2.model)




    def validate_config(self):

        if training_hyperparameters['second_caption_offset'] or training_hyperparameters['same_captions'] or training_hyperparameters['same_encoder']:
            assert training_hyperparameters['text_only'], "second_caption_offset, same_captions, same_encoder only work when text_only is True"

        return
