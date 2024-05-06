import torch
from transformers import AutoTokenizer, CLIPTextConfig,GPT2Tokenizer, CLIPVisionConfig
from src.config import *
import clip
from clips.text_encoder import TextEncoder
from clips.image_encoder import ImageEncoder
import copy
from clips.hf_clip import HFClip

from clips.projection_layer import ProjectionLayer




class ClipAssembler():

    def __init__(self) -> None:
       
        '''
        Setting tokenizers
        '''

        assert torch.initial_seed() == wandb.config['seed'], "Seed not set properly"

        self.validate_config()
        self.device = config_cuda_device if torch.cuda.is_available() else "cpu"

        self.clip_tokenizer = AutoTokenizer.from_pretrained(wandb.config['hf_clip_model'])

        if wandb.config['second_caption_offset']:
            self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
            self.gpt_tokenizer.pad_token = self.clip_tokenizer.pad_token
            self.gpt_tokenizer.eos_token = self.clip_tokenizer.eos_token


        '''
        Setting image preprocessors
        '''

        _, self.image_preprocessor = clip.load(wandb.config['openai_clip_model'], device=self.device)


        '''
        Setting text configs
        '''

        if wandb.config['second_caption_offset']:
            self.clip_text_config = CLIPTextConfig(vocab_size=self.gpt_tokenizer.vocab_size)
        else:
            self.clip_text_config = CLIPTextConfig()


        '''
        Setting vision configs
        '''

        self.clip_vision_config = CLIPVisionConfig()

        if wandb.config['clip_projection_dim'] != 512:

            print('setting projection dim to ', wandb.config['clip_projection_dim'])
            self.clip_vision_config.projection_dim = wandb.config['clip_projection_dim']
            self.clip_text_config.projection_dim = wandb.config['clip_projection_dim']

        if wandb.config['vision_model'] == 'VIT':
            # using base patch 16 now
            # self.clip_vision_config.patch_size = 16
            pass # sticking to 32


        if wandb.config['shared_transformer_layers']:
            # set text config to be same as vision config

            self.clip_text_config.hidden_size = self.clip_vision_config.hidden_size
            self.clip_text_config.layer_norm_eps = self.clip_vision_config.layer_norm_eps
            self.clip_text_config.num_attention_heads = self.clip_vision_config.num_attention_heads
            self.clip_text_config.attention_dropout = self.clip_vision_config.attention_dropout
            self.clip_text_config.initializer_factor = self.clip_vision_config.initializer_factor
           
        '''
        Setting Encoders
        '''

        if wandb.config['encoder1_modality'] == 'text':
            print()
            print("--- ENCODER 1 = TEXT--- ")
            print()
            self.encoder1 = TextEncoder(self.clip_tokenizer, self.clip_text_config, from_pretrained=(not wandb.config['train_from_scratch']), name='CLIP Text Encoder')

        elif wandb.config['encoder1_modality'] == 'image':
            print()
            print("--- ENCODER 1 = IMAGE --- ")
            print()
            self.encoder1 = ImageEncoder(self.image_preprocessor, self.clip_vision_config,from_pretrained=(not wandb.config['train_from_scratch']), name='CLIP Image Encoder')

        else:
            raise ValueError("Encoder 1 modality not set properly")


        if wandb.config['one_encoder']:
            print()
            print("---  ENCODER 2 =  ENCODER 1. ONE ENCODER ONLY --- ")
            print()
            # self.encoder2 = None
            self.encoder2 = self.encoder1

        elif wandb.config['same_encoder']:
            print()
            print("--- Initializing text encoders to be SAME AT INIT --- ")
            print()
            self.encoder2 = copy.deepcopy(self.encoder1)

        elif wandb.config['encoder2_modality'] == 'image':
            print()
            print("--- ENCODER 2 = IMAGE --- ")
            print()
            self.encoder2 = ImageEncoder(self.image_preprocessor, self.clip_vision_config, from_pretrained=(not wandb.config['train_from_scratch']), name='Image Encoder 2')
        elif wandb.config['encoder2_modality'] == 'text':
            print()
            print("--- ENCODER 2 = TEXT --- ")
            print()
            self.encoder2 = TextEncoder(self.clip_tokenizer, self.clip_text_config, from_pretrained=(not wandb.config['train_from_scratch']), name=f"Text Encoder with {'GPT2' if wandb.config['second_caption_offset'] else 'CLIP'} tokenizer")

        else:
            raise ValueError("Encoder 2 modality not set properly")

        '''
        Check 
        '''

        if wandb.config['same_encoder']:
            assert str(self.encoder1.state_dict()) == str(self.
            encoder2.state_dict()), "Encoder 1 and Encoder 2 are not same at init"

        

        
        if wandb.config['second_caption_offset']:
            print()
            print("--- Setting Second Text Encoder to have GPT tokenizer --- ")
            self.encoder2.tokenizer = self.gpt_tokenizer

        if wandb.config['common_projection_layer']:
            print()
            print("--- Setting common projection layer --- ")
            self.projection_layer = ProjectionLayer()
        else:
            self.projection_layer = None


        if wandb.config['shared_transformer_layers']:
            print()
            print("--- Setting shared transformer layers --- ")
            
            if type(self.encoder1) == TextEncoder:

                if type(self.encoder2) == ImageEncoder:

                    self.encoder1.text_model.text_model.encoder = self.encoder2.image_model.vision_model.encoder

                    self.encoder1.text_model.text_projection = self.encoder2.image_model.visual_projection

                elif type(self.encoder2) == TextEncoder:

                    self.encoder1.text_model.text_model.encoder = self.encoder2.text_model.text_model.encoder

                    self.encoder1.text_model.text_projection = self.encoder2.text_model.text_projection
            elif type(self.encoder1) == ImageEncoder:

                if type(self.encoder2) == ImageEncoder:


                    print('hereeee')

                    self.encoder1.image_model.vision_model.encoder = self.encoder2.image_model.vision_model.encoder

                    self.encoder1.image_model.visual_projection = self.encoder2.image_model.visual_projection

                    self.encoder1.image_model.vision_model.post_layernorm = self.encoder2.image_model.vision_model.post_layernorm

                    self.encoder1.image_model.vision_model.pre_layrnorm = self.encoder2.image_model.vision_model.pre_layrnorm

                    # self.encoder1.image_model.vision_model.embeddings = self.encoder2.image_model.vision_model.embeddings

                    # check if encoder weights are same
                    assert str(self.encoder1.image_model.vision_model.encoder.state_dict()) == str(self.encoder2.image_model.vision_model.encoder.state_dict()), "Encoder 1 and Encoder 2 are not same at init"


                    assert str(self.encoder1.image_model.visual_projection.state_dict()) == str(self.encoder2.image_model.visual_projection.state_dict()), "Projection 1 and Projection 2 are not same at init"

                elif type(self.encoder2) == TextEncoder:

                    self.encoder1.image_model.vision_model.encoder = self.encoder2.text_model.text_model.encoder

                    self.encoder1.image_model.visual_projection = self.encoder2.text_model.text_projection

            

            # config stuff used in encoder
            # hidden_size, layer_norm_eps, 



            



        # if wandb.config['W_layer_gap'] >= 0:
        #     self.W: torch.FloatTensor = None


        '''
        Setting CLIP model
        '''

        self.clip_model = HFClip(self.encoder1, self.encoder2, self.projection_layer)


    def validate_config(self):

        assert wandb.config['encoder1_modality'] in ['text', 'image'], "encoder1_modality not set properly"
        assert wandb.config['encoder2_modality'] in ['text', 'image'], "encoder2_modality not set properly"

        # make sure second_caption_offset, same_inputs, same_encoder are only set when text_only is True
        if wandb.config['second_caption_offset'] or wandb.config['same_inputs'] or wandb.config['same_encoder']:
            assert wandb.config['encoder1_modality'] == wandb.config['encoder2_modality'], "second_caption_offset, same_inputs, same_encoder can only be set when encoders have same modality"

        if wandb.config['one_encoder']:
            # make sure same_encoder, second_caption_offset, same_inputs are not set when one_encoder is set
            assert not wandb.config['same_encoder'] and not wandb.config['second_caption_offset'] and not wandb.config['same_inputs'], "one_encoder cannot be set with same_encoder, second_caption_offset, same_inputs"

            

        return
