import torch
from torch import nn



from transformers import ViTModel
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer

import numpy as np
import torch.nn.functional as F

from clip_parent import ClipParent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageProjector(nn.Module):
    
    def __init__(self):

        super().__init__()

        # self.layer1 = nn.Linear(1024, 1024, device=device)
        # self.layer2 = nn.Linear(1024, 512, device=device)

        self.layer1 = nn.Linear(1024, 512, device=device)  


        
        
        # initialize weights using xavier uniform initialization
        nn.init.xavier_uniform_(self.layer1.weight)
        # nn.init.xavier_uniform_(self.layer2.weight)

        
        

    def forward(self, image):
        x = self.layer1(image)
        # do relu
        # x = F.relu(x)
        # x = self.layer2(x)
        return x
        

class TextProjector(nn.Module):
        
    def __init__(self):
        super().__init__()

        # self.layer1 = nn.Linear(768, 768, device=device)
        # self.layer2 = nn.Linear(768, 512, device=device)
        self.layer1 = nn.Linear(768, 512, device=device)
        
        nn.init.xavier_uniform_(self.layer1.weight)
        # nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, text):
        x  = self.layer1(text)
        # x = F.relu(x)
        # x = self.layer2(x)
        return x

class MyClip(ClipParent):

    # init
    def __init__(self):

        super().__init__()

        # self.image_feature_extractor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
        self.image_encoder = ViTModel.from_pretrained('google/vit-large-patch16-224').to(device)

        gpt_configuration = GPT2Config(summary_type="cls_index", device=device)

        self.text_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", device=device)
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token # set to zero LATER
        self.text_encoder = GPT2Model.from_pretrained("gpt2", config=gpt_configuration).to(device)

        '''
        setup learnable parameters
        '''

        self.image_projector = ImageProjector()
        self.text_projector = TextProjector()

        '''
        freeze parameters for image encoder and text encoder
        '''
        # for param in self.image_encoder.parameters():
        #     param.requires_grad = False
        
        # for param in self.text_encoder.parameters():
        #     param.requires_grad = False

        '''
        Maybe unfreeze the pooling layer of the image encoder
        '''
        # for param in self.image_encoder.pooler.parameters():
        #     param.requires_grad = True

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).to(device)
        



    def encode_image(self, image):
        # inputs = self.image_feature_extractor(images=image, return_tensors="pt")
        # outputs = self.image_encoder(**inputs)
        # print('image shape, ' , image.shape)
        image = image.to(device)
        outputs = self.image_encoder(image)
        pooler_output = outputs.pooler_output # shape: ([1, 1024])
        return pooler_output


    def encode_text(self, text):
        # add sot and eot tokens to text, look into whether tokenizer does this automatically or not LATER

        # sot_token = _tokenizer.encoder["<|startoftext|>"]
        # eot_token = _tokenizer.encoder["<|endoftext|>"]
        # all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

        # print('text ', text)
        inputs = self.text_tokenizer(text, return_tensors="pt", padding='max_length', max_length=77, add_special_tokens=True) # has keys: ['input_ids', 'attention_mask']

        # print('inputs ', inputs)
        '''
        - I set padding token to eos_token, so eos_token already exists at the last position. 
        - For now, set attention_mask for eot token to 1, since I'm taking outputs of last token for text representation
        - argmax cuz eot token 
        '''
        inputs['attention_mask'][:, inputs['input_ids'].argmax(dim=-1)] = 1
        # print('inputs ', inputs)
        inputs.to(device)
        # inputs has keys: ['input_ids', 'attention_mask']
        # inputs['input_ids'] has shape: [64, 77]
        # print('inputs shape ', inputs['input_ids'].shape)
        outputs = self.text_encoder(**inputs)
        last_hidden_states = outputs.last_hidden_state # shape: ([64, 77, 768])
        eos_representation = last_hidden_states[torch.arange(inputs['input_ids'].shape[0]), inputs['input_ids'].argmax(dim=-1)] #  shape: ([batch_size, 768])

        # print('eos_representation ', eos_representation.shape)
        return eos_representation


    def project_text(self, text):
        return self.text_projector(text)

    def project_image(self, image):
        return self.image_projector(image)
    
    def forward(self, image, text, scale=True):


        image_features = self.encode_image(image)
        # image_features = self.encode_text(image)
        text_features = self.encode_text(text)
        
        image_features = self.project_image(image_features)
        # image_features = self.project_text(image_features)
        text_features = self.project_text(text_features)

        # print('image_features ', image_features)

        # print('text_features ', text_features)

        

        # normalize features
        image_features = image_features / torch.norm(image_features, dim=1, keepdim=True)
        text_features = text_features / torch.norm(text_features, dim=1, keepdim=True)

        # print('image_features ', image_features)

        # print('text_features ', text_features)

        

        # from clip
        if scale:
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
        else:
            logits_per_image = image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text



class MyClipLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits_per_image, logits_per_text):
        # logits_per_image shape = [global_batch_size, global_batch_size]
        # logits_per_text shape = [global_batch_size, global_batch_size]
        # labels shape = [global_batch_size]
        global_batch_size = logits_per_image.shape[0]
        assert logits_per_image.shape == (global_batch_size, global_batch_size)
        assert logits_per_text.shape == (global_batch_size, global_batch_size)
        labels = torch.arange(global_batch_size).to(logits_per_image.device)
        image_loss = self.loss(logits_per_image, labels)
        text_loss = self.loss(logits_per_text, labels)
        loss = (image_loss + text_loss) / 2
        return loss