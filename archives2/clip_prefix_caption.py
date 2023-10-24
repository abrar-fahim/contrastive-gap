'''
Image-captioning with CLIP
Take image embedding of input image, use that embedding as a prefix to language model to generate caption

from https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing#scrollTo=V7xocT3TUgey
'''


from PIL import Image
import requests
from archives2.CLIPWrapper import CLIPWrapper

import skimage.io as io

import clip


# GPT imports

import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange

from PIL import Image



similarity_measure = 'cosine_similarity'

cat_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
cat_url = "https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/CONCEPTUAL_04.jpg"
cat_url = "https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/CONCEPTUAL_01.jpg"
cat_url = "https://upload.wikimedia.org/wikipedia/commons/5/57/Mobfooty.jpg"
cat_url = "https://datasets-server.huggingface.co/assets/nlphuji/mscoco_2014_5k_test_image_text_retrieval/--/TEST/test/0/image/image.jpg"
cat_url = "https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/COCO_val2014_000000562207.jpg"
cat_url = "https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/COCO_val2014_000000165547.jpg"
cat_url = "https://images.fineartamerica.com/images/artworkimages/mediumlarge/2/zebra-in-living-room-smelling-rug-side-matthias-clamer.jpg"
cat_url = "https://static.uwalls.com/products/9000/9004/u07312pik1m_1200.webp"
cat_url = "https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/COCO_val2014_000000354533.jpg"
cat_url = "https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/CONCEPTUAL_05.jpg"
cat_url = "https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/COCO_val2014_000000562207.jpg"
cat_url = "https://i.natgeofe.com/n/535f3cba-f8bb-4df2-b0c5-aaca16e9ff31/giza-plateau-pyramids_16x9.jpg"
cat_url = "https://cdn.mos.cms.futurecdn.net/YMa7Wx2FyjQFUjEeqa72Rm.jpg"
cat_url = "https://images.immediate.co.uk/production/volatile/sites/7/2023/01/Conspiracy-WL-Pyramids-b5d0e0e.jpg"
cat_url = "https://images.unsplash.com/photo-1653038546613-6eece21ed1c5?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1yZWxhdGVkfDR8fHxlbnwwfHx8fHw%3D&w=1000&q=80"
cat_url = "https://i.redd.it/y85hj65gk5e31.jpg"
cat_url = "https://c8.alamy.com/comp/2CC54N3/modern-interior-at-home-wooden-chair-plant-on-pot-and-light-curtains-on-windows-2CC54N3.jpg"


cat_url = "https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/COCO_val2014_000000165547.jpg" # OG dining table image
cat_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSXW_doyDZ1kPWK3KR3-AB50oJxT2nrvC0BuZMwjp3E&s"

cat_url = "https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/COCO_val2014_000000579664.jpg"  # BANANAS
cat_url = "https://github.com/rmokady/CLIP_prefix_caption/raw/main/Images/CONCEPTUAL_04.jpg"  # BANANAS


cat_url = "https://datasets-server.huggingface.co/assets/nlphuji/mscoco_2014_5k_test_image_text_retrieval/--/TEST/test/2/image/image.jpg"  # BANANAS
cat_url = "https://datasets-server.huggingface.co/assets/nlphuji/mscoco_2014_5k_test_image_text_retrieval/--/TEST/test/3/image/image.jpg"  # BANANAS
cat_url = "https://www.livingspaces.com/globalassets/images/blog/2018/09/0910_console_vs_side_table_square.jpg"  # BANANAS
cat_url = "https://m.media-amazon.com/images/I/61U7gUPxlvL.jpg"  # BANANAS
cat_url = "https://ronixtools.com/en/blog/wp-content/uploads/2021/03/Learn-how-to-make-a-simple-wooden-table-at-home1.jpg"  # BANANAS
cat_url = "https://i.ytimg.com/vi/V1-JeJawN80/maxresdefault.jpg"  # BANANAS
cat_url = "https://media.istockphoto.com/id/996168058/vector/black-led-tv-television-screen-blank-on-white-wall-background.jpg?s=612x612&w=0&k=20&c=MQI3naQQ7dteEbt8xmiE97OEPy2UA30OAh0pwtyJt9I="  # BANANAS

cat_url = ''



# load image from url
image = Image.open(requests.get(cat_url, stream=True).raw)

# image = io.imread(cat_url)
# pil_image = Image.fromarray(image)

# images = [pil_image] # since processor expects a list of images
images = [image] # since processor expects a list of images



placeholder_captions = 'blah'

clip_wrapper = CLIPWrapper(placeholder_captions, images, similarity_measure)



# get image embeddings

image_embeddings = clip_wrapper.get_image_embeddings() # shape: (n, 512)

# feed image embeddings into language model to generate caption



'''
GPT 2 stuff start
'''


# type definitions
N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]


D = torch.device
CPU = torch.device('cpu')

def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')

CUDA = get_device

# model_path = './pretrained_models/model_weights_conceptual.pt'
model_path = './pretrained_models/model_weights_coco.pt'
#@title Model

class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    #@functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        #print(embedding_text.size()) #torch.Size([5, 67, 768])
        #print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

#@title Caption prediction

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in trange(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


#@title GPU/CPU


is_gpu = False #@param {type:"boolean"}


device = CUDA(0) if is_gpu else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


#@title Load model weights


prefix_length = 10

model = ClipCaptionModel(prefix_length)

# print state dict from file
print(len(torch.load(model_path, map_location=CPU).keys()))

altered_state_dict = torch.load(model_path, map_location=CPU)

# remove attn bias and masked bias from state dict
for i in range(12):
    del altered_state_dict['gpt.transformer.h.' + str(i) + '.attn.bias']
    del altered_state_dict['gpt.transformer.h.' + str(i) + '.attn.masked_bias']

model.load_state_dict(altered_state_dict)



model = model.eval()
device = CUDA(0) if is_gpu else "cpu"

# use openai's github clip model, CHANGE LATER
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=True)


model = model.to(device)



#@title Inference


use_beam_search = False #@param {type:"boolean"}

# image = image
#pil_img = Image(filename=UPLOADED_FILE)

# display image
image.show()

image = io.imread(cat_url)
pil_image = Image.fromarray(image)

# display(pil_image)

image = preprocess(pil_image).unsqueeze(0).to(device)







with torch.no_grad():
    # if type(model) is ClipCaptionE2E:
    #     prefix_embed = model.forward_image(image)
    # else:

    # CHANGE THIS LATER
    prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)

    text = clip.tokenize('A TV in a room.').to(device)
    
    # my_prefix = image_embeddings[0].to(device, dtype=torch.float32)

    # my_prefix = my_prefix.reshape(1, -1)
 
    text_prefix = clip_model.encode_text(text).to(device, dtype=torch.float32)

    print('text_prefix shape: ', text_prefix.shape)

    print('prefix shape: ', prefix.shape)



    prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
    # prefix_embed = model.clip_project(text_prefix).reshape(1, prefix_length, -1)
    # prefix_embed = model.clip_project(my_prefix).reshape(1, prefix_length, -1)
if use_beam_search:
    generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
else:
    generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)


# print('transformers version ', transformers.__version__)
print('torch version: ', torch.__version__)
print('\n')
print("Generated caption:")
print(generated_text_prefix)