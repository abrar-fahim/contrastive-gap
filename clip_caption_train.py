import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as dset
import clip
from hf_clip import HFClip
import numpy as np
from config import ClipModels, selected_clip_model
from training_utils import collate_fn
from config import *


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix

    
    def make_all_data(self):
        all_data = {
            'clip_embedding': None, # tensor of shape: ([num_images, 512])
            'captions': [], # list of dictionaries, each dictionary has keys: 'image_id', 'caption', 'clip_embedding'
            'my_captions': [], # list of captions, where each caption is the first caption of the corresponding image
            'caption2embedding': [], # maps caption index to index of corresponding image embedding in self.prefixes
        }

        all_embeddings = []
        all_captions = []
        caption2embedding = []

        model, preprocess = clip.load(training_hyperparameters['openai_clip_model'], device=self.device)

        model = HFClip().to(self.device)

        train_dataset = dset.CocoCaptions(root = './datasets/mscoco/val2014',
                        annFile = 'datasets/mscoco/annotations/captions_val2014.json',
                        # transform=[transforms.PILToTensor()])
                        transform=preprocess,)
        
        subset_indices = torch.randint(0, len(train_dataset) , (10000,)) # always defined and exists, but only used when small training loader is used, and we're not loading from checkpoint at start

        train_data_subset = Subset(train_dataset, subset_indices)
        
        # create dataloader
        train_dataloader = DataLoader(train_data_subset, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_fn)

        print('making all_data')
        
        
        # add image encodings to clip_embedding of all_data
        for i, (images, captions) in enumerate(tqdm(train_dataloader)):
            # images, captions = train_dataloader[i]
            # caption is a list (len = batch_size) of strings 
            # images = images.to(self.device)
            with torch.no_grad():
                outputs = model(images, captions, return_all=True)
                image_encodings = outputs.image_embeds.cpu()
                # text_encodings = outputs.text_embeds.cpu()
            all_embeddings.append(image_encodings)


            all_captions.extend(captions) # so that captions is just one list of captions, instead of list of lists.

            start = i * images.shape[0]
            end = start + images.shape[0]
            caption2embedding.extend(np.arange(start, end)) # so that caption2embedding is just one list of indices, instead of list of lists.


        # add clip_embedding to all_data
        all_data['clip_embedding'] = torch.cat(all_embeddings, dim=0)

        # add captions to all_data
        all_data['my_captions'] = all_captions

        all_data['caption2embedding'] = caption2embedding


    
        print("%0d embeddings saved " % len(all_embeddings))

        return all_data



    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix

        if os.path.isfile('caption_dataset/all_data.pkl'):
            with open('caption_dataset/all_data.pkl', 'rb') as f:
                all_data = pickle.load(f)
        else:
            all_data = self.make_all_data()
            with open('caption_dataset/all_data.pkl', 'wb') as f:
                pickle.dump(all_data, f)

        # all_data = self.make_all_data()
        # with open(data_path, 'rb') as f:
        #     all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        # clip_embedding is the image embedding from CLIP
        sys.stdout.flush()

        print('building prefixes')


        self.prefixes = all_data["clip_embedding"]
        # this is all clip_embeddings of images, concatenated together of shape: ([num_images, 512])
        # captions_raw = all_data["captions"]
        captions_raw = all_data["my_captions"]
        # captions is a list all strings

        self.caption2embedding = all_data["caption2embedding"]

        print('building prefixes done')



        # self.image_ids = [caption["image_id"] for caption in captions_raw]
        # self.captions = [caption['caption'] for caption in captions_raw]
        
        
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            # self.caption2embedding = [] # maps caption index to index of corresponding image embedding in self.prefixes
            # print('cap2emb ', self.caption2embedding)
            max_seq_len = 0
            for i, caption in tqdm(enumerate(captions_raw)):
                # self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64))

                # print('i ', i)

        
                # self.caption2embedding.append(self.caption2embedding[i])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            print('captions tokens shape ', len(self.captions_tokens))
            print('caption2embedding shape ', self.caption2embedding[0].shape)


            # with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
            #     pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser


def train(dataset: ClipCocoDataset, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):
    
    print('starting train')


    lr = clip_caption_model_train_hyperparameters['lr']


    device = torch.device('cuda:0')
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )


    # print model parameters that are trainable
    # print('model parameters')
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data.shape)
    # save_config(args)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}_{selected_clip_model.name}.pt"),
            )
            # save_config(args)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./caption_dataset/split_train.pkl')
    parser.add_argument('--out_dir', default='./caption_checkpoints')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    # parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=clip_caption_model_train_hyperparameters['n_epochs'])
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    # parser.add_argument('--bs', type=int, default=40)
    parser.add_argument('--bs', type=int, default=clip_caption_model_train_hyperparameters['batch_size'])
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    args = parser.parse_args()

    args.save_every = clip_caption_model_train_hyperparameters['save_every']

    print("selected clip model ", selected_clip_model.name)

    # args.only_prefix = True

    if clip_caption_model_train_hyperparameters['model_config'] == ClipCaptionModelMapping.TRANSFORMER:
        
        args.only_prefix = True
        args.mapping_type = 'transformer'
        args.num_layers = 8
        args.prefix_length = 40
        args.prefix_length_clip = 40
    

    prefix_length = args.prefix_length
        
    



    dataset = ClipCocoDataset(args.data, prefix_length, normalize_prefix=args.normalize_prefix)
    print('data setup done')
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]

     # device cuda or cpu 
    device = "cuda" if torch.cuda.is_available() else "cpu"


    if selected_clip_model == ClipModels.FINETUNED:
        args.prefix = "finetuned_clip_coco_prefix"
    elif selected_clip_model == ClipModels.FINETUNED_TEMP:
        args.prefix = "finetuned_temp_clip_coco_prefix"
    elif selected_clip_model == ClipModels.DEFAULT:
        args.prefix = "default_clip_coco_prefix"

    if clip_caption_model_train_hyperparameters['model_config'] == ClipCaptionModelMapping.TRANSFORMER:
        # add the word transformer_ to the prefix
        args.prefix = 'transformer_' + args.prefix


    print('args.only_prefix ', args.only_prefix)
    if args.only_prefix:
        print('training only prefix')
        model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        # this freezes the gpt's parameters
        
    else:
        print('training prefix and gpt')
        model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        # this trains gpt as well

        print("Train both prefix and GPT")
        sys.stdout.flush()

    if not clip_caption_model_train_hyperparameters['train_from_scratch']:
            print('loading pretrained caption model')
        
            # get model weights from pretrained caption model
            model_path = "caption_checkpoints/coco_weights.pt"
            altered_state_dict = torch.load(model_path, map_location=device)
            for i in range(12):
                del altered_state_dict['gpt.transformer.h.' + str(i) + '.attn.bias']
                del altered_state_dict['gpt.transformer.h.' + str(i) + '.attn.masked_bias']
            model.load_state_dict(altered_state_dict)
            # now, model has same weights as the authors had when they trained the caption model
    else:
        print('DID NOT load pretrained caption model')


    if clip_caption_model_train_hyperparameters['continue_train_from_prev_checkpoint']:
        print()
        print('Continuing training from prev checkpoint')
        print()
        model.load_state_dict(torch.load(os.path.join(args.out_dir, f"{args.prefix}-{clip_caption_model_train_hyperparameters['prev_checkpoint_epoch']:03d}.pt"), map_location=device))
            
    
    train(dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix)


if __name__ == '__main__':
    main()
