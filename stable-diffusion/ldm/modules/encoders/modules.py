'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
 * Modified from Stable Diffusion repo: https://github.com/CompVis/stable-diffusion
 * Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors
'''


import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import kornia

from transformers import T5Tokenizer
from transformers import T5Model

from .translators import Translator_clip as Translator
from .translators import Translator_noln

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForMaskedLM, AutoModel

import open_clip
from transformers import T5ForConditionalGeneration, AutoTokenizer

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()
#         self.mean_list = []
#         self.std_list = []
#         self.global_ep = 0

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))



class RobMlgEmbedder_multi(AbstractEncoder):
    """Uses the BertMlgEmbedder tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed=768, n_layer=32, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        
        self.tknz_fn = AutoTokenizer.from_pretrained('xlm-roberta-large')
        self.model = AutoModel.from_pretrained("xlm-roberta-large").to(device)
        self.device = device
        
    
        path_ad_spanish = '../checkpoints_all/gluenet_checkpoint/gluenet_Spanish_clip_overnorm_over3_noln.ckpt'
        self.adapter_spanish = Translator_noln(77, 1024, 768, 2, 5)
        self.adapter_spanish.load_state_dict(torch.load(path_ad_spanish))
        
        path_ad_chinese = '../checkpoints_all/gluenet_checkpoint/gluenet_Chinese_clip_overnorm_over3_noln.ckpt'
        self.adapter_chinese = Translator_noln(77, 1024, 768, 2, 5)
        self.adapter_chinese.load_state_dict(torch.load(path_ad_chinese))
        
        
        path_ad_italian = '../checkpoints_all/gluenet_checkpoint/gluenet_Italian_clip_overnorm_over3_noln.ckpt'
        self.adapter_italian = Translator_noln(77, 1024, 768, 2, 5)
        self.adapter_italian.load_state_dict(torch.load(path_ad_italian))
        
        path_ad_french = '../checkpoints_all/gluenet_checkpoint/gluenet_French_clip_overnorm_over3_noln.ckpt'
        self.adapter_french = Translator_noln(77, 1024, 768, 2, 5)
        self.adapter_french.load_state_dict(torch.load(path_ad_french))
        
        path_ad_japanese = '../checkpoints_all/gluenet_checkpoint/gluenet_Japanese_clip_overnorm_over3_noln.ckpt'
        self.adapter_japanese = Translator_noln(77, 1024, 768, 2, 5)
        self.adapter_japanese.load_state_dict(torch.load(path_ad_japanese))

        self.tensor_norm = torch.Tensor([[43.8203],[28.3668],[27.9345],[28.0084],[28.2958],[28.2576],[28.3373],[28.2695],[28.4097],[28.2790],[28.2825],[28.2807],[28.2775],[28.2708],[28.2682],[28.2624],[28.2589],[28.2611],[28.2616],[28.2639],[28.2613],[28.2566],[28.2615],[28.2665],[28.2799],[28.2885],[28.2852],[28.2863],[28.2780],[28.2818],[28.2764],[28.2532],[28.2412],[28.2336],[28.2514],[28.2734],[28.2763],[28.2977],[28.2971],[28.2948],[28.2818],[28.2676],[28.2831],[28.2890],[28.2979],[28.2999],[28.3117],[28.3363],[28.3554],[28.3626],[28.3589],[28.3597],[28.3543],[28.3660],[28.3731],[28.3717],[28.3812],[28.3753],[28.3810],[28.3777],[28.3693],[28.3713],[28.3670],[28.3691],[28.3679],[28.3624],[28.3703],[28.3703],[28.3720],[28.3594],[28.3576],[28.3562],[28.3438],[28.3376],[28.3389],[28.3433],[28.3191]]).to(device)
        
        
    def forward(self, text):
        if '__' in text[0]:
            text[0], language = text[0].split('__')
            if language == 'Spanish':
                    self.adapter = self.adapter_spanish
            elif language == 'Chinese':
                    self.adapter = self.adapter_chinese
            elif language == 'Italian':
                    self.adapter = self.adapter_italian
            elif language == 'French':
                    self.adapter = self.adapter_french
            elif language == 'Japanese':
                    self.adapter = self.adapter_japanese
            else:
                raise NameError('No Matched Languages')
        else:
            raise NameError('No Matched Languages')
            
        with  torch.no_grad():
            tokens = self.tknz_fn(text, truncation=True, max_length=77, padding="max_length", return_tensors='pt').to(self.device)

            output = self.model(**tokens)
            sel_feat = output['last_hidden_state'] / 3
        
        sel_feat_ip = self.adapter(sel_feat) * (self.tensor_norm/2)
        return sel_feat_ip #z

    def encode(self, text):
        return self(text)
    
    

from audioclip.model import AudioCLIP 
from .translators import Translator_w_head
class FrozenAudioCLIPEmbedder(nn.Module):
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        
        self.model = AudioCLIP(pretrained='../checkpoints_all/audioclip_checkpoint/AudioCLIP-Full-Training.pt')
        self.device = device
        
        self.adapter = Translator_w_head(16, 77, 64, 768, 4, 5) #.to(device)
        
        path_ad = '../checkpoints_all/gluenet_checkpoint/gluenet_sound2img_audioclip_us8k.ckpt'
        self.adapter.load_state_dict(torch.load(path_ad, map_location=torch.device('cpu')))
        
        self.tensor_norm = torch.Tensor([[43.8203],[28.3668],[27.9345],[28.0084],[28.2958],[28.2576],[28.3373],[28.2695],[28.4097],[28.2790],[28.2825],[28.2807],[28.2775],[28.2708],[28.2682],[28.2624],[28.2589],[28.2611],[28.2616],[28.2639],[28.2613],[28.2566],[28.2615],[28.2665],[28.2799],[28.2885],[28.2852],[28.2863],[28.2780],[28.2818],[28.2764],[28.2532],[28.2412],[28.2336],[28.2514],[28.2734],[28.2763],[28.2977],[28.2971],[28.2948],[28.2818],[28.2676],[28.2831],[28.2890],[28.2979],[28.2999],[28.3117],[28.3363],[28.3554],[28.3626],[28.3589],[28.3597],[28.3543],[28.3660],[28.3731],[28.3717],[28.3812],[28.3753],[28.3810],[28.3777],[28.3693],[28.3713],[28.3670],[28.3691],[28.3679],[28.3624],[28.3703],[28.3703],[28.3720],[28.3594],[28.3576],[28.3562],[28.3438],[28.3376],[28.3389],[28.3433],[28.3191]]).to(device)
        
    def forward(self, audio):
        ((z, _, _), _), _ = self.model(audio=audio)
        z = z.reshape(z.shape[0], 16, 1024 // 16)
        z = self.adapter(z) * (self.tensor_norm/2)
        
        return z

    def encode(self, text):
        return self(text)
    
if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)