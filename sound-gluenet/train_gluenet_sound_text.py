'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
'''
import os
import pandas as pd
import pdb
import time

import torch 
from torch import cuda
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from rich.console import Console

import numpy as np

from model import Discriminator, Translator_w_head_v0

import json
import random
from rich.table import Column, Table
from rich import box
import random

from util.utils import get_dataloader

#----------------------------------------------------------------------------

console = Console(record=True)

params = {
    "TRAIN_BATCH_SIZE": 8,  # training batch size
    "TRAIN_EPOCHS": 10,  # number of training epochs
    "LEARNING_RATE": 5e-5,  # learning rate
    "MAX_LENGTH": 77
}


image_templates = [
    'a vivid photo of {}',
    'a vivid rendering of {}',
    'the vivid photo of {}',
    'a vivid photo of {}',
    'a good photo of {}',
    'a vivid illustration of {}',
]

training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Alpha", justify="center"),
    Column("Time Left", justify="center"),
    Column("Loss", justify="center"),
    Column("LossDT", justify="center"),
    Column("LossDTR", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

device = torch.device('cuda')


console.log(f"[Loading Models]...\n")

version = "openai/clip-vit-large-patch14"
srcTokenizer = CLIPTokenizer.from_pretrained(version)

srcTransformer = CLIPTextModel.from_pretrained(version)
srcTransformer = srcTransformer.to(device)
srcTransformer.eval()

from audioclip.model import AudioCLIP
model_path = '../checkpoints_all/audioclip_checkpoint/AudioCLIP-Full-Training.pt'
tarModel = AudioCLIP(pretrained=model_path)


tarModel = tarModel.to(device)
tarModel.eval()

console.log(f"[Data]: Reading data...\n")
config_path = '../stable-diffusion/audioclip/protocols/audioclip-us8k.json'
train_loader_us8k, _ = get_dataloader(config_path, params)

train_loader = [train_loader_us8k]

adapter = Translator_w_head_v0(16, 77, 64, 768, 4, 5).to(device)
adapter_rec = Translator_w_head_v0(77, 16, 768, 64,  4, 5).to(device)

adapter = adapter.to(device)

adapter_rec = adapter_rec.to(device)

discriminator_token = Discriminator(768, 2).to(device)

parameters = list(adapter.parameters()) + \
            list(discriminator_token.parameters()) + list(adapter_rec.parameters())

optimizer = torch.optim.Adam(
        params=parameters, lr=params["LEARNING_RATE"]
    )

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

loss_domain = torch.nn.NLLLoss()
loss_domain = loss_domain.to(device)

loss_mse = torch.nn.MSELoss(reduction='none')

tensor_norm = torch.Tensor([[43.8203],[28.3668],[27.9345],[28.0084],[28.2958],[28.2576],[28.3373],[28.2695],[28.4097],[28.2790],[28.2825],[28.2807],[28.2775],[28.2708],[28.2682],[28.2624],[28.2589],[28.2611],[28.2616],[28.2639],[28.2613],[28.2566],[28.2615],[28.2665],[28.2799],[28.2885],[28.2852],[28.2863],[28.2780],[28.2818],[28.2764],[28.2532],[28.2412],[28.2336],[28.2514],[28.2734],[28.2763],[28.2977],[28.2971],[28.2948],[28.2818],[28.2676],[28.2831],[28.2890],[28.2979],[28.2999],[28.3117],[28.3363],[28.3554],[28.3626],[28.3589],[28.3597],[28.3543],[28.3660],[28.3731],[28.3717],[28.3812],[28.3753],[28.3810],[28.3777],[28.3693],[28.3713],[28.3670],[28.3691],[28.3679],[28.3624],[28.3703],[28.3703],[28.3720],[28.3594],[28.3576],[28.3562],[28.3438],[28.3376],[28.3389],[28.3433],[28.3191]]).to(device)

tensor_norm_ave = (tensor_norm/tensor_norm.mean()).unsqueeze(0)

word_dict = {'jackhammer': ['jackhammer', 'pneumatic drill', 'concrete breaker', 'demolition hammer'],
 'drilling': ['a machine is drilling', 'a machine is piercing', 'a machine is perforating', 'a machine is reaming'],
 'siren': ['patrol car, driving', 'ambulance, driving', 'fire truck, driving', 'fire alarm'],
 'car horn': ['car horning', 'automobile horning', 'car beeping'],
 'street music': ['street music, artists are performing', 'sidewalk music, artists are performing', 'outdoor music, artists are performing'], 
 'engine idling': [ 'engine is idling', 'engine is running', 'parked engine is running'],
 'dog bark': ['dog is barking', 'dogs are barking', 'puppy is barking', 'doggy is barking'],
 'children playing': ['children are playing'],
 'air conditioner': ['air conditioner'],
 'gun shot': ['gun is shoting', 'gunfire']
}

def train_alignment(epoch, tokenizer_src, model_src, model_tar, adapter, adapter_rec, device, loader, optimizer, params, console):
    loader_us8k = loader[0]
    len_dataloader = len(loader_us8k) # + len(loader_esc) 
    
    adapter.train()
    adapter_rec.train()
    
    discriminator_token.train()
    
    domain_label_s_tokenWise = torch.zeros(params["TRAIN_BATCH_SIZE"] * params["MAX_LENGTH"]).to(device)
    domain_label_s_tokenWise = domain_label_s_tokenWise.long()

    domain_label_t_tokenWise = torch.ones(params["TRAIN_BATCH_SIZE"] * params["MAX_LENGTH"]).to(device)
    domain_label_t_tokenWise = domain_label_t_tokenWise.long()
    
    
    loss_list= [0, 0, 0]
    
    for i, data in enumerate(loader_us8k):
        y = []
        for item in data[2]:
            word = item[0]
            word_sel = random.choice(word_dict[word])
            item = random.choice(image_templates).format(word_sel) if random.random() < 1 else word_sel
            y.append(item) 
            
        tokens_src_raw  = tokenizer_src(y, truncation=True, max_length=params["MAX_LENGTH"], return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        
        tokens_src = tokens_src_raw["input_ids"].to(device)
        
        feat_src_ = model_src(tokens_src).last_hidden_state.detach()
        
        feat_src = feat_src_ / (tensor_norm/2) 

        ((feat_tar, _, _), _), _ = model_tar(audio=data[0].to(device))
        
        feat_tar = feat_tar.reshape(feat_tar.shape[0], 16, 1024 // 16).detach()
        
        feat_tar_ad = adapter(feat_tar)
        feat_tar_rec = adapter_rec(feat_tar_ad)
        
        size_0, size_1, size_2 = feat_src.shape
        
        feat_src_tokenWise = feat_src.view(size_0 * size_1, size_2)
        feat_tar_tokenWise = feat_tar_ad.reshape(size_0 * size_1, size_2)

        pred_s_tokenWise = discriminator_token(feat_src_tokenWise)
        pred_t_tokenWise = discriminator_token(feat_tar_tokenWise)

        feat_src_mean = torch.mean(feat_src_, dim=0)
        feat_src_norm = torch.mean(torch.abs(feat_src_mean - feat_src_mean[-1,:]),dim=1)
        feat_src_norm = feat_src_norm / feat_src_norm.mean()
        feat_src_norm[-1] = (feat_src_norm[-2] + feat_src_norm[-3] ) /2
        feat_src_norm = feat_src_norm.unsqueeze(0).unsqueeze(2)
        
        loss_adv = 0.1 * (loss_domain(pred_s_tokenWise, domain_label_s_tokenWise) + loss_domain(pred_t_tokenWise, domain_label_t_tokenWise))
        
        loss_distill = 10000 * (loss_mse(feat_tar_ad * 1, feat_src* 1) * feat_src_norm).mean() # 
        loss_rec = 10000 * (loss_mse(feat_tar_rec* 1, feat_tar* 1)).mean()
        
        loss_distill_rec = 10 * loss_mse(feat_tar_ad * (tensor_norm/2), feat_src_).mean() #
        
        loss = loss_distill + loss_rec + loss_adv
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_list[0] += loss_rec.item()
        loss_list[1] += loss_distill.item()
        loss_list[2] += loss_distill_rec.item()
            
        if i > 0 and i % 20 == 0:
            p_ = float(i   + epoch * len_dataloader + 1) / params["TRAIN_EPOCHS"] / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p_)) - 1
            time_all = time.time() - time_start
            time_remain = time_all / p_ - time_all
            m, s = divmod(time_remain, 60) 
            h, m = divmod(m, 60)

            time_remain_str = 'Remain:{:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s))
            
            loss_list = [item/20 for item in loss_list]
            
            training_logger.add_row(str(epoch), str(i), str(round(alpha, 3) ), time_remain_str, str(round(loss_list[0],4)), str(round(loss_list[1],4)), str(round(loss_list[2],4)))
            console.print(training_logger)
            loss_list= [0, 0, 0]
    
console.log(f"[Begin Training]...\n")

time_start = time.time()
for epoch in range(params["TRAIN_EPOCHS"]):
    train_alignment(epoch, srcTokenizer, srcTransformer, tarModel, adapter, adapter_rec, device, train_loader, optimizer, params, console)
    scheduler.step()

PATH_ad = 'gluenet_sound2img_audioclip_us8k.ckpt'
adapter_to_save = adapter.module if hasattr(adapter, "module") else adapter  # Take care of distributed/parallel training
torch.save(adapter_to_save.state_dict(), PATH_ad)
