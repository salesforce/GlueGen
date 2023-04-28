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
from rich.console import Console
import numpy as np

from model import Discriminator, Translator, Translator_noln
from dataset import CustomTextDataset
from transformers import AutoTokenizer, AutoModel

import argparse
import random

#----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='multilingual translation')
    parser.add_argument('--DATA_PATH_SRC', type=str, default='data/laion-01M-trans-en.txt',
                        help='dir to src')
    parser.add_argument('--DATA_PATH_TAR', type=str, default='data/laion-01M-trans-zh-cn.txt',
                        help='dir to tar')
    parser.add_argument('--DATA_PATH_SRC_1', type=str, default='',
                        help='dir to src')
    parser.add_argument('--DATA_PATH_TAR_1', type=str, default='',
                        help='dir to tar')
    parser.add_argument('--tarLanguage', type=str, default='Chinese',
                        help='target Language')
    args = parser.parse_args()
    return args

args = parse_args()
console = Console(record=True)

params = {
    "TRAIN_BATCH_SIZE": 32,  # training batch size
    "TRAIN_EPOCHS": 4,  # number of training epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_LENGTH": 77,
    "DATA_PATH_SRC" : args.DATA_PATH_SRC,
    "DATA_PATH_TAR" : args.DATA_PATH_TAR,
}

print('Called with args:')
print(args)

device = 'cuda' if cuda.is_available() else 'cpu'

console.log(f"[Loading Models]...\n")

from transformers import CLIPTokenizer, CLIPTextModel

version = "openai/clip-vit-large-patch14"
srcTokenizer = CLIPTokenizer.from_pretrained(version)

srcTransformer = CLIPTextModel.from_pretrained(version)
srcTransformer = srcTransformer.to(device)
srcTransformer.eval()
tokenizerRob = AutoTokenizer.from_pretrained('xlm-roberta-large')
modelRob = AutoModel.from_pretrained("xlm-roberta-large")
modelRob = modelRob.to(device)

from rich.table import Column, Table
from rich import box

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
        
console.log(f"[Data]: Reading data...\n")


def open_txt_2_loader(path):
    with open(path) as f:
        cap_data_raw = f.readlines()
        
    cap_data = []
    
    for item in cap_data_raw:
        cap_data.append(item[:-1])
        
    text_tr = pd.DataFrame({'Text': cap_data})
    TD_tr = CustomTextDataset(text_tr['Text'])
    
    train_params = {
        "batch_size": params["TRAIN_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 4,
        "drop_last": True,
    }
    training_loader = DataLoader(TD_tr, **train_params)
    
    return cap_data, training_loader

def open_2txt_2_list(path, path_1):
    with open(path) as f:
        cap_data_raw = f.readlines()
        
    cap_data = []
    
    for item in cap_data_raw:
        cap_data.append(item[:-1])
    
    with open(path_1) as f:
        cap_data_raw_1 = f.readlines()
        
    for item in cap_data_raw_1:
        cap_data.append(item[:-1] + '.')
        
    return cap_data


path_s = params["DATA_PATH_SRC"]
path_t = params["DATA_PATH_TAR"]

train_params = {
        "batch_size": params["TRAIN_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 4,
        "drop_last": True,
    }


if args.DATA_PATH_SRC_1 and args.DATA_PATH_TAR_1:
    path_s_1 = args.DATA_PATH_SRC_1
    path_t_1 = args.DATA_PATH_TAR_1
    
    cap_data_s = open_2txt_2_list(path_s, path_s_1)
    cap_data_t = open_2txt_2_list(path_t, path_t_1)
    assert len(cap_data_s) == len(cap_data_t)
    
    randseed = random.randint(0, 100)
    random.seed(randseed)
    
    random.shuffle(cap_data_s)
    random.seed(randseed)
    random.shuffle(cap_data_t)
    
    text_tr_s = pd.DataFrame({'Text': cap_data_s})
    TD_tr_s = CustomTextDataset(text_tr_s['Text'])
    loader_src = DataLoader(TD_tr_s, **train_params)
    
    text_tr_t = pd.DataFrame({'Text': cap_data_t})
    TD_tr_t = CustomTextDataset(text_tr_t['Text'])
    loader_tar = DataLoader(TD_tr_t, **train_params)
    
else:
    cap_data_s, loader_src = open_txt_2_loader(path_s)
    cap_data_t, loader_tar = open_txt_2_loader(path_t) 

len_src = len(cap_data_s)
len_tar = len(cap_data_t)
assert len_src == len_tar

print('length:', len_src)


adapter = Translator_noln(77, 1024, 768, 2, 5).to(device)
adapter_rec = Translator_noln(77, 768, 1024, 2, 5).to(device)


discriminator_token = Discriminator(768, 2).to(device)


parameters = list(adapter.parameters()) + \
            list(discriminator_token.parameters()) + list(adapter_rec.parameters())

optimizer = torch.optim.Adam(
        params=parameters, lr=params["LEARNING_RATE"]
    )

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

loss_domain = torch.nn.NLLLoss()
loss_domain = loss_domain.to(device)

loss_mse_wise = torch.nn.MSELoss(reduction='none')
loss_mse_wise = loss_mse_wise.to(device)

loss_mse = torch.nn.MSELoss()
loss_mse = loss_mse.to(device)

tensor_norm = torch.Tensor([[43.8203],[28.3668],[27.9345],[28.0084],[28.2958],[28.2576],[28.3373],[28.2695],[28.4097],[28.2790],[28.2825],[28.2807],[28.2775],[28.2708],[28.2682],[28.2624],[28.2589],[28.2611],[28.2616],[28.2639],[28.2613],[28.2566],[28.2615],[28.2665],[28.2799],[28.2885],[28.2852],[28.2863],[28.2780],[28.2818],[28.2764],[28.2532],[28.2412],[28.2336],[28.2514],[28.2734],[28.2763],[28.2977],[28.2971],[28.2948],[28.2818],[28.2676],[28.2831],[28.2890],[28.2979],[28.2999],[28.3117],[28.3363],[28.3554],[28.3626],[28.3589],[28.3597],[28.3543],[28.3660],[28.3731],[28.3717],[28.3812],[28.3753],[28.3810],[28.3777],[28.3693],[28.3713],[28.3670],[28.3691],[28.3679],[28.3624],[28.3703],[28.3703],[28.3720],[28.3594],[28.3576],[28.3562],[28.3438],[28.3376],[28.3389],[28.3433],[28.3191]]).to(device)

tensor_norm_ave = (tensor_norm/tensor_norm.mean()).unsqueeze(0)

def train_alignment(epoch, tokenizer_src, tokenizer_tar, model_src, model_tar, adapter, device, loader_s, loader_t, optimizer, params, console):
    len_dataloader = len(loader_s)
    
    adapter.train()
    adapter_rec.train()
    
    discriminator_token.train()
    
    domain_label_s_tokenWise = torch.zeros(params["TRAIN_BATCH_SIZE"] * params["MAX_LENGTH"]).to(device)
    domain_label_s_tokenWise = domain_label_s_tokenWise.long()

    domain_label_t_tokenWise = torch.ones(params["TRAIN_BATCH_SIZE"] * params["MAX_LENGTH"]).to(device)
    domain_label_t_tokenWise = domain_label_t_tokenWise.long()
    
    loss_list= [0, 0, 0]
    
    for i, data in enumerate(zip(loader_s, loader_t)):
        p = float(i + epoch * len_dataloader) / params["TRAIN_EPOCHS"] / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        data_s = data[0]
        data_t = data[1]
        
        with torch.no_grad():
            text_s = data_s["Text"]#.to(device)
            tokens_src_raw  = tokenizer_src(text_s, truncation=True, max_length=params["MAX_LENGTH"], return_length=True,
                                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens_src = tokens_src_raw["input_ids"].to(device)

            feat_src = model_src(tokens_src).last_hidden_state.detach()
            feat_src_over = feat_src / (tensor_norm/2)

            text_t = data_t["Text"]

            tokens_tar = tokenizer_tar(text_t, truncation=True, max_length=77, padding="max_length", return_tensors='pt').to(device)
            output_tar = model_tar(**tokens_tar)

            feat_tar = output_tar['last_hidden_state'] / 3

        feat_tar_ad = adapter(feat_tar)
        feat_tar_rec = adapter_rec(feat_tar_ad)
        
        size_0, size_1, size_2 = feat_src.shape
        
        feat_src_tokenWise = feat_src.view(size_0 * size_1, size_2)
        feat_tar_tokenWise = feat_tar_ad.reshape(size_0 * size_1, size_2)
        
        pred_s_tokenWise = discriminator_token(feat_src_tokenWise)
        pred_t_tokenWise = discriminator_token(feat_tar_tokenWise)
        
        feat_src_mean = torch.mean(feat_src, dim=0)
        feat_src_norm = torch.mean(torch.abs(feat_src_mean - feat_src_mean[-1,:]),dim=1)
        feat_src_norm = feat_src_norm / feat_src_norm.mean()
        feat_src_norm[-1] = (feat_src_norm[-2] + feat_src_norm[-3] ) /2
        feat_src_norm = feat_src_norm.unsqueeze(0).unsqueeze(2)
        
        loss_adv = 0.1 * (loss_domain(pred_s_tokenWise, domain_label_s_tokenWise) + loss_domain(pred_t_tokenWise, domain_label_t_tokenWise))
        
        loss_distill = 10000 * (loss_mse_wise(feat_tar_ad * 1, feat_src_over* 1) * feat_src_norm).mean() # 
        loss_rec = 10000 * (loss_mse_wise(feat_tar_rec* 1, feat_tar* 1)).mean()
        
        loss_distill_rec = 10 * loss_mse_wise(feat_tar_ad * (tensor_norm/2), feat_src).mean()
                          
        loss = loss_distill + loss_rec + loss_adv
        
        loss_list[0] += loss_rec.item()
        loss_list[1] += loss_distill.item()
        loss_list[2] += loss_distill_rec.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        if i > 0 and i % 500 == 0:
            p_ = float(i + epoch * len_dataloader + 1) / params["TRAIN_EPOCHS"] / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p_)) - 1
            time_all = time.time() - time_start
            time_remain = time_all / p_ - time_all
            m, s = divmod(time_remain, 60) 
            h, m = divmod(m, 60)
            
            time_remain_str = 'Remain:{:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s))
            
            loss_list = [item/500 for item in loss_list]
            
            training_logger.add_row(str(epoch), str(i), str(round(alpha, 3) ), time_remain_str, str(round(loss_list[0],4)), str(round(loss_list[1],4)), str(round(loss_list[2],4)))
            console.print(training_logger)
            loss_list= [0, 0, 0]


console.log(f"[Begin Training]...\n")

time_start = time.time()
for epoch in range(params["TRAIN_EPOCHS"]):
    train_alignment(epoch, srcTokenizer, tokenizerRob,srcTransformer, modelRob, adapter, device, loader_src, loader_tar, optimizer, params, console)
    scheduler.step()
        
PATH_ad = 'gluenet_multi_' + args.tarLanguage + '.ckpt'
torch.save(adapter.state_dict(), PATH_ad)