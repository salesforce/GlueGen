import os
import pandas as pd
import pdb
import time

import torch 
from torch import cuda
import torch.nn as nn

import numpy as np


def train_alignment(epoch, tokenizer_src, tokenizer_tar, model_src, model_tar, adapter, discriminator, device, loader, optimizer, params, console):
    len_dataloader = len(loader)
    
    adapter.train()
    adapter_rec.train()
    
    discriminator.train()
    discriminator_token.train()

    domain_label_s = torch.zeros(params["TRAIN_BATCH_SIZE"]).to(device)
    domain_label_s = domain_label_s.long()

    domain_label_t = torch.ones(params["TRAIN_BATCH_SIZE"]).to(device)
    domain_label_t = domain_label_t.long()
    
    domain_label_s_tokenWise = torch.zeros(params["TRAIN_BATCH_SIZE"] * params["MAX_LENGTH"]).to(device)
    domain_label_s_tokenWise = domain_label_s_tokenWise.long()

    domain_label_t_tokenWise = torch.ones(params["TRAIN_BATCH_SIZE"] * params["MAX_LENGTH"]).to(device)
    domain_label_t_tokenWise = domain_label_t_tokenWise.long()
    
    for i, data in enumerate(loader):
        p = float(i + epoch * len_dataloader) / params["TRAIN_EPOCHS"] / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        y = data["Text"]#.to(device)
        tokens_src_raw  = tokenizer_src(y, truncation=True, max_length=params["MAX_LENGTH"], return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens_src = tokens_src_raw["input_ids"].to(device)
        
        feat_src = model_src(tokens_src, return_embeddings=True)

        tokens_tar_raw = tokenizer_tar(y, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens_tar = tokens_tar_raw["input_ids"].to(device)

        feat_tar_all = model_tar.encoder(input_ids=tokens_tar, return_dict=True) # key line
        feat_tar = feat_tar_all.last_hidden_state
        
        feat_tar_ad = adapter(feat_tar)
#         feat_tar_tonW = adapter_token(feat_tar)
        feat_tar_rec = adapter_rec(feat_tar_ad)
        
        
#         size_0, size_1, size_2 = feat_src.shape

        
#         feat_src_senWise = torch.mean(feat_src, dim=1)
#         feat_tar_senWise = torch.mean(feat_tar_ad, dim=1)
        
#         feat_src_tokenWise = feat_src.view(size_0 * size_1, size_2)
#         feat_tar_tokenWise = feat_tar_ad.reshape(size_0 * size_1, size_2)
        
#         domain_pred_s_tokenWise = discriminator_token(feat_src_tokenWise, alpha)
#         domain_pred_t_tokenWise = discriminator_token(feat_tar_tokenWise, alpha)
        
#         domain_pred_s_senWise = discriminator(feat_src_senWise, alpha)
#         domain_pred_t_senWise = discriminator(feat_tar_senWise, alpha)
        
        loss_distill = loss_mse(feat_tar_ad, feat_src)
        loss_rec = loss_mse(feat_tar_rec, feat_tar)
        loss = loss_distill + loss_rec
        
#         loss = 0.1 * (loss_domain(domain_pred_s_senWise, domain_label_s) + loss_domain(domain_pred_t_senWise, domain_label_t) + \
#         loss_domain(domain_pred_s_tokenWise, domain_label_s_tokenWise) + loss_domain(domain_pred_t_tokenWise, domain_label_t_tokenWise)) + loss_distill
        
        if i % 10 == 0:
            p_ = float(i + epoch * len_dataloader + 1) / params["TRAIN_EPOCHS"] / len_dataloader
            time_all = time.time() - time_start
            time_remain = time_all / p_ - time_all
            m, s = divmod(time_remain, 60) 
            h, m = divmod(m, 60)
            
            time_remain_str = 'Remain:{:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s))
            
            training_logger.add_row(str(epoch), str(i), str(round(alpha, 3) ), time_remain_str, str(round(loss_rec.item(),4)), str(round(loss_distill.item(),4)))
            console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()