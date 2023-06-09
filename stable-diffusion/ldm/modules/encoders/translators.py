"""
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
"""

import pdb

import torch 
import torch.nn as nn
from ldm.modules.x_transformer import AttentionLayers
#----------------------------------------------------------------------------
class translator_base(nn.Module):
    def __init__(self, num_tok, dim, dim_out, mult=2):
        super().__init__()
        
        self.dim_in = dim
        self.dim_out = dim_out

        self.net_tok = nn.Sequential(
            nn.Linear(num_tok, int(num_tok * mult)),
            nn.LayerNorm(int(num_tok * mult)),
            nn.GELU(),
            nn.Linear(int(num_tok * mult), int(num_tok * mult)),
            nn.LayerNorm(int(num_tok * mult)),
            nn.GELU(),
            nn.Linear(int(num_tok * mult), num_tok),
            nn.LayerNorm(num_tok),
            
        )
        
        self.net_sen = nn.Sequential(
            nn.Linear(dim, int(dim * mult)),
            nn.LayerNorm(int(dim * mult)),
            nn.GELU(),
            nn.Linear(int(dim * mult), int(dim * mult)),
            nn.LayerNorm(int(dim * mult)),
            nn.GELU(),
            nn.Linear(int(dim * mult), dim_out),
            nn.LayerNorm(dim_out)
        )

    def forward(self, x):
        if self.dim_in == self.dim_out:
            indentity_0 = x
            x = self.net_sen(x)
            x += indentity_0
            x = x.transpose(1,2)

            indentity_1 = x
            x = self.net_tok(x)
            x += indentity_1
            x = x.transpose(1,2)
        else:
            x = self.net_sen(x)
            x = x.transpose(1,2)

            x = self.net_tok(x)
            x = x.transpose(1,2)
        return x

class Translator(nn.Module):
    def __init__(self, num_tok, dim, dim_out, mult=2, depth=5):
        super().__init__()
        
        self.blocks = nn.ModuleList(
            [translator_base(num_tok, dim, dim, mult=2)
                for d in range(depth)]
        )
        self.gelu = nn.GELU()

        self.tail = translator_base(num_tok, dim, dim_out, mult=2)
        
    def forward(self, x):
        
        for block in self.blocks:
            x = block(x) + x
            x = self.gelu(x)
            
        x = self.tail(x)
        return x
    
class translator_base_noln(nn.Module):
    def __init__(self, num_tok, dim, dim_out, mult=2):
        super().__init__()
        
        self.dim_in = dim
        self.dim_out = dim_out

        self.net_tok = nn.Sequential(
            nn.Linear(num_tok, int(num_tok * mult)),
            nn.GELU(),
            nn.Linear(int(num_tok * mult), int(num_tok * mult)),
            nn.GELU(),
            nn.Linear(int(num_tok * mult), num_tok),            
        )
        
        self.net_sen = nn.Sequential(
            nn.Linear(dim, int(dim * mult)),
            nn.GELU(),
            nn.Linear(int(dim * mult), int(dim * mult)),
            nn.GELU(),
            nn.Linear(int(dim * mult), dim_out),
        )

    def forward(self, x):
        if self.dim_in == self.dim_out:
            indentity_0 = x
            x = self.net_sen(x)
            x += indentity_0
            x = x.transpose(1,2)

            indentity_1 = x
            x = self.net_tok(x)
            x += indentity_1
            x = x.transpose(1,2)
        else:
            x = self.net_sen(x)
            x = x.transpose(1,2)

            x = self.net_tok(x)
            x = x.transpose(1,2)
        return x
    
class Translator_noln(nn.Module):
    def __init__(self, num_tok, dim, dim_out, mult=2, depth=5):
        super().__init__()
        
        self.blocks = nn.ModuleList(
            [translator_base(num_tok, dim, dim, mult=2)
                for d in range(depth)]
        )
        self.gelu = nn.GELU()

        self.tail = translator_base_noln(num_tok, dim, dim_out, mult=2)
        
    def forward(self, x):
        
        for block in self.blocks:
            x = block(x) + x
            x = self.gelu(x)
            
        x = self.tail(x)
        return x
class Translator_attn(nn.Module):
    def __init__(self, num_tok, dim, dim_out, mult=2, depth=5):
        super().__init__()
        self.attn = AttentionLayers(dim=dim, depth=1)
        
        self.blocks = nn.ModuleList(
            [translator_base(num_tok, dim, dim, mult=2)
                for d in range(depth)]
        )
        self.gelu = nn.GELU()

        self.tail = translator_base(num_tok, dim, dim_out, mult=2)
        
    def forward(self, x):
        
        x = self.attn(x)
        for block in self.blocks:
            x = block(x) + x
            x = self.gelu(x)
            
        x = self.tail(x)
        return x


class translator_clip(nn.Module):
    def __init__(self, num_tok, dim, dim_out, mult=2):
        super().__init__()
        
        self.dim_in = dim
        self.dim_out = dim_out

        self.net_tok = nn.Sequential(
            nn.Linear(num_tok, int(num_tok * mult)),
            nn.LayerNorm(int(num_tok * mult)),
            nn.GELU(),
            nn.Linear(int(num_tok * mult), int(num_tok * mult)),
            nn.LayerNorm(int(num_tok * mult)),
            nn.GELU(),
            nn.Linear(int(num_tok * mult), num_tok),
            nn.LayerNorm(num_tok),
            
        )
        
        self.net_sen = nn.Sequential(
            nn.Linear(dim, int(dim * mult)),
            nn.LayerNorm(int(dim * mult)),
            nn.GELU(),
            nn.Linear(int(dim * mult), int(dim * mult)),
            nn.LayerNorm(int(dim * mult)),
            nn.GELU(),
            nn.Linear(int(dim * mult), dim_out),
            #nn.GELU(),
            nn.LayerNorm(dim_out)
        )

    def forward(self, x):
        if self.dim_in == self.dim_out:
            indentity_0 = x
            x = self.net_sen(x)
            x += indentity_0
            x = x.transpose(1,2)

            indentity_1 = x
            x = self.net_tok(x)
            x += indentity_1
            x = x.transpose(1,2)
        else:
#             indentity_0 = x
            x = self.net_sen(x)
#             x += indentity_0
            x = x.transpose(1,2)

            x = self.net_tok(x)
            x = x.transpose(1,2)
        return x
    
class Translator_clip(nn.Module):
    def __init__(self, num_tok, dim, dim_out, mult=2, depth=5):
        super().__init__()
        
#         self.head = translator_base(num_tok, dim, dim, mult=2)
        
        self.blocks = nn.ModuleList(
            [translator_base(num_tok, dim, dim, mult=2)
                for d in range(depth)]
        )
        self.gelu = nn.GELU()

        self.tail = translator_clip(num_tok, dim, dim_out, mult=2)
        
    def forward(self, x):
        
#         x = self.head(x)
        
        for block in self.blocks:
            x = block(x) + x
            x = self.gelu(x)
        
        x = self.tail(x)
        return x

class translator_tok_dim(nn.Module):
    def __init__(self, num_tok, num_tok_out, dim, dim_out, mult=2, last_ln = True):
        super().__init__()
        
        self.dim_in = dim
        self.dim_out = dim_out
        
        self.tok_in = num_tok
        self.tok_out = num_tok_out
        
        if last_ln == True:
            self.net_tok = nn.Sequential(
                nn.Linear(self.tok_in, int(self.tok_in * mult)),
                nn.LayerNorm(int(self.tok_in * mult)),
                nn.GELU(),
                nn.Linear(int(self.tok_in * mult), int(self.tok_in * mult)),
                nn.LayerNorm(int(self.tok_in * mult)),
                nn.GELU(),
                nn.Linear(int(self.tok_in * mult), self.tok_out),
                nn.LayerNorm(self.tok_out),

            )
        else:
            self.net_tok = nn.Sequential(
                nn.Linear(self.tok_in, int(self.tok_in * mult)),
                #nn.LayerNorm(int(self.tok_in * mult)),
                nn.GELU(),
                nn.Linear(int(self.tok_in * mult), int(self.tok_in * mult)),
                #nn.LayerNorm(int(self.tok_in * mult)),
                nn.GELU(),
                nn.Linear(int(self.tok_in * mult), self.tok_out),
                #nn.LayerNorm(self.tok_out),

            )
        if last_ln == True:
            self.net_sen = nn.Sequential(
                nn.Linear(self.dim_in, int(self.dim_in * mult)),
                nn.LayerNorm(int(self.dim_in * mult)),
                nn.GELU(),
                nn.Linear(int(self.dim_in * mult), int(self.dim_in * mult)),
                nn.LayerNorm(int(self.dim_in * mult)),
                nn.GELU(),
                nn.Linear(int(self.dim_in * mult), self.dim_out),
                nn.LayerNorm(self.dim_out)
            )
        else:
            self.net_sen = nn.Sequential(
                nn.Linear(self.dim_in, int(self.dim_in * mult)),
#                 nn.LayerNorm(int(self.dim_in * mult)),
                nn.GELU(),
                nn.Linear(int(self.dim_in * mult), int(self.dim_in * mult)),
#                 nn.LayerNorm(int(self.dim_in * mult)),
                nn.GELU(),
                nn.Linear(int(self.dim_in * mult), self.dim_out),
                nn.GELU(),
#                 nn.LayerNorm(self.dim_out)
            )

    def forward(self, x, residual=None ):
        x = self.net_sen(x)
        x = x.transpose(1,2)
        
        if not torch.is_tensor(residual):
            x = self.net_tok(x)
            x = x.transpose(1,2)
        else:
            residual = residual.transpose(1,2)
            x = x + residual
            x = self.net_tok(x)
            x = x.transpose(1,2)
        return x
    
class Translator_w_head(nn.Module):
    def __init__(self, num_tok, num_tok_out, dim, dim_out, mult=2, depth=5):
        super().__init__()
        
        self.trans = translator_tok_dim(num_tok, num_tok_out, dim, dim_out, mult=mult, last_ln=True)
        self.tail_1 = translator_tok_dim(num_tok_out, num_tok_out, dim_out, dim_out, mult=mult, last_ln=False)
        self.tail_2 = translator_tok_dim(num_tok_out, num_tok_out, dim_out, dim_out, mult=mult, last_ln=False)
        
        self.in_blocks = nn.ModuleList(
            [translator_base(num_tok, dim, dim, mult=mult)
                for d in range(3)]
        )
        
        self.out_blocks = nn.ModuleList(
            [translator_base(num_tok_out, dim_out, dim_out, mult=mult)
                for d in range(depth - 3)]
        )
        
        self.gelu = nn.GELU()
#         self.tail = translator_clip(num_tok_out, dim, dim_out, mult=2)
        
    def forward(self, x):
#         x = self.head(x)
        
        for block in self.in_blocks:
            x = block(x) + x
            x = self.gelu(x)
            
        x = self.trans(x)
        
        for block in self.out_blocks:
            x = block(x) + x
            x = self.gelu(x)
        
        x = self.tail_2(self.tail_1(x) + x)
        
        return x