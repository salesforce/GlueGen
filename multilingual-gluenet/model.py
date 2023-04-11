import pdb

import torch 
import torch.nn as nn

from torch.autograd import Function
from util.x_transformer import AttentionLayers

def count_parameters(model, grad_flag=False):
    if grad_flag:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

class GradReverse(Function):     
    @staticmethod
    def forward(self, x, lambd):
        self.lambd = lambd
        return x.view_as(x)
    
    @staticmethod
    def backward(self, grad_output):
        output = grad_output * -self.lambd
        return output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class Discriminator(nn.Module):
    def __init__(self, dim, dim_out=None, mult=0.25, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.BatchNorm1d(inner_dim),
            nn.ReLU(True),
            nn.Linear(inner_dim, dim_out),
            nn.LogSoftmax(dim=1)
        )
        

    def forward(self, x, eta=0.1, reverse=True):
        if reverse:
            x = grad_reverse(x, eta)
        return self.net(x)

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
    
class translator_base_rev(nn.Module):
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
            x = x.transpose(1,2)
            indentity_1 = x
            x = self.net_tok(x)
            x += indentity_1
            
            x = x.transpose(1,2)
            indentity_0 = x
            x = self.net_sen(x)
            x += indentity_0
        else:
            x = x.transpose(1,2)
            x = self.net_sen(x)
            
            x = x.transpose(1,2)
            x = self.net_tok(x)
        return x