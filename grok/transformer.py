#!/usr/bin/env python
from argparse import ArgumentParser, Namespace
from typing import Tuple, List, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import cos, sin, sqrt
from torch import tensor, Tensor
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl

from argparse import ArgumentParser

class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, layer_dim, vocab_len):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        output_dim = vocab_len * 6
        self.layer_dim = layer_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.linear_init = nn.Linear(1, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.vocab_len = vocab_len

    def forward(self, x, h0=None, c0=None, save_activations: bool = False,):
        # If hidden and cell states are not provided, initialize them as zeros
        # Forward pass through LSTM
        x = x.reshape(x.size(0), x.size(1), 1)
        x = x.float()
        x = self.linear_init(x)
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Selecting the last output
        out = out.reshape(out.size(0), 6, self.vocab_len)
        return out, None, None

class MLPlarge(torch.nn.Module):

    def __init__(self,num_i,num_h,vocab_len):
        super(MLPlarge,self).__init__()

        self.linear1=torch.nn.Linear(num_i,num_h)
        self.relu=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(num_h,num_h)
        self.relu2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(num_h,num_h)
        self.relu3=torch.nn.ReLU()
        self.linear4=torch.nn.Linear(num_h,num_h)
        self.relu4=torch.nn.ReLU()
        self.linear5=torch.nn.Linear(num_h,num_h)
        self.relu5=torch.nn.ReLU()
        self.linear6=torch.nn.Linear(num_h,num_h)
        self.relu6=torch.nn.ReLU()
        self.linear7=torch.nn.Linear(num_h,num_h)
        self.relu7=torch.nn.ReLU()
        self.linear8=torch.nn.Linear(num_h,num_h)
        self.relu8=torch.nn.ReLU()
        self.linearo=torch.nn.Linear(num_h,vocab_len*6)
        self.vocab_len = vocab_len

    def forward(self, x, save_activations: bool = False,):
        # Make sure sampling inputs are on the correct device
        x = x.to(self.linear2.weight.device)
        x = x.float()
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        x = self.linear6(x)
        x = self.relu6(x)
        x = self.linear7(x)
        x = self.relu7(x)
        x = self.linear8(x)
        x = self.relu8(x)
        x = self.linearo(x)
        nbz = x.shape[0]
        x = x.reshape(nbz, 6, self.vocab_len)
        return x, None, None

class MLPmedium(torch.nn.Module):

    def __init__(self,num_i,num_h,vocab_len):
        super(MLPmedium,self).__init__()

        self.linear1=torch.nn.Linear(num_i,num_h)
        self.relu=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(num_h,num_h)
        self.relu2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(num_h,num_h)
        self.relu3=torch.nn.ReLU()
        self.linear4=torch.nn.Linear(num_h,num_h)
        self.relu4=torch.nn.ReLU()
        self.linear5=torch.nn.Linear(num_h,num_h)
        self.relu5=torch.nn.ReLU()
        self.linearo=torch.nn.Linear(num_h,vocab_len*6)
        self.vocab_len = vocab_len

    def forward(self, x, save_activations: bool = False,):
        # Make sure sampling inputs are on the correct device
        x = x.to(self.linear2.weight.device)
        x = x.float()
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        x = self.linearo(x)
        nbz = x.shape[0]
        x = x.reshape(nbz, 6, self.vocab_len)
        return x, None, None

class MLPsmall(torch.nn.Module):

    def __init__(self,num_i,num_h,vocab_len):
        super(MLPsmall,self).__init__()

        self.linear1=torch.nn.Linear(num_i,num_h)
        self.relu=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(num_h,num_h)
        self.relu2=torch.nn.ReLU()
        self.linearo=torch.nn.Linear(num_h,vocab_len*6)
        self.vocab_len = vocab_len

    def forward(self, x, save_activations: bool = False,):
        # Make sure sampling inputs are on the correct device
        x = x.to(self.linear2.weight.device)
        x = x.float()
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linearo(x)
        nbz = x.shape[0]
        x = x.reshape(nbz, 6, self.vocab_len)
        return x, None, None

class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise")
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight_noise > 0 and self.training:
            bias = self.bias if self.bias is None else self.bias + torch.randn_like(self.bias) * self.weight_noise
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
            # weight = self.weight * torch.exp(torch.randn_like(self.weight) * self.weight_noise)
        else:
            bias = self.bias
            weight = self.weight
            
        return F.linear(
            input,
            weight,
            bias,
        )

class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise")
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight_noise > 0 and self.training:
            bias = self.bias if self.bias is None else self.bias + torch.randn_like(self.bias) * self.weight_noise
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
            # weight = self.weight * torch.exp(torch.randn_like(self.weight) * self.weight_noise)
        else:
            bias = self.bias
            weight = self.weight
        return F.layer_norm(
            input,
            self.normalized_shape,
            weight,
            bias,
            self.eps,
        )


class Embedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        self.weight_noise = kwargs.pop("weight_noise")
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight_noise > 0 and self.training:
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
            # weight = self.weight * torch.exp(torch.randn_like(self.weight) * self.weight_noise)
        else:
            weight = self.weight
        return F.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_key: int, weight_noise: float) -> None:

        super().__init__()

        self.d_key = d_key

        # head projections
        self.Wq = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)
        self.Wk = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)
        self.Wv = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Union[Tensor, None] = None,
        save_activations: bool = False,
    ) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:

        # project queries, keys, values
        queries = self.Wq(queries)
        keys = self.Wk(keys)
        values = self.Wv(values)

        # calculate compatibility function
        attn = torch.matmul(queries, torch.transpose(keys, -2, -1))
        attn = attn / sqrt(self.d_key)

        # Filter out attention to future positions
        if mask is not None:
            attn.masked_fill_(mask == 0, float("-inf"))

        # softmax
        attn = self.softmax(attn)

        # sum the weighted value vectors
        result: Tensor = torch.matmul(attn, values)  # shape = (max_context_len, d_key)
        if save_activations:
            leaf_attn = attn.clone().detach()  # type: ignore
            leaf_values = values.clone().detach()  # type: ignore
        else:
            leaf_attn = None  # type: ignore
            leaf_values = None  # type: ignore

        return result, leaf_attn, leaf_values


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, weight_noise: float = 0.0) -> None:
        super().__init__()
        d_key = int(d_model / heads)

        attn_heads = [
            AttentionHead(d_model, d_key, weight_noise=weight_noise)
            for _ in range(heads)
        ]
        self.attn_heads = nn.ModuleList(attn_heads)
        self.Wo = Linear(d_model, d_model, bias=False, weight_noise=weight_noise)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Tensor = None,
        save_activations=False,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:

        head_outputs = [
            h(
                queries=queries,
                keys=keys,
                values=values,
                mask=mask,
                save_activations=save_activations,
            )
            for h in self.attn_heads
        ]
        head_results = [output[0] for output in head_outputs]

        if save_activations:
            layer_attns = list([output[1] for output in head_outputs])
            layer_values = list([output[2] for output in head_outputs])
        else:
            layer_attns = []
            layer_values = []

        multihead_result = torch.cat(head_results, dim=-1)
        multihead_result = self.Wo(multihead_result)
        return multihead_result, layer_attns, layer_values


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        multiplier: int = 4,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()

        d_ff = int(multiplier * d_model)

        non_linearities = {"relu": nn.ReLU, "gelu": nn.GELU}

        self.ffn = nn.Sequential(
            Linear(d_model, d_ff, bias=False, weight_noise=weight_noise),
            non_linearities[non_linearity](),
            Linear(d_ff, d_model, bias=False, weight_noise=weight_noise),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        dropout: float,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, heads, weight_noise=weight_noise)
        # self.self_attn_drop = nn.Dropout(p=dropout)
        self.self_attn_norm = LayerNorm(d_model, weight_noise=weight_noise)

        self.ffn = FFN(d_model, non_linearity=non_linearity, weight_noise=weight_noise)
        self.ffn_drop = nn.Dropout(p=dropout)
        self.ffn_norm = LayerNorm(d_model, weight_noise=weight_noise)

    def forward(
        self,
        x: Tensor,
        self_attn_mask: Tensor = None,
        save_activations: bool = False,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        a1, layer_attns, layer_values = self.self_attn(
            x, x, x, self_attn_mask, save_activations
        )
        # a1 = self.self_attn_drop(a1)
        a1 = self.self_attn_norm(x + a1)

        a2 = self.ffn(a1)
        a2 = self.ffn_drop(a2)
        a2 = self.ffn_norm(a1 + a2)

        return a2, layer_attns, layer_values


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        num_blocks: int,
        dropout: float,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model, heads, dropout, non_linearity, weight_noise=weight_noise
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        x: Tensor,
        self_attn_mask: Tensor = None,
        save_activations=False,
    ) -> Tuple[Tensor, List[List[Tensor]], List[List[Tensor]]]:

        a = x
        attentions = []
        values = []
        for block in self.blocks:
            a, layer_attentions, layer_values = block(
                a, self_attn_mask, save_activations=save_activations
            )
            if save_activations:
                attentions.append(layer_attentions)
                values.append(layer_values)
        return a, attentions, values


class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        n_heads: int = 4,
        d_model: int = 256,
        dropout: float = 0.1,
        max_context_len: int = 1024,
        vocab_len: int = 2000,
        non_linearity: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.max_context_len = max_context_len
        self.non_linearity = non_linearity

        self.vocab_len = vocab_len

        self.embedding = Embedding(vocab_len, d_model, weight_noise=weight_noise)  # type: ignore
        self.register_buffer(
            "position_encoding", self._position_encoding(max_context_len, d_model)
        )
        self.register_buffer("self_attn_mask", self.make_mask(max_context_len))

        self.decoder = Decoder(
            d_model,
            n_heads,
            n_layers,
            dropout,
            self.non_linearity,
            weight_noise=weight_noise,
        )

        self.linear = Linear(d_model, vocab_len, bias=False, weight_noise=weight_noise)

    @staticmethod
    def make_mask(context_len: int) -> Tensor:
        return torch.ones([context_len, context_len]).tril()

    @classmethod
    def _position_encoding(cls, context_len: int, d_model: int) -> Tensor:
        rows = [
            tensor(
                [
                    sin(pos / (10000 ** (i / d_model)))
                    if i % 2 == 0
                    else cos(pos / (10000 ** ((i - 1) / d_model)))
                    for i in range(d_model)
                ]
            )
            for pos in range(context_len)
        ]
        stack = torch.stack(rows, dim=1)

        return stack.T  # type: ignore

    def embed(self, indices: Tensor) -> Tensor:
        context_len = indices.shape[-1]
        pe = self.position_encoding[:context_len, :]  # type: ignore

        embedded = self.embedding(indices)

        return pe + embedded

    def forward(
        self,
        x: Tensor,
        pos: int = None,
        save_activations: bool = False,
    ) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        """parameters:
        x:  (rank-1 tensor) vocab indices of decoder input token
                     sequence"""

        # Make sure sampling inputs are on the correct device
        x = x.to(self.embedding.weight.device)

        # make_attention mask
        this_max_context_len = x.shape[-1]
        self_attn_mask = self.self_attn_mask[  # type: ignore
            :this_max_context_len, :this_max_context_len
        ]

        # Decode
        x = self.embed(x)
        decoded, attentions, values = self.decoder(
            x, self_attn_mask, save_activations=save_activations
        )

        # Return predictions for specific token
        if pos is not None:
            decoded = decoded[:, pos, :]

        y_hat = self.linear(decoded)
        return y_hat, attentions, values
