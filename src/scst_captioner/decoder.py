"""Caption decoder with a shared forward path for full sequences and KV caching."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from scst_captioner.config import ModelConfig


@dataclass(frozen=True)
class LayerKVCache:
    memory_key: torch.Tensor
    memory_value: torch.Tensor
    self_key: torch.Tensor | None = None
    self_value: torch.Tensor | None = None


@dataclass(frozen=True)
class DecodeCache:
    """Per-rollout tensors; never stored in the model or detached from autograd."""

    layers: tuple[LayerKVCache, ...]
    padding_mask: torch.Tensor
    position: int = 0


def _heads(value: torch.Tensor, attention: nn.MultiheadAttention) -> torch.Tensor:
    return value.reshape(value.size(0), value.size(1), attention.num_heads, -1).transpose(1, 2)


def _attend(
    attention: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed: torch.Tensor | None = None,
) -> torch.Tensor:
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=allowed,
        dropout_p=attention.dropout if attention.training else 0.0,
    )
    # Match MultiheadAttention's output layout so seeded dropout stays equivalent.
    length, batch_size = query.size(2), query.size(0)
    output = output.permute(2, 0, 1, 3).reshape(length * batch_size, attention.embed_dim)
    return (
        attention.out_proj(output).reshape(length, batch_size, attention.embed_dim).transpose(0, 1)
    )


class CaptionDecoderLayer(nn.Module):
    """One decoder block; attention modules retain the checkpoint parameter names."""

    def __init__(self, config: ModelConfig, encoder_hidden_size: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.d_model, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.dropout,
            kdim=encoder_hidden_size,
            vdim=encoder_hidden_size,
            batch_first=True,
        )
        self.linear1 = nn.Linear(config.d_model, config.feedforward_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.feedforward_dim, config.d_model)
        self.norm_first = False
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.activation = {"relu": F.relu, "gelu": F.gelu}[config.activation]

    def init_cache(self, memory: torch.Tensor) -> LayerKVCache:
        """Project raw image features with this layer's own K/V weights."""
        attention = self.multihead_attn
        width = attention.embed_dim
        if attention.in_proj_weight is not None:
            key, value = F.linear(
                memory, attention.in_proj_weight[width:], attention.in_proj_bias[width:]
            ).chunk(2, dim=-1)
        else:
            key = F.linear(
                memory, attention.k_proj_weight, attention.in_proj_bias[width : 2 * width]
            )
            value = F.linear(memory, attention.v_proj_weight, attention.in_proj_bias[2 * width :])
        return LayerKVCache(_heads(key, attention), _heads(value, attention))

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor | None,
        allowed: torch.Tensor,
        cache: LayerKVCache | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        if cache is None:
            if memory is None:
                raise ValueError("Decoder requires image memory or a KV cache")
            cache = self.init_cache(memory)

        residual = target
        source = self.norm1(target) if self.norm_first else target
        attention = self.self_attn
        query, key, value = F.linear(
            source, attention.in_proj_weight, attention.in_proj_bias
        ).chunk(3, dim=-1)
        query, key, value = (_heads(item, attention) for item in (query, key, value))
        if cache.self_key is not None:
            # Concatenation preserves gradients through earlier caption tokens.
            key = torch.cat((cache.self_key, key), dim=2)
            value = torch.cat((cache.self_value, value), dim=2)
        target = residual + self.dropout1(_attend(attention, query, key, value, allowed))
        if not self.norm_first:
            target = self.norm1(target)

        residual = target
        source = self.norm2(target) if self.norm_first else target
        attention = self.multihead_attn
        width = attention.embed_dim
        query_weight = (
            attention.in_proj_weight[:width]
            if attention.in_proj_weight is not None
            else attention.q_proj_weight
        )
        query = _heads(F.linear(source, query_weight, attention.in_proj_bias[:width]), attention)
        target = residual + self.dropout2(
            _attend(attention, query, cache.memory_key, cache.memory_value)
        )
        if not self.norm_first:
            target = self.norm2(target)

        residual = target
        source = self.norm3(target) if self.norm_first else target
        target = residual + self.dropout3(
            self.linear2(self.dropout(self.activation(self.linear1(source))))
        )
        if not self.norm_first:
            target = self.norm3(target)
        return target, LayerKVCache(cache.memory_key, cache.memory_value, key, value)


class CaptionDecoder(nn.Module):
    """Stack decoder layers using the same computation with or without a cache."""

    def __init__(self, config: ModelConfig, encoder_hidden_size: int) -> None:
        super().__init__()
        layer = CaptionDecoderLayer(config, encoder_hidden_size)
        # Match TransformerDecoder's initialization while owning distinct parameters.
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(config.num_decoder_layers))
        self.norm: nn.Module | None = None

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor | None,
        allowed: torch.Tensor,
        cache: tuple[LayerKVCache, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        previous = cache if cache is not None else (None,) * len(self.layers)
        layers = []
        for layer, layer_cache in zip(self.layers, previous, strict=True):
            target, layer_cache = layer(target, memory, allowed, layer_cache)
            layers.append(layer_cache)
        if self.norm is not None:
            target = self.norm(target)
        return target, tuple(layers)
