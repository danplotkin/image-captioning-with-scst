"""CPTR-inspired captioning model with independently owned submodules."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from scst_captioner.config import ModelConfig
from scst_captioner.decoder import CaptionDecoder, DecodeCache


class VisionBackbone(nn.Module):
    """Thin wrapper that keeps a frozen vision encoder deterministic."""

    def __init__(self, backbone: nn.Module, *, frozen: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.frozen = frozen
        if frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> VisionBackbone:
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            with torch.no_grad():
                output = self.backbone(pixel_values)
        else:
            output = self.backbone(pixel_values)
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        raise TypeError("Vision encoder must return a tensor or an object with last_hidden_state")


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, pad_token_id: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.embedding_dim = embedding_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids) * math.sqrt(self.embedding_dim)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int, dropout: float) -> None:
        super().__init__()
        positions = torch.arange(max_seq_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_seq_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", encoding.unsqueeze(0), persistent=True)

    def forward(self, embeddings: torch.Tensor, offset: int = 0) -> torch.Tensor:
        if offset < 0 or offset + embeddings.size(1) > self.pe.size(1):
            raise ValueError(
                f"Sequence length {embeddings.size(1)} exceeds positional capacity {self.pe.size(1)}"
            )
        return self.dropout(embeddings + self.pe[:, offset : offset + embeddings.size(1)])


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_seq_length: int,
        pad_token_id: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, embedding_dim, pad_token_id)
        self.positional_encoding = SinusoidalPositionalEncoding(
            embedding_dim, max_seq_length, dropout
        )

    def forward(self, token_ids: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return self.positional_encoding(self.embedding(token_ids), offset=offset)


class CPTRCaptioner(nn.Module):
    """A CPTR-inspired ViT encoder plus autoregressive Transformer decoder.

    The field names intentionally match the legacy notebook state dict. Unlike
    the notebook, every instance constructs and owns its decoder and encoder.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        vocab_size: int,
        pad_token_id: int,
        backbone: nn.Module | None = None,
        encoder_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if backbone is None:
            from transformers import ViTModel

            backbone = ViTModel.from_pretrained(
                config.encoder_name,
                revision=config.encoder_revision,
                add_pooling_layer=False,
            )
        if encoder_hidden_size is None:
            encoder_hidden_size = getattr(getattr(backbone, "config", None), "hidden_size", None)
        if encoder_hidden_size is None:
            raise ValueError("encoder_hidden_size is required for an injected backbone")

        self.config = config
        self.pad_token_id = pad_token_id
        self.max_length = config.max_seq_length
        self.backbone = VisionBackbone(backbone, frozen=config.freeze_encoder)
        self.positional_embedding = PositionalEmbedding(
            vocab_size,
            config.d_model,
            config.max_seq_length,
            pad_token_id,
            config.dropout,
        )
        self.decoder = CaptionDecoder(config, encoder_hidden_size)
        self.fc = nn.Linear(config.d_model, vocab_size)

    @property
    def encoder_is_frozen(self) -> bool:
        return not any(parameter.requires_grad for parameter in self.backbone.parameters())

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values)

    def decode(self, token_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        target = self.positional_embedding(token_ids)
        allowed = ~(
            self.compute_causal_mask(token_ids)[None, None, :, :]
            | self.compute_padding_mask(token_ids)[:, None, None, :]
        )
        output, _ = self.decoder(target, memory, allowed)
        return self.fc(output)

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decode(input_ids, self.encode_images(pixel_values))

    def predict_next_token(self, token_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.decode(token_ids, memory)[:, -1, :]

    def init_decode_cache(self, memory: torch.Tensor) -> DecodeCache:
        """Project image K/V once per layer, retaining gradients for SCST."""
        if self.training:
            raise ValueError("KV caching requires eval mode; train-mode dropout uses full prefixes")
        layers = tuple(layer.init_cache(memory) for layer in self.decoder.layers)
        padding = torch.empty(memory.size(0), 0, dtype=torch.bool, device=memory.device)
        return DecodeCache(layers, padding)

    def decode_step(
        self, token_ids: torch.Tensor, cache: DecodeCache
    ) -> tuple[torch.Tensor, DecodeCache]:
        """Consume one token per image with the existing decoder weights.

        Eval mode disables dropout, not gradients. Concatenation (rather than
        detached or in-place cache writes) preserves the complete SCST graph.
        """
        if self.training:
            raise ValueError("KV caching requires eval mode")
        if token_ids.shape != (cache.padding_mask.size(0), 1):
            raise ValueError("decode_step expects [batch, 1] tokens")
        padding = torch.cat((cache.padding_mask, self.compute_padding_mask(token_ids)), dim=1)
        target = self.positional_embedding(token_ids, offset=cache.position)
        # The newest query may attend to every non-padding cached position.
        output, layers = self.decoder(target, None, ~padding[:, None, None, :], cache.layers)
        return self.fc(output[:, 0]), DecodeCache(layers, padding, cache.position + 1)

    def compute_padding_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.eq(self.pad_token_id)

    @staticmethod
    def compute_causal_mask(token_ids: torch.Tensor) -> torch.Tensor:
        length = token_ids.size(1)
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=token_ids.device), diagonal=1
        )


def build_model(config: ModelConfig, tokenizer: Any) -> CPTRCaptioner:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("The tokenizer must define pad_token_id")
    return CPTRCaptioner(
        config,
        vocab_size=len(tokenizer),
        pad_token_id=pad_token_id,
    )
