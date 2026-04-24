from typing import Optional, Union

import torch
from torch import nn

from seagull.nn.modules.module import Module
from seagull.nn.modules.rms_norm import RMSNorm


class Embedding(Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 768,
        max_positions: int = 512,
        padding_idx: Optional[int] = None,
        use_rope: bool = True,
        layer_norm_type: Optional[str] = None,
        dropout_proba: float = 0.1,
    ):
        """
        Initializes token embeddings (one for each token in the vocabulary) using the given parameters.

        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary; ``vocab_size`` total embeddings are initialized using :py:class:`~nn.Embedding`.
        embedding_dim : int
            The required embedding dimension.
        max_positions : int
            The number of max positions of the embedding
        padding_idx : int
            The token index corresponding to padding tokens; the padded token embedding is a vector of all zeros.
        use_rope : bool
            If set to False, the model will not use rotary positional embedding
        layer_norm_type : Optional[str]
            If set to "rms", the model will use root mean square layer normalization.
            Otherwise, it applies Layer Normalization over a mini-batch of inputs as described in
            the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.
        dropout_proba : float
            During training, randomly zeroes some of the elements of the input tensor with probability `dropout_proba`.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self._dropout_proba = dropout_proba

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx
        )
        self.use_rope = use_rope
        if not use_rope:
            self.position_embedding = nn.Embedding(num_embeddings=max_positions, embedding_dim=embedding_dim)
        self.apply_layer_norm = layer_norm_type is not None
        if layer_norm_type is not None:
            self.layer_norm = self._get_layer_norm(layer_norm_type=layer_norm_type)

    def _get_layer_norm(self, layer_norm_type: str) -> Union[Module, nn.Module]:
        if layer_norm_type.startswith("rms"):
            return RMSNorm(dimension=self.embedding_dim, eps=1e-8, dropout_proba=self._dropout_proba)
        else:
            return nn.Sequential(
                nn.LayerNorm(normalized_shape=self.embedding_dim, eps=1e-8), nn.Dropout(p=self._dropout_proba)
            )

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward passes the given embedding through the MHA model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tensor representing the input ids to the embedding.
        position_ids : Optional[torch.Tensor]
            Optional position IDs to use for positional embeddings.

        Returns
        -------
        torch.Tensor
            The tensor of embedded inputs.
        """
        # TODO-3
        # 0. get the batch size and sequence length
        batch_size, seq_len = input_ids.shape

        # 1. pass the input_ids through the self.token_embedding layer
        token_embedding = self.token_embedding(input_ids)

        # 2. position embedding
        if self.use_rope is False: # use learnable embedding
          if position_ids is not None:
            position_embedding = self.position_embedding(position_ids)
          else:
            position_ids_single_batch = torch.arange(seq_len).unsqueeze(0) # (seq_len, )->(1, seq_len)
            position_ids = position_ids_single_batch.expand(batch_size, -1) # (1, seq_len)->(batch_size, seq_len)
            position_embedding = self.position_embedding(position_ids)

          # 3. add the positional embeddings and token embeddfings
          final_embeddings = token_embedding+position_embedding
        else: 
          final_embeddings = token_embedding

        # 3. do layer normalization
        if self.apply_layer_norm:
          final_embeddings = self.layer_norm(final_embeddings)
        return final_embeddings

