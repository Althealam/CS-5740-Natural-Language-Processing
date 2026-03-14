import logging
from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from masking.nn.module import Module


class RNN(Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ):
        """
        A multi-layer recurrent neural network with ReLU, tanh, or PReLU nonlinearity to an input sequence.

        Parameters
        ----------
        embedding_dim : int
            Number of dimensions of an input embedding.
        hidden_dim : int
            Number of dimensions of the hidden layer(s).
        output_dim : int
            Number of dimensions for the output layer.
        num_layers : int, default: 1
            Number of layers in the multi-layer RNN model.
        bias : bool, default: True
            If set to False, the input-to-hidden and hidden-to-hidden transformations will not include bias. Note: the
            hidden-to-output transformation remains unaffected by ``bias``.
        nonlinearity : {"tanh", "relu", "prelu"}, default: "tanh"
            Name of the nonlinearity to be applied during the forward pass.
        """
        super().__init__()

        assert num_layers > 0

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        logging.info(f"no shared weights across layers")

        nonlinearity_dict = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "prelu": nn.PReLU()}
        if nonlinearity not in nonlinearity_dict:
            raise ValueError(f"{nonlinearity} not supported, choose one of: [tanh, relu, prelu]")
        self.nonlinear = nonlinearity_dict[nonlinearity]

        # TODO-5-1
        self.W = nn.ModuleList()
        self.U = nn.ModuleList()

        for i in range(num_layers):
          if i==0:
            self.W.append(nn.Linear(embedding_dim, hidden_dim, bias=bias))
          else:
            self.W.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
          
          self.U.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        
        self.V = nn.Linear(hidden_dim, output_dim)

        self.apply(self.init_weights)

    def _initial_hidden_states(
        self, batch_size: int, init_zeros: bool = False, device: torch.device = torch.device("cpu")
    ) -> List[torch.Tensor]:
        """
        Returns a list of :py:attr:`~masking.nn.models.rnn.RNN.num_layers` number of initial hidden states.

        Parameters
        ----------
        batch_size : int
            The processing batch size.
        init_zeros : bool, default: False
            If False, the hidden states will be initialized using the normal distribution; otherwise, they will be
            initialized as all zeros.
        device: torch.device
            The device to be used in storing the initialized tensors.

        Returns
        -------
        List[torch.Tensor]
            List holding tensors of initialized initial hidden states of shape `(num_layers, batch_size, hidden_dim)`.
        """
        if init_zeros:
            hidden_states = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        else:
            hidden_states = nn.init.xavier_normal_(
                torch.empty(self.num_layers, batch_size, self.hidden_dim, device=device)
            )
        return list(map(torch.squeeze, hidden_states))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass of the given input embeddings through the RNN model.

        Parameters
        ----------
        embeddings : torch.Tensor
            Input tensor of embeddings of shape ``(batch_size, max_length, embedding_dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor resulting from forward pass of shape ``(batch_size, max_length, output_dim)``.
        """
        # TODO-5-2
        batch_size, max_length, _ = embeddings.shape
        device = embeddings.device

        hidden_states = self._initial_hidden_states(batch_size, device = device)

        outputs = []

        for t in range(max_length):
          x = embeddings[:, t, :]
          for layer in range(self.num_layers):
            prev_h = hidden_states[layer]
            h = self.nonlinear(
              self.W[layer](x)+self.U[layer](prev_h)
            )
            hidden_states[layer] = h
            x = h # 下一层输入
          
          y = self.V(h) # (batch_size, output_dim)

          outputs.append(y)
        outputs = torch.stack(outputs, dim = 1)
        return outputs