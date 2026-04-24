from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from seagull.nn.modules.module import Module
from seagull.nn.modules.utils.utils import set_jit_flags

set_jit_flags()


@torch.jit.script
def fused_gain_bias_dropout(
    input_normed: torch.Tensor,
    gain: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dropout_proba: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """
    Applies a gain multiplier, adds bias, and applies dropout to an input tensor.

    Parameters
    ----------
    input_normed : torch.Tensor
        Input tensor which has been RMS-normalized.
    gain : torch.Tensor
        Gain values to be multiplied element-wise by the input.
    bias : torch.Tensor
        Optional bias term to be added to input.
    dropout_proba : float
        Dropout probability to be used for dropout computation.
    training : bool
        Applies dropout if true.

    Returns
    -------
    torch.Tensor
        Input tensor with gain, bias, and dropout applied.
    """
    y = gain * input_normed
    y = y + bias if bias is not None else y
    return F.dropout(y, p=dropout_proba, training=training) if dropout_proba > 0.0 else y


class RMSNorm(Module):
    __constants__ = ["dimension", "p", "eps", "bias", "dropout_proba"]

    def __init__(
        self, dimension: int, p: float = -1, eps: float = 1e-8, bias: bool = False, dropout_proba: float = 0.0
    ):
        """
        An RMS normalization layer, see https://arxiv.org/pdf/1910.07467.pdf.

        Parameters
        ----------
        dimension : int
            Dimension of RMS normalization; must be equal to dimension of input along first axis.
        p : float
            The order of the matrix norm.
        eps : float
            Small value added to denominator during normalization to avoid division by zero.
        bias : bool
            If set to `False`, the layer will not learn an additive bias.
        dropout_proba : float
            Dropout probability to be used for dropout computation.
        """
        super().__init__()

        self.dimension = dimension
        self.p = p
        self.eps = eps
        self.dropout_proba = dropout_proba

        self.gain = nn.Parameter(torch.ones(dimension), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(dimension), requires_grad=True)
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Computes forward pass through RMS normalization layer.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor to RMS-normalize.

        Returns
        -------
        torch.Tensor
            Input tensor which has been RMS-normalized.
        """
        assert input.shape[-1] == self.dimension, f"RMSNorm.dimension: {self.dimension} must equal input.shape[-1]"
        if self.p < 0.0 or self.p > 1.0:
            norm = input.norm(p=2, dim=-1, keepdim=True)
            rms_input = norm / (self.dimension**0.5)
        else:
            partial_size = int(self.dimension * self.p)
            partial_input, _ = torch.split(input, [partial_size, self.dimension - partial_size], dim=-1)
            norm = partial_input.norm(p=2, dim=-1, keepdim=True)
            rms_input = norm / (partial_size**0.5)
        input_normed = input / (rms_input + self.eps)
        return fused_gain_bias_dropout(
            input_normed=input_normed,
            gain=self.gain,
            bias=self.bias,
            dropout_proba=self.dropout_proba,
            training=self.training,
        )
