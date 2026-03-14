from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch import nn


def compute_loss(loss_fn: Callable, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute loss given a loss function, logits, and labels. Note: the logits and labels must be on the same device.

    Parameters
    ----------
    loss_fn : Callable
        The loss function (criterion).
    preds : torch.Tensor
        The prediction IDs output by the model.
    labels : torch.Tensor
        The target label IDs.

    Returns
    -------
    torch.Tensor
        The computed loss, returned as a tensor on the same device as logits and labels.
    """
    # preds: (batch_size, max_length, output_dim)
    # labels: (batch_size, max_length)
    assert len(preds.shape) >= 2 and len(labels.shape) >= 1
    return loss_fn(preds.view(-1, preds.shape[-1]), labels.view(-1))


def compute_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    labels_ignore_idx: Optional[int] = None,
    other_mask_tag_idx: Optional[int] = None,
    average: str = "weighted",
):
    """
    Compute metrics given predictions and labels.

    Parameters
    ----------
    preds : torch.Tensor
        The prediction IDs output by the model.
    labels : torch.Tensor
        The target label IDs.
    padding_mask : Optional[torch.Tensor], default: None
        The padding mask.
    labels_ignore_idx : Optional[int], default: None
        If ``padding_mask`` is not provided, the ``padding_mask`` can be auto-inferred from ``labels_ignore_idx``, if
        ``labels_ignore_idx`` is provided. Corresponds to :py:const:`~masking.data_processing.constants.PAD_MASK_TAG` in
        :py:const:`~masking.data_processing.constants.MASK_ENCODING_MAP`.
    other_mask_tag_idx : Optional[int]
        The index of the "O" tag; if provided, the performance computation ignores the "O" tags. Corresponds to ``O``
        in :py:const:`~masking.data_processing.constants.MASK_ENCODING_MAP`.
    average : str
        The averaging method for aggregating scores.

    Returns
    -------
    Dict[str, float]
        A dictionary of metrics, includes loss, precision, recall, accuracy, F1.
    """
    # preds: (batch_size, max_length, output_dim)
    # labels, padding_mask: (batch_size, max_length)
    assert len(preds.shape) >= 2 and len(labels.shape) >= 1
    assert labels_ignore_idx is not None or padding_mask is not None, "labels_ignore_idx or padding_mask must be given"

    preds = preds.view(-1, preds.shape[-1]).argmax(dim=-1)
    labels = labels.view(-1)

    if padding_mask is not None and (padding_mask.dtype == torch.long or padding_mask.dtype == torch.int):
        padding_mask = torch.BoolTensor(padding_mask == 1)
    mask = (~padding_mask.view(-1)) if padding_mask is not None else (labels != labels_ignore_idx)
    y_true, y_pred = labels[mask].cpu().numpy(), preds[mask].cpu().numpy()

    if other_mask_tag_idx is not None:
        mask = mask & (labels != other_mask_tag_idx)
    y_true, y_pred = labels[mask].cpu().numpy(), preds[mask].cpu().numpy()

    metrics = {
        "precision": precision_score(y_true=y_true, y_pred=y_pred, zero_division=0, average=average),
        "recall": recall_score(y_true=y_true, y_pred=y_pred, zero_division=0, average=average),
        "f1": f1_score(y_true=y_true, y_pred=y_pred, zero_division=0, average=average),
        "accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
    }
    return metrics


if __name__ == "__main__":

    @dataclass
    class TestConfig:
        loss_fn = nn.CrossEntropyLoss()

        batch_size = 6
        max_length = 10
        output_dim = 5

        pad_mas,_tag_idx = 9
        other_mask_tag_idx = 4

        average = "weighted"

    test_config = TestConfig()
    test_preds = torch.randn((test_config.batch_size, test_config.max_length, test_config.output_dim))
    test_labels = torch.randint(0, test_config.output_dim, (test_config.batch_size, test_config.max_length))
    test_padding_mask = torch.where(torch.BoolTensor(test_labels == test_config.pad_mask_tag_idx), 1, 0)

    test_loss = compute_loss(test_config.loss_fn, preds=test_preds, labels=test_labels)
    assert torch.isclose(
        test_loss, F.cross_entropy(input=test_preds.view(-1, test_config.output_dim), target=test_labels.view(-1))
    )

    test_metrics_with_padding_mask = compute_metrics(
        preds=test_preds,
        labels=test_labels,
        padding_mask=test_padding_mask,
        other_mask_tag_idx=test_config.other_mask_tag_idx,
        average=test_config.average,
    )
    test_metrics_with_ignore_idx = compute_metrics(
        preds=test_preds,
        labels=test_labels,
        labels_ignore_idx=test_config.pad_mask_tag_idx,
        other_mask_tag_idx=test_config.other_mask_tag_idx,
        average=test_config.average,
    )
    assert np.allclose(list(test_metrics_with_padding_mask.values()), list(test_metrics_with_ignore_idx.values()))
