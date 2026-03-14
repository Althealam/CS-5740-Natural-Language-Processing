from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple, Union
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import nn

from masking.data_processing.constants import MASK_DECODING_MAP, MASK_ENCODING_MAP, PAD_TOKEN, UNK_TOKEN
from masking.data_processing.tokenizer import Tokenizer
from masking.models.mask_predictor import MaskPredictor
from masking.nn.module import Module
from masking.utils.utils import colored


def _print_convention():
    """Print the color convention used in activation visualization plots."""
    print(
        f"{colored('Convention', attrs=['underline'])}: [when labels are provided,] "
        f"{colored('dark green', color='dark green', attrs=['bold'])} text indicates that the true label and "
        f"predicted \nlabel match (e.g., pred = 'USERNAME', true = 'USERNAME'); "
        f"{colored('red', color='red')} text indicates that "
        f"predicted and true labels mismatch (e.g., pred = \n'USERNAME', true = 'IDCARD').\n"
    )


def _get_pred_tags_and_unk_tokens_from_text(
    tokenizer: Tokenizer,
    model: Module,
    text: List,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[str], np.ndarray]:
    """
    Get the tag predictions (using a model) and unknown tokens from the input list of tokens.

    Parameters
    ----------
    tokenizer : Tokenizer
        Tokenizer.
    model : Module
        Model.
    text : List
        Text.
    device : torch.device
        Device.

    Returns
    -------
    List[str]
        Predicted tags.
    np.ndarray
        Unknown tokens.
    """
    input_ids = tokenizer.tokenize(input_seq=text, max_length=None)["input_ids"].unsqueeze(0)
    preds = model(input_ids.to(device)).squeeze().argmax(-1).tolist()
    pred_tags = [MASK_DECODING_MAP[_] for _ in preds]
    unk_tokens = np.array(text)[np.where(input_ids[0] == tokenizer.token2id[tokenizer.unk_token], True, False)]
    return pred_tags, unk_tokens


def visualize_activations(
    tokenizer: Tokenizer,
    model: Module,
    module: Union[Module, nn.Module],
    text: List,
    prev_layer_module: Optional[Union[Module, nn.Module]] = None,
    labels: Optional[List] = None,
    nonlinearity: Optional[Callable] = None,
    device: torch.device = torch.device("cpu"),
    cbar: Optional[bool] = True,
    figsize: Tuple[int, int] = None,
    fontsize: int = 8,
):
    """
    A function to visualize activations by attaching forward hooks to the model.

    Parameters
    ----------
    tokenizer : Tokenizer
        Tokenizer.
    model : Module
        Model.
    module : Union[Module, nn.Module]
        Module.
    text : List
        Text.
    prev_layer_module : Optional[Union[Module, nn.Module]]
        Previous layer module.
    labels : Optional[List]
        Labels.
    nonlinearity : Optional[Callable]
        Nonlinearity.
    device : torch.device
        Device.
    cbar : Optional[bool]
        Whether to show colorbar.
    figsize : Tuple[int, int]
        Figure size: (width, height).
    fontsize : int
        Font size.
    """
    if nonlinearity is None:
        nonlinearity = lambda _outputs: _outputs
    _nonlinearity = nonlinearity if prev_layer_module is None else lambda _outputs: _outputs

    model.hooks["outputs"] = {"current_layer": [], "previous_layer": []}

    def get_activations_hook(layer_type, _module, _inputs, _outputs):
        model.hooks["outputs"][f"{layer_type}_layer"].append(_nonlinearity(_outputs).detach().cpu().squeeze())

    model.attach_hook(module=module, hook=partial(get_activations_hook, "current"), hook_type="forward")
    if prev_layer_module is not None:
        model.attach_hook(module=prev_layer_module, hook=partial(get_activations_hook, "previous"), hook_type="forward")

    pred_tags, unk_tokens = _get_pred_tags_and_unk_tokens_from_text(
        tokenizer=tokenizer, model=model, text=text, device=device
    )
    activations = torch.vstack(model.hooks["outputs"]["current_layer"]).squeeze()[-len(text) :]
    if prev_layer_module is not None:
        prev_layer_activations = torch.vstack(model.hooks["outputs"]["previous_layer"]).squeeze()[-len(text) :]
        activations = nonlinearity(activations + prev_layer_activations)
    activations = activations.numpy()
    model.detach_all_hooks()

    _print_convention()

    fig, ax1 = plt.subplots(1, figsize=(figsize if figsize is not None else (10, int(4.5 * (len(text) / 20)))))
    ax2 = ax1.twinx()
    plt_token_labels = [token if token not in unk_tokens else f"{token}:{UNK_TOKEN}" for token in text]
    sns.set(font_scale=0.6)
    sns.heatmap(
        activations,
        ax=ax1,
        yticklabels=plt_token_labels,
        cbar=cbar,
        cbar_kws=dict(use_gridspec=False, shrink=0.5, location="right", pad=0.1),
    )

    sns.heatmap(activations, ax=ax2, yticklabels=pred_tags, cbar=False)
    ax1.set_xlabel(f"output: {module}", fontsize=fontsize)
    ax1.set_ylabel("tokens", fontsize=fontsize)
    ax2.set_ylabel("predictions", fontsize=fontsize)
    ax1.set_yticklabels(plt_token_labels, fontsize=fontsize, rotation=0)
    ax2.set_yticklabels(pred_tags, fontsize=fontsize, rotation=0)
    if labels is not None:
        for ytick, pred_tag, label in zip(ax2.get_yticklabels(), pred_tags, labels):
            if pred_tag == label:
                if pred_tag != "O":
                    ytick.set_color("green")
                    ytick.set_fontweight("bold")
            elif pred_tag.split("-")[-1] == label.split("-")[-1]:
                ytick.set_color("green")
            elif pred_tag != label:
                ytick.set_color("red")
    ax1.tick_params(axis="x", labelsize=fontsize)
    plt.show()


def inspect_preds(
    tokenizer: Tokenizer,
    model: Module,
    text: List,
    labels: Optional[List] = None,
    device: torch.device = torch.device("cpu"),
) -> List[str]:
    """
    Inspect predictions at a token-level.

    Parameters
    ----------
    tokenizer : Tokenizer
        Tokenizer.
    model : Module
        Model.
    text : List
        Text.
    labels : Optional[List]
        Labels.
    device : torch.device
        Device.

    Returns
    -------
    List[str]
        Predicted tags.
    """
    model = model.to(device)

    pred_tags, unk_tokens = _get_pred_tags_and_unk_tokens_from_text(
        tokenizer=tokenizer, model=model, text=text, device=device
    )

    idx_col_max_len = max([len(str(idx)) for idx in range(len(text))] + [len("idx")])
    is_unk_col_man_len = len("is-unk?")
    token_col_max_len = max([len(token) for token in text] + [len("token")])
    tag_col_max_len = max([len(tag) for tag in pred_tags] + [len("pred")])

    _print_convention()
    print(
        f"{' '.ljust(idx_col_max_len)} "
        f"{'token'.ljust(token_col_max_len)}  "
        f"{'is-unk?'.ljust(is_unk_col_man_len)}  "
        f"{'pred'.ljust(tag_col_max_len)}" + (f"  {'true'.ljust(tag_col_max_len)}" if labels is not None else "")
    )
    print(
        f"{'-' * idx_col_max_len} "
        f"{'-' * token_col_max_len}  "
        f"{'-' * is_unk_col_man_len}  "
        f"{'-' * tag_col_max_len}" + (f"  {'-' * tag_col_max_len}" if labels is not None else "")
    )
    for idx, token in enumerate(text):
        is_unk = "✓" if token in unk_tokens else " "
        color, attrs = None, []
        if labels is not None:
            if pred_tags[idx] == labels[idx]:
                if labels[idx] != "O":
                    color = "dark green"
                    attrs = ["bold"]
            elif pred_tags[idx].split("-")[-1] == labels[idx].split("-")[-1]:
                color = "green"
            elif pred_tags[idx] != labels[idx]:
                color = "red"

        print(
            f"{str(idx).ljust(idx_col_max_len)} "
            f"{token.ljust(token_col_max_len)}  "
            f"{is_unk.ljust(is_unk_col_man_len)}  "
            f"{colored(pred_tags[idx].ljust(tag_col_max_len), color=color, attrs=attrs)}"
            + (f"  {labels[idx].ljust(tag_col_max_len)}" if labels is not None else "")
        )

    return pred_tags