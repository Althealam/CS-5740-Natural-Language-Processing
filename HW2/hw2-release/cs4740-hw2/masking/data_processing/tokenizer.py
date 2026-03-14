# AUTO-GENERATED (DO NOT MODIFY)
# NET IDS: ZL2254,KL2235

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from typing import List, Dict, Optional, Union

import datasets
import torch

from masking.data_processing.constants import PAD_TOKEN, UNK_TOKEN


class Tokenizer(object):
    def __init__(
        self,
        pad_token: str = PAD_TOKEN,
        unk_token: str = UNK_TOKEN,
        lowercase: bool = False,
    ) -> None:
        """
        Creates a :py:class:`~masking.data_processing.tokenizer.Tokenizer` object using the given parameters.

        Parameters
        ----------
        pad_token : str, default: :py:const:`~masking.data_processing.constants.PAD_TOKEN`
            The padding token.
        unk_token : str, default: :py:const:`~masking.data_processing.constants.UNK_TOKEN`
            The unknown token.
        lowercase : bool, default: False
            Determines whether the :py:class:`~masking.data_processing.tokenizer.Tokenizer` converts tokens to lowercase
            before processing.
        """
        super().__init__()

        self.pad_token = pad_token
        self.unk_token = unk_token

        if lowercase:
            logging.warning(f"lowercase set to {lowercase}, which could impact masking")
        self.lowercase = lowercase

        self.token2id = {pad_token: 0, unk_token: 1}

    @property
    def id2token(self) -> Dict[int, str]:
        """
        Retrieves a one-to-one mapping from an ID to a token in the vocabulary.

        Returns
        -------
        Dict[int, str]
            A dictionary that associates a token ID to its corresponding token.
        """
        return {token_id: token for token, token_id in self.token2id.items()}

    @property
    def vocab(self) -> List[str]:
        """
        Retrieves the vocabulary used by the :py:class:`~masking.data_processing.tokenizer.Tokenizer`.

        Returns
        -------
        List[str]
            The vocabulary used by the :py:class:`~masking.data_processing.tokenizer.Tokenizer`; this includes the
            padding and unknown tokens.
        """
        return list(self.token2id.keys())

    @property
    def vocab_size(self) -> int:
        """
        Computes the number of words in the vocabulary.

        Returns
        -------
        int
            The size of the vocabulary (this includes padding and unknown tokens).
        """
        return len(self.token2id.keys())

    def extend(self, tokens: List) -> None:
        """
        Adds a list of words to the current vocabulary (if they don't already exist) and
        consequently, updates the vocabulary size.

        Parameters
        ----------
        tokens : List
            A list of tokens to add to the :py:class:`~masking.data_processing.tokenizer.Tokenizer`'s existing vocabulary.
        """
        existing_vocab_size = self.vocab_size
        if self.lowercase:
            tokens = [token.lower() for token in tokens]

        for token_id, token in enumerate(tokens):
            if token not in self.token2id:
                self.token2id.update({token: existing_vocab_size + token_id})

    def reset(self) -> None:
        """Resets the vocabulary to just the padding and unknown tokens."""
        self.token2id = {self.pad_token: 0, self.unk_token: 1}

    def __str__(self) -> str:
        """
        The ``__str__`` method for the :py:class:`~masking.data_processing.tokenizer.Tokenizer` object.
        """
        return (
            f"Tokenizer(vocab_size={self.vocab_size}, "
            f"pad_token={self.pad_token}, "
            f"unk_token={self.unk_token}, "
            f"lowercase={self.lowercase})"
        )

    def train(
        self,
        train_data: datasets.Dataset,
        text_colname: str = "text",
        min_freq: int = 2,
        remove_frac: float = 0.3,
        reset: bool = True,
    ) -> None:
        """
        Trains the :py:class:`~masking.data_processing.tokenizer.Tokenizer` using the training data, which involves setting
        the vocabulary based on the input hyperparameters ``min_freq`` and ``remove_frac``.

        Parameters
        ----------
        train_data : :py:class:`~datasets.Dataset`
            The training data (arrow format) to create the vocabulary from.
        text_colname : str, default: "text"
            The name of the column in the ``train_data`` that contains the text (sequence of tokens).
        min_freq : int, default: 2
            Determines the minimum frequency of tokens to include in the vocabulary (e.g., ``min_freq = 2`` includes
            all the tokens appearing at least twice).
        remove_frac : float, default: 0.3
            Determines the fraction of tokens to remove from vocabulary; ``int(remove_frac * total_num_tokens)`` are
            removed. Note that ``remove_frac`` is applied on ``min_freq``-filtered output (not the other way around).
        reset : bool, default: True
            Determines whether to reset the vocabulary before training.
        """
        if reset:
            self.reset()
        existing_vocab_size = self.vocab_size

        text_data = chain(*train_data[text_colname])
        if self.lowercase:
            text_data = [token.lower() for token in text_data]
        token_freqs = Counter(text_data)

        valid_tokens = [token for token, freq in token_freqs.items() if freq >= min_freq]
        logging.info(f"num of unique tokens retained after min freq of {min_freq} filtering: {len(valid_tokens)}")
        top_tokens = sorted(valid_tokens, key=lambda token: token_freqs[token], reverse=True)
        top_tokens = top_tokens[: len(top_tokens) - int(remove_frac * len(top_tokens))]
        logging.info(f"num of unique tokens retained after {remove_frac} fraction of tokens removed: {len(top_tokens)}")

        self.token2id.update({token: existing_vocab_size + token_id for token_id, token in enumerate(top_tokens)})

    def save(self, filepath: str) -> None:
        """
        Saves the :py:attr:`~masking.data_processing.tokenizer.Tokenizer.token2id` (token to ID mapping) as a .json dump,
         to given filepath.

        Parameters
        ----------
        filepath : str
            The filepath to save the :py:attr:`~masking.data_processing.tokenizer.Tokenizer.token2id` to.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as fp:
            json.dump(self.token2id, fp)

    def from_file(self, filepath: str) -> None:
        """
        Loads :py:attr:`~masking.data_processing.tokenizer.Tokenizer.token2id` (token to ID mapping) from a .json file.

        Parameters
        ----------
        filepath : str
            The filepath to load the :py:attr:`~masking.data_processing.tokenizer.Tokenizer.token2id` from.
        """
        with open(filepath, "r") as fp:
            self.token2id = json.load(fp)

    def from_dict(self, token2id_dict: Dict[str, int]) -> None:
        """
        Loads :py:attr:`~masking.data_processing.tokenizer.Tokenizer.token2id` (token to ID mapping) from a dictionary.

        Parameters
        ----------
        token2id_dict : Dict[str, int]
            Dictionary containing token to ID mappings.
        """
        self.token2id = token2id_dict

    def tokenize(
        self,
        input_seq: Union[List[str], str],
        max_length: Optional[int] = None,
        padding_side: str = "right",
        truncation_side: str = "right",
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenizes a given input sequence to return ``input_ids`` and ``padding_mask``.

        Parameters
        ----------
        input_seq : Union[List[str], str]
            The input sequence to tokenize.
        max_length : Optional[int], default: None
            The desired length of the tokenized sequence; think length of ``input_ids``.
        padding_side : {"left", "right"}, default: "right"
            Indicates which side to pad the input sequence on.
        truncation_side : {"left", "right"}, default: "right"
            Indicates which side to truncate the input sequence on.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary that contains the token IDs of tokens in the input sequence under the key ``"input_ids"`` and
            their corresponding padding mask under the key ``"padding_mask"``.
        """

        # TODO-1.1-1
        # NOTE: 1=padding, 0=real token
        if isinstance(input_seq, str):
            input_seq = input_seq.split()

        if self.lowercase:
            input_seq = [token.lower() for token in input_seq]

        input_ids = [self.token2id.get(token, self.token2id[self.unk_token]) for token in input_seq]
        # NOTE: change the mask value
        padding_mask = [0] * len(input_ids)

        if max_length is not None:
            if len(input_ids) > max_length:
                if truncation_side == "right":
                    input_ids = input_ids[:max_length]
                    padding_mask = padding_mask[:max_length]
                elif truncation_side == "left":
                    input_ids = input_ids[-max_length:]
                    padding_mask = padding_mask[-max_length:]
                else:
                    raise ValueError("truncation_side must be either 'left' or 'right'")

            elif len(input_ids) < max_length:
                pad_len = max_length - len(input_ids)
                pad_id = self.token2id[self.pad_token]

                if padding_side == "right":
                    input_ids = input_ids + [pad_id] * pad_len
                    # NOTE: change the mask value
                    padding_mask = padding_mask + [1] * pad_len
                elif padding_side == "left":
                    input_ids = [pad_id] * pad_len + input_ids
                    # NOTE: change the mask value
                    padding_mask = [1] * pad_len + padding_mask
                else:
                    raise ValueError("padding_side must be either 'left' or 'right'")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "padding_mask": torch.tensor(padding_mask, dtype=torch.long),
        }




        
    def decode(self, input_ids: torch.Tensor, return_as_list=False) -> Union[List[str], str]:
        """
        Converts a sequence of token IDs back into a sequence of tokens.

        Parameters
        ----------
        input_ids : torch.Tensor
            The sequence to input IDs to convert.
        return_as_list : bool, default: False
            If False, returns the sequence as a string. Otherwise, returns as a list of strings.

        Returns
        -------
        Union[List[str], str]
            The decoded sequence of tokens corresponding to ``input_ids``.
        """
        if return_as_list:
            return [self.id2token[input_id] for input_id in input_ids.numpy()]
        return " ".join([self.id2token[input_id] for input_id in input_ids.numpy()])
