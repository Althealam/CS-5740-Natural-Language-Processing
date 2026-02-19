from typing import List, Dict
import csv
import re

class Example:
    """
    Data wrapper for a single entailment example.

    Attributes:
        premise (List[str]): list of tokens in premise
        hypothesis (List[str]): list of tokens in hypothesis
        label (int): 0 = contradiction, 1 = entailment
    """
    def __init__(self, premise: List[str], hypothesis: List[str], label: int):
        self.premise = premise
        self.hypothesis = hypothesis
        self.label = label

    def get_combined_words(self) -> List[str]:
        """
        Returns concatenated tokens for feature extraction with a <SEP> token.
        """
        return self.premise + ["<SEP>"] + self.hypothesis

    def __repr__(self):
        return f"Premise={self.premise}; Hypothesis={self.hypothesis}; label={self.label}"

    def __str__(self):
        return self.__repr__()


def tokenize_and_clean(sentence: str) -> List[str]:
    """
    Tokenizes and cleans a sentence.
    """
    sentence = sentence.lower()
    sentence = re.sub(r'http\S+|www\.\S+', '', sentence)
    sentence = re.sub(r'[^a-z0-9\s!?.,-]', '', sentence)
    tokens = [token.strip() for token in sentence.split() if token.strip()]
    return tokens

def read_examples(examples: List[Dict]) -> List[Example]:
    """
    Reads examples from a list of dict items with keys 'premise', 'hypothesis', 'label'.
    """
    exs = []

    for i in range(len(examples["premise"])):
        premise_tokens = tokenize_and_clean(examples['premise'][i])
        hypothesis_tokens = tokenize_and_clean(examples['hypothesis'][i])
        label = int(examples['label'][i])

        if premise_tokens and hypothesis_tokens:
            exs.append(Example(premise_tokens, hypothesis_tokens, label))


    return exs

def split_data(examples: List[Example], train_ratio: float = 0.8) -> tuple:
    """
    Splits examples into training and validation sets.
    """
    import random
    random.shuffle(examples)
    split_idx = int(len(examples) * train_ratio)
    return examples[:split_idx], examples[split_idx:]
