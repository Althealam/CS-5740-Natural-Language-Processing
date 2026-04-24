import textwrap
from typing import Optional

import torch
import datasets

from seagull.data_processing.bbpe import BBPETokenizer
from seagull.data_processing.constants import END_OF_CAPTION_TOKEN
from seagull.model.heads.seagull_lm import SeagullLM
from seagull.utils.torch_utils import set_pytorch_backends
from seagull.utils.utils import colored
from torch.utils.data import DataLoader
from seagull.utils.metrics import compute_perplexity

text_wrapper = textwrap.TextWrapper(width=120)
set_pytorch_backends()


@torch.no_grad()
def make_seagull_talk(
    model: SeagullLM,
    bbpe_tokenizer: BBPETokenizer,
    prompt: Optional[str] = None,
    max_new_tokens: int = 20,
    num_samples: int = 5,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> None:
    model.eval()

    initial_seq_to_print = "\n".join(text_wrapper.wrap(prompt if prompt is not None else ""))
    if prompt is None or prompt == "":
        prompt = bbpe_tokenizer.eos_token  # generate unconditional samples
    input_ids = torch.tensor(bbpe_tokenizer.encode(text=prompt, add_special_tokens=False), dtype=torch.long)
    input_seq_length = len(input_ids)
    generated_samples = model.talk(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_samples=num_samples,
        top_k=top_k,
        eos_token_id=bbpe_tokenizer.token2id(END_OF_CAPTION_TOKEN),
    )
    for sample in generated_samples:
        decoded_sample_to_print = "\n".join(text_wrapper.wrap(bbpe_tokenizer.decode(sample[input_seq_length:])))
        print(f"{initial_seq_to_print}\n>>>{colored(decoded_sample_to_print, 'blue')}")
        print("-" * 120)

@torch.no_grad()
def compute_seagull_perplexity(
    model: SeagullLM,
    dataset: datasets.Dataset,
    batch_size: int,
    ignore_label_idx: int
) -> torch.Tensor:

    # Set the model to evaluation mode
    model.eval()
    
    # TODO-7
    # Tip: Divide the corpus into batches (in your experiments, always pass batch size <32, colab may crash with larger batch sizes)    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    perplexities = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(model.device)
        labels = batch["labels"].to(model.device)

        lm_logits = model(input_ids=input_ids)[0]

        batch_perplexity = compute_perplexity(
            lm_logits,
            labels,
            ignore_label_idx
        )
        perplexities.append(batch_perplexity)

    perplexity = torch.stack(perplexities).mean()
    return perplexity