import json
import logging
import os
import random
from functools import lru_cache
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
import torch
from IPython.display import HTML

from masking.data_processing.constants import MASK_ENCODING_MAP, PAD_MASK_TAG, MASK_DECODING_MAP
from masking.utils.styling import COLOR_MAP, ATTRS_MAP

def set_seed(seed: int = 4740):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # Setting `torch.backends.cudnn.benchmark = False` slows down training.
    # Reference: https://pytorch.org/docs/stable/notes/randomness.html.
    torch.backends.cudnn.benchmark = True


@lru_cache(10)
def warn_once(message: str):
    logging.warning(message)


def load_json(filepath: str):
    with open(filepath, "r") as fp:
        data = json.loads(fp.read())
    return data


def colored(text, color: str = "", attrs: Optional[List] = None) -> str:
    if attrs is None:
        attrs = []
    for attr in attrs:
        text = f"{ATTRS_MAP.get(attr, '')}{text}\033[0m"
    return f"{COLOR_MAP.get(color, '')}{text}\033[0m"


def success() -> HTML:
    success_videos = [
        """<img src="https://media.giphy.com/media/3oEdv6UTqzNk9Y5i36/giphy.gif"/>""",
        """<img src="https://media.giphy.com/media/3oz9ZE2Oo9zRC/giphy.gif"/>""",
        """<iframe frameBorder="0" height="270" width="480"
        src="https://giphy.com/embed/8VDO7Fy2PFohfdAnpJ/video"></iframe>""",
        """<iframe frameBorder="0" height="360" width="480"
        src="https://giphy.com/embed/rwqt1f492BBGpAbbSY/video">""",
        """<img src="https://media.giphy.com/media/Srf1W4nnQIb0k/giphy.gif"/>""",
        """<img src="https://media.giphy.com/media/xT8qBepJQzUjXpeWU8/giphy.gif"/>""",
        """<img src="https://media.giphy.com/media/cOvgh3VjLmeg8LLBtk/giphy.gif">""",
        """<img src="https://media.giphy.com/media/12d19apJyRsmA/giphy.gif"/>""",
        """<iframe frameBorder="0" height="320"  width="480"
        src="https://giphy.com/embed/uh26nURBaRpBzy8YRo/video"></iframe>""",
        """<img src="https://media.giphy.com/media/lxyDpcWSJ0a3UdkOfx/giphy.gif">""",
        """<iframe frameBorder="0" height="270" width="480"
        src="https://giphy.com/embed/U4dLVG7d5KsqnN8pBG/video"></iframe>""",
        """<img src="https://media.giphy.com/media/rLENR3QvrRf4A/giphy.gif"/>""",
        """<iframe frameBorder="0" height="270" width="480"
        src="https://giphy.com/embed/cNdJPpoJhOz4D3Aw8G/video"></iframe>""",
        """<img src="https://media.giphy.com/media/3o6Mbnm7WMv7O6yj5K/giphy.gif"/>""",
        """<img src="https://media.giphy.com/media/3o6Mbolqx8Ses8KXoQ/giphy.gif"/>""",
        """<img src="https://media.giphy.com/media/QW5nKIoebG8y4/giphy.gif"/>""",
    ]
    return HTML(random.sample(success_videos, 1)[0])

def get_mask_spans(
    encoded_ids: Union[List, np.ndarray], token_idxs: Optional[Union[List, np.ndarray]] = None
) -> Dict[str, List[Tuple[int]]]:
    extra_tags = [PAD_MASK_TAG, "NONE"]
    label_dict = {k : [] for k in MASK_ENCODING_MAP if not k in extra_tags}

    decoded_tags = [MASK_DECODING_MAP[int(tag_id)] for tag_id in encoded_ids]
    token_idxs = token_idxs if token_idxs is not None else list(range(len(decoded_tags)))
    for decoded_tag, token_idx in zip(decoded_tags, token_idxs):
        if not decoded_tag in extra_tags:
            label_dict[decoded_tag].append(token_idx)

    return label_dict
