#: The special representation for a padding token.
PAD_TOKEN = "<|pad|>"

#: The special representation for an unknown token.
UNK_TOKEN = "<|unk|>"

#: The MASK label placeholder associated with a :py:const:`~masking.data_processing.constants.PAD_TOKEN`.
PAD_MASK_TAG = "<|pad|>"

#: A dictionary mapping possible MASK labels to unique IDs.
MASK_ENCODING_MAP = {PAD_MASK_TAG: -100, "TIME": 0, "USERNAME": 1, "IDCARD": 2, "EMAIL": 3, "SOCIALNUMBER": 4, "PASSPORT": 5, "DRIVERLICENSE": 6, "BOD": 7, "IP": 8, "NONE": 9}

#: The reverse mapping of :py:const:`~masking.data_processing.constants.MAKS_ENCODING_MAP`.
MASK_DECODING_MAP = {entity_id: entity for entity, entity_id in MASK_ENCODING_MAP.items()}

if __name__ == "__main__":
    assert len(set(MASK_ENCODING_MAP.values())) == len(MASK_ENCODING_MAP.keys())
