import code
from datasets import load_dataset, Dataset
from collections import defaultdict
from tqdm.auto import tqdm
import json
import pathlib
import random

def load_sem_anto_neg(*args, **kwargs):
    """Load rows SemAntoNeg and shuffle the answers in a deterministic way"""

    ds = load_dataset("json", data_files={"test": "https://raw.githubusercontent.com/Helsinki-NLP/SemAntoNeg/6b059edbfc60dfb0872d863adecbeddd6121e8d3/SemAntoNeg_v1.0.json"}, split="test")
    assert all(x == 2 for x in ds["label"]) # We are doing a shuffle, let's not get an unpleasant surprise wherein the last answer is not the correct one
    random.seed(194123)
    result: list[dict] = []
    for sentences, input_sentece, idx in zip(ds["sentences"], ds["input"], ds["idx"]):
        shuffle = list(range(len(sentences)))
        random.shuffle(shuffle)
        # One-index label
        result.append({"label": shuffle.index(len(sentences)-1), "input": input_sentece, "idx": idx})
        result[-1]["sentences"] = [sentences[i] for i in shuffle]
    return {"test": Dataset.from_list(result)}