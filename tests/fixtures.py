import pytest

from transformers import AutoTokenizer
import dataloader


@pytest.fixture(scope="session")
def data():
    data = dataloader.BertDataset("books-wiki-tokenized", False)
    return data


@pytest.fixture(scope="session")
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased-whole")
    return tokenizer
