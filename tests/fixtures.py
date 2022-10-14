import pytest

from transformers import AutoTokenizer
from pretrain_bert import get_model_and_config, BertPretrainingCriterion
import dataloader


@pytest.fixture(scope="session")
def data():
    data = dataloader.BertDataset("books-wiki-tokenized", False)
    return data


@pytest.fixture(scope="session")
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased-whole")
    return tokenizer


@pytest.fixture(scope="session")
def model_config():
    config_file = "BERT/bert_configs/base.json"
    model, config = get_model_and_config(config_file, True)
    criterion = BertPretrainingCriterion(config.vocab_size, sequence_output_is_dense=True)
    return model, config, criterion
    
