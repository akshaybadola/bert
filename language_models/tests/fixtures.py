import pytest

from transformers import AutoTokenizer
from pretrain import get_model_and_config, BertPretrainingCriterion
import dataloader


@pytest.fixture(scope="session")
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased-whole")
    return tokenizer


@pytest.fixture(scope="session")
def model_config(request):
    if hasattr(request, "param") and request.param:
        param = request.param
    else:
        param = "base"
    if param == "base":
        config_file = "configs/bert_base.json"
    elif param == "small":
        config_file = "configs/bert_small.json"
    elif param == "tiny":
        config_file = "configs/bert_tiny_alt.json"
    model_name = config_file.split("/")[1].split(".")[0]
    model, config = get_model_and_config(model_name, config_file, True)
    criterion = BertPretrainingCriterion(config.vocab_size, sequence_output_is_dense=True)
    return model, config, criterion


@pytest.fixture(scope="session")
def data_512(tokenizer, request):
    if hasattr(request, "param") and request.param:
        param = request.param
    else:
        param = "books"
    if param == "books":
        data = dataloader.BertDatasetBooksWiki("books-wiki-tokenized-with-tokens",
                                               tokenizer, shuffle=False,
                                               max_seq_len=512, min_seq_len=30,
                                               truncate_strategy="truncate_second")
    elif param == "owt":
        data = dataloader.BertDatasetOWT("owt-filtered-with-tokens",
                                         tokenizer, whole_word_mask=True,
                                         max_seq_len=512, min_seq_len=30,
                                         truncate_strategy="truncate_second")
    else:
        raise ValueError
    return data


@pytest.fixture(scope="session")
def data_128(tokenizer, request):
    if hasattr(request, "param") and request.param:
        param = request.param
    else:
        param = "books"
    if param == "books":
        data = dataloader.BertDatasetBooksWiki("books-wiki-tokenized-with-tokens",
                                               tokenizer, shuffle=False,
                                               max_seq_len=128, min_seq_len=30,
                                               truncate_strategy="truncate_second")
    elif param == "owt":
        data = dataloader.BertDatasetOWT("owt-filtered-with-tokens",
                                         tokenizer, whole_word_mask=True,
                                         max_seq_len=128, min_seq_len=30,
                                         truncate_strategy="truncate_second")
    else:
        raise ValueError
    return data
