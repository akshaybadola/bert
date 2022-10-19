from functools import partial

from transformers import BertTokenizer, BertModel, AutoTokenizer
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import BertTokenizerFast


def load_data() -> None:
    wiki = load_dataset("wikipedia", "20220301.en", split="train")
    books = load_dataset("bookcorpus", split="train")

    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
    train_data = concatenate_datasets([books, wiki])
    # TODO: Dedupe maybe
    return train_data


def create_tokenizer(train_data, tokenizer_id="bert-base-uncased-custom"):
    # create a python generator to dynamically load the data
    def batch_iterator(batch_size=10000):
        for i in range(0, len(train_data), batch_size):
            yield train_data[i: i + batch_size]["text"]

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bert_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(),
                                                       vocab_size=32000)
    bert_tokenizer.save_pretrained("./" + tokenizer_id)


def load_saved_tokenizer(tokenizer_id="bert-base-uncased-custom"):
    return BertTokenizerFast.from_pretrained(tokenizer_id)


# tokenizer = load_saved_tokenizer()
# train_data = load_data()
def preprocess_data(train_data, tokenizer, num_proc=64, remove_text=True):
    def group_texts(tokenizer, examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True,
                         truncation=True, max_length=tokenizer.model_max_length)
    from functools import partial
    preproc_func = partial(group_texts, tokenizer)
    if remove_text:
        tokenized_dataset = train_data.map(preproc_func, batched=True,
                                           remove_columns=["text"], num_proc=num_proc,
                                           keep_in_memory=True)
    else:
        tokenized_dataset = train_data.map(preproc_func, batched=True,
                                           num_proc=num_proc, keep_in_memory=True)
    return tokenized_dataset


def preprocess_data_with_tokens(train_data, tokenizer, num_proc=64, remove_text=True):
    def group_texts(tokenizer, examples):
        return {**tokenizer(examples["text"], add_special_tokens=False,
                            return_special_tokens_mask=False, return_attention_mask=False,
                            return_token_type_ids=False, truncation=True,
                            max_length=tokenizer.model_max_length),
                "tokens": tokenizer.tokenize(examples["text"])}
    from functools import partial
    preproc_func = partial(group_texts, tokenizer)
    if remove_text:
        tokenized_dataset = train_data.map(preproc_func, batched=False,
                                           remove_columns=["text"], num_proc=num_proc,
                                           keep_in_memory=True)
    else:
        tokenized_dataset = train_data.map(preproc_func, batched=False,
                                           num_proc=num_proc, keep_in_memory=True)
    return tokenized_dataset


def preprocess_data_text(train_data, tokenizer, num_proc=64, min_length=5):
    def _filter_func(tokenizer, min_length, line):
        tokenized = tokenizer.tokenize(line)
        return tokenized and len(tokenized) > min_length

    filter_func = partial(_filter_func, tokenizer, min_length)

    def preprocess(tokenizer, examples):
        lines = examples['text'].split("\n\n")
        filtered = [*filter(filter_func, lines)]
        tokens = [*map(tokenizer.tokenize, filtered)]
        input_ids = [*map(tokenizer.convert_tokens_to_ids, tokens)]
        return {"text": "\n\n".join(filtered),
                "tokens": tokens,
                "input_ids": input_ids}
    preproc_func = partial(preprocess, tokenizer)
    processed_dataset = train_data.map(preproc_func, batched=False,
                                       num_proc=num_proc,
                                       keep_in_memory=True)
    return processed_dataset


def preprocess_data_with_markers(train_data, tokenizer, num_proc=64, remove_text=True):
    def group_texts(tokenizer, examples):
        tokens = tokenizer.tokenize(examples["text"])
        # i = 0
        # j = 0
        # for x in tokens:
        #     if x.startswith("#"):
        #         split_a.append(j)
        #     else:
        #         split_a.append(i)
        #         j = i
        #         i += 1
        # text = examples["text"]
        # text == "The child is eating strawberries"
        # markers == ["child", "eating", "straberries"]
        # tokens == ["The", "child", "is", "eating", "straw", "##berries"]
        # output == [0, 1, 0, 1, 1, 1]
        markers = []
        return {**tokenizer(examples["text"], add_special_tokens=False,
                            return_special_tokens_mask=False, return_attention_mask=False,
                            return_token_type_ids=False, truncation=True,
                            max_length=tokenizer.model_max_length),
                "markers": markers,
                "tokens": tokens}
    from functools import partial
    preproc_func = partial(group_texts, tokenizer)
    if remove_text:
        tokenized_dataset = train_data.map(preproc_func, batched=False,
                                           remove_columns=["text"], num_proc=num_proc,
                                           keep_in_memory=True)
    else:
        tokenized_dataset = train_data.map(preproc_func, batched=False,
                                           num_proc=num_proc, keep_in_memory=True)
    return tokenized_dataset


def save_data(dataset, path="./books-wiki-tokenized"):
    dataset.save_to_disk(path)


def load_saved_data(path="./books-wiki-tokenized"):
    return load_from_disk(path)


def build_wiki_books(with_tokens=False):
    data = load_data()
    tokenizer = load_saved_tokenizer("bert-base-uncased-whole")
    if with_tokens:
        processed_data = preprocess_data_with_tokens(data, tokenizer, remove_text=False)
    else:
        processed_data = preprocess_data(data, tokenizer, remove_text=False)
    save_data(processed_data, path="./books-wiki-tokenized-with-tokens")


def process_owt(num_proc=8):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased-whole")
    owt = load_dataset("openwebtext")
    processed_data = preprocess_data_text(owt['train'], tokenizer, num_proc, 5)
    save_data(processed_data, path="./owt-filtered-with-tokens")
