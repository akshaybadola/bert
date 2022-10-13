from transformers import BertTokenizer, BertModel
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


def save_data(dataset, seed=1122, path="./books-wiki-tokenized"):
    dataset.save_to_disk(path)


def load_saved_data(path="./books-wiki-tokenized"):
    return load_from_disk(path)


def build_wiki_books():
    data = load_data()
    tokenizer = load_saved_tokenizer("bert-base-uncased-whole")
    processed_data = preprocess_data(data, tokenizer, remove_text=False)
    save_data(processed_data)
