from functools import partial

import torch
from transformers import AutoTokenizer
import numpy as np

import dataloader
from common_pyutil.monitor import Timer
timer = Timer()


def test_dataloader(data, tokenizer):
    batch_size = 32
    inds = np.arange(len(data))
    batch_inds = np.random.choice(inds, batch_size)
    batch = [data[i] for i in batch_inds]
    collated = dataloader.mlm_collator(batch, 8, tokenizer)
    assert collated['input_ids'].shape[1] <= 512
    lengths = np.array([x.shape[1] for x in collated.values() if len(x.shape) == 2])
    assert np.all(lengths == lengths[0])

    collate_fn = partial(dataloader.mlm_collator, seq_align_len=8, tokenizer=tokenizer)
    loader = torch.utils.data.DataLoader(data, shuffle=False,
                                         num_workers=32, drop_last=False,
                                         pin_memory=True, batch_size=batch_size,
                                         collate_fn=collate_fn)
    batch = loader.__iter__().__next__()
    assert batch['input_ids'].shape[1] <= 512
    lengths = np.array([x.shape[1] for x in batch.values() if len(x.shape) == 2])
    assert np.all(lengths == lengths[0])


def test_dataloader_whole_words(tokenizer):
    data = dataloader.BertDataset("books-wiki-tokenized-with-tokens", tokenizer,
                                  shuffle=False,
                                  max_seq_len=128, min_seq_len=30,
                                  truncate_strategy="truncate_second")
    batch_size = 1024
    collate_fn = partial(dataloader.mlm_collator, seq_align_len=8,
                         tokenizer=tokenizer, mask_whole_words=True)
    loader = torch.utils.data.DataLoader(data, shuffle=False,
                                         num_workers=4, drop_last=False,
                                         pin_memory=True, batch_size=batch_size,
                                         collate_fn=collate_fn)
    iter = loader.__iter__()
    for _ in range(10):
        with timer:
            batch = iter.__next__()
            assert batch['input_ids'].shape[1] <= 512
            lengths = np.array([x.shape[1] for x in batch.values() if len(x.shape) == 2])
            assert np.all(lengths == lengths[0])
        print(timer.time)


def test_data_128_without_tokens(tokenizer):
    data = dataloader.BertDataset("books-wiki-tokenized", shuffle=False,
                                  whole_word_mask=False,
                                  max_seq_len=128, min_seq_len=30,
                                  truncate_strategy="truncate_second")
    batch_size = 32
    inds = np.arange(len(data))
    batch_inds = np.random.choice(inds, batch_size)
    batch = [data[i] for i in batch_inds]
    collated = dataloader.mlm_collator(batch, 8, tokenizer)
    assert collated['input_ids'].shape[1] <= 128
    lengths = np.array([x.shape[1] for x in collated.values() if len(x.shape) == 2])
    assert np.all(lengths == lengths[0])


def test_data_128_with_tokens(tokenizer):
    data = dataloader.BertDataset("books-wiki-tokenized-with_tokens", shuffle=False,
                                  whole_word_mask=False,
                                  max_seq_len=128, min_seq_len=30,
                                  truncate_strategy="truncate_second")
    batch_size = 32
    inds = np.arange(len(data))
    batch_inds = np.random.choice(inds, batch_size)
    batch = [data[i] for i in batch_inds]
    collated = dataloader.mlm_collator(batch, 8, tokenizer)
    assert collated['input_ids'].shape[1] <= 128
    lengths = np.array([x.shape[1] for x in collated.values() if len(x.shape) == 2])
    assert np.all(lengths == lengths[0])



def test_data_whole_word(tokenizer):
    data = dataloader.BertDataset("books-wiki-tokenized-with-tokens", tokenizer,
                                  shuffle=False,
                                  max_seq_len=128, min_seq_len=30,
                                  truncate_strategy="truncate_second")
    batch_size = 32
    inds = np.arange(len(data))
    batch_inds = np.random.choice(inds, batch_size)
    batch = [data[i] for i in batch_inds]
    with timer:
        collated = dataloader.mlm_collator(batch, 8, tokenizer)
    print(timer.time)
    with timer:
        collated_whole = dataloader.mlm_collator(batch, 8, tokenizer, mask_whole_words=True)
    print(timer.time)
    assert collated['input_ids'].shape[1] <= 128
    lengths = np.array([x.shape[1] for x in collated.values() if len(x.shape) == 2])
    assert np.all(lengths == lengths[0])
    assert collated_whole['input_ids'].shape[1] <= 128
    lengths = np.array([x.shape[1] for x in collated_whole.values() if len(x.shape) == 2])
    assert np.all(lengths == lengths[0])
    
