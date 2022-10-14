from functools import partial

import torch
from transformers import AutoTokenizer
import numpy as np

import dataloader


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



def test_data_128(tokenizer):
    data = dataloader.BertDataset("books-wiki-tokenized", shuffle=False,
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
