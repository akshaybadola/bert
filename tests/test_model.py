import sys
from functools import partial

import torch
from transformers import AutoTokenizer
from common_pyutil.monitor import Timer

import dataloader
from pretrain_bert import get_model_and_config, dict_to, BertPretrainingCriterion
from lamb_amp_opt.fused_lamb import FusedLAMBAMP


timer = Timer()


def loop(iter, model, optimizer, criterion):
    batch = iter.__next__()
    batch = dict_to(batch, torch.device("cuda:0"))
    prediction_scores, seq_relationship_score = model(input_ids=batch['input_ids'],
                                                      token_type_ids=batch['token_type_ids'],
                                                      attention_mask=batch['attention_mask'],
                                                      masked_lm_labels=batch['masked_lm_labels'])
    optimizer.zero_grad()
    loss = criterion(prediction_scores, seq_relationship_score,
                     batch['masked_lm_labels'], batch['next_sentence_labels'])
    loss.backward()
    optimizer.step()


def get_objects(config, batch_size):
    with timer:
        data = dataloader.BertDataset("books-wiki-tokenized", False)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased-whole")
        collate_fn = partial(dataloader.collator, seq_align_len=8, tokenizer=tokenizer, pad_full=True)
        loader = torch.utils.data.DataLoader(data, shuffle=False,
                                             num_workers=16, drop_last=False,
                                             pin_memory=False, batch_size=batch_size,
                                             collate_fn=collate_fn)
        iter = loader.__iter__()
    print(f"dataloader time: {timer.time}")
    criterion = BertPretrainingCriterion(config.vocab_size, sequence_output_is_dense=True)
    return iter, loader, criterion


def test_model_fp32():
    config_file = "BERT/bert_configs/base.json"
    model, config = get_model_and_config(config_file, True)
    batch_size = 64
    print(f"batch size is {batch_size}")
    iter, loader, criterion = get_objects(config, batch_size)
    model.train()
    model = model.cuda(0)
    optimizer = torch.optim.Adam(model.parameters())

    try:
        for _ in range(10):
            with timer:
                loop(iter, model, optimizer, criterion)
            print(f"Loop time: {timer.time}")
    except KeyboardInterrupt:
        return


def test_model_fp16():
    config_file = "BERT/bert_configs/base.json"
    model, config = get_model_and_config(config_file, True)
    batch_size = 96
    print(f"batch size is {batch_size}")
    iter, loader, criterion = get_objects(config, batch_size)
    model.train()
    model = model.half().cuda(0)
    optimizer = torch.optim.Adam(model.parameters())

    try:
        for _ in range(10):
            with timer:
                loop(iter, model, optimizer, criterion)
            print(f"Loop time: {timer.time}")
    except KeyboardInterrupt:
        return


def test_model_lamb():
    config_file = "BERT/bert_configs/base.json"
    model, config = get_model_and_config(config_file, True)
    batch_size = 128
    print(f"batch size is {batch_size}")
    iter, loader, criterion = get_objects(config, batch_size)
    model.train()
    model = model.half().cuda(0)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = FusedLAMBAMP(optimizer_grouped_parameters, lr=0.01)

    try:
        for _ in range(10):
            with timer:
                loop(iter, model, optimizer, criterion)
            print(f"Loop time: {timer.time}")
    except KeyboardInterrupt:
        return
