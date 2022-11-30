import sys
from functools import partial

import pytest

import torch
from transformers import AutoTokenizer
from common_pyutil.monitor import Timer

import dataloader
from lamb_amp_opt.fused_lamb import FusedLAMBAMP


timer = Timer()


def loop(iter, model, optimizer, criterion):
    batch = iter.__next__()
    batch = {k: v.to(torch.device("cuda:0")) for k, v in batch.items()}
    prediction_scores, seq_relationship_score = model(input_ids=batch['input_ids'],
                                                      token_type_ids=batch['token_type_ids'],
                                                      attention_mask=batch['attention_mask'],
                                                      masked_lm_labels=batch['masked_lm_labels'])
    optimizer.zero_grad()
    loss = criterion(prediction_scores, seq_relationship_score,
                     batch['masked_lm_labels'], batch['next_sentence_labels'])
    loss.backward()
    optimizer.step()


def take_training_step(batch, grad_scaler, model, criterion, accumulation_steps):
    with torch.cuda.amp.autocast(enabled=True):
    # with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
        prediction_scores, seq_relationship_score = model(input_ids=batch['input_ids'],
                                                          token_type_ids=batch['token_type_ids'],
                                                          attention_mask=batch['attention_mask'],
                                                          masked_lm_labels=batch['masked_lm_labels'])
        loss = criterion(prediction_scores, seq_relationship_score,
                         batch['masked_lm_labels'], batch['next_sentence_labels'])
    # loss = loss / accumulation_steps
    grad_scaler.scale(loss).backward()


def take_optimizer_step(optimizer, grad_scaler):
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad(set_to_none=True)


def get_loader(data, tokenizer, batch_size):
    collate_fn = partial(dataloader.mlm_collator, seq_align_len=8,
                         tokenizer=tokenizer, pad_full=data.max_seq_len)
    loader = torch.utils.data.DataLoader(data, shuffle=False,
                                         num_workers=16, drop_last=False,
                                         pin_memory=False, batch_size=batch_size,
                                         collate_fn=collate_fn)
    return loader


@pytest.mark.parametrize("model_config", ["base"],
                         indirect=True)
@pytest.mark.parametrize("data_512", ["books", "owt"],
                         indirect=True)
def test_model_fp32_seq_512(data_512, tokenizer, model_config):
    model, config, criterion = model_config
    batch_size = 64
    print(f"batch size is {batch_size}")
    loader = get_loader(data_512, tokenizer, batch_size)
    print("Got loader")
    iter = loader.__iter__()
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


@pytest.mark.parametrize("model_config", ["base", "small", "tiny"],
                         indirect=True)
@pytest.mark.parametrize("data_128", ["books", "owt"],
                         indirect=True)
def test_model_fp32_seq_128(data_128, tokenizer, model_config):
    model, config, criterion = model_config
    print(f"model is {model.model_name}")
    batch_size = 64
    print(f"batch size is {batch_size}")
    loader = get_loader(data_128, tokenizer, batch_size)
    print("Got loader")
    iter = loader.__iter__()
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


@pytest.mark.parametrize("model_config", ["base"],
                         indirect=True)
@pytest.mark.parametrize("data_512", ["books", "owt"],
                         indirect=True)
def test_model_fp16_seq_512(data_512, tokenizer, model_config):
    model, config, criterion = model_config
    print(f"model is {model.model_name}")
    batch_size = 96
    print(f"batch size is {batch_size}")
    loader = get_loader(data_512, tokenizer, batch_size)
    iter = loader.__iter__()
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

@pytest.mark.parametrize("model_config", ["base"],
                         indirect=True)
@pytest.mark.parametrize("data_128", ["books", "owt"],
                         indirect=True)
def test_model_fp16_seq_128(tokenizer, model_config):
    model, config, criterion = model_config
    print(f"model is {model.model_name}")
    batch_size = 512
    print(f"batch size is {batch_size}")
    loader = get_loader(data_128, tokenizer, batch_size)
    iter = loader.__iter__()
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


def test_model_lamb(tokenizer, model_config):
    model, config, criterion = model_config
    with timer:
        data = dataloader.BertDataset("books-wiki-tokenized", False,
                                      max_seq_len=128, min_seq_len=30,
                                      truncate_strategy="truncate_second")
    batch_size = 512
    print(f"batch size is {batch_size}")
    loader = get_loader(data, tokenizer, batch_size)
    iter = loader.__iter__()
    model.train()
    model = model.half().cuda(0)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = FusedLAMBAMP(optimizer_grouped_parameters, lr=0.01)
    optimizer.setup_fp32_params()

    accumulation_steps = 8
    num_loops = 2
    total_steps = num_loops * accumulation_steps

    grad_scaler = torch.cuda.amp.GradScaler(init_scale=2.0 ** 20, enabled=True)

    try:
        for i in range(total_steps):
            with timer:
                batch = iter.__next__()
                batch = {k: v.to(torch.device("cuda:0")) for k, v in batch.items()}
                take_training_step(batch, grad_scaler, model, criterion, accumulation_steps)
            print(f"Loop time: {timer.time}")
            if not (i+1) % accumulation_steps:
                with timer:
                    take_optimizer_step(optimizer, grad_scaler)
                print(f"Optimizer step took time {timer.time}")
    except KeyboardInterrupt:
        return


def test_model_adam(tokenizer, model_config):
    model, config, criterion = model_config
    with timer:
        data = dataloader.BertDataset("books-wiki-tokenized", False,
                                      max_seq_len=128, min_seq_len=30,
                                      truncate_strategy="truncate_second")
    batch_size = 512
    print(f"batch size is {batch_size}")
    loader = get_loader(data, tokenizer, batch_size)
    iter = loader.__iter__()
    model.train()
    model = model.cuda(0)
    # model = model.half().cuda(0)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=0.01)
    accumulation_steps = 1
    num_loops = 2
    total_steps = num_loops * accumulation_steps * 8

    grad_scaler = torch.cuda.amp.GradScaler(init_scale=2.0 ** 20, enabled=True)

    try:
        for i in range(total_steps):
            with timer:
                batch = iter.__next__()
                batch = {k: v.to(torch.device("cuda:0")) for k, v in batch.items()}
                take_training_step(batch, grad_scaler, model, criterion, accumulation_steps)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()
            print(f"Loop time: {timer.time}")
    except KeyboardInterrupt:
        return


def test_model_adam_dataparallel(tokenizer, model_config):
    model, config, criterion = model_config
    with timer:
        data = dataloader.BertDataset("books-wiki-tokenized", False,
                                      max_seq_len=128, min_seq_len=30,
                                      truncate_strategy="truncate_second")
    batch_size = 512 * 2
    print(f"batch size is {batch_size}")
    loader = get_loader(data, tokenizer, batch_size)
    iter = loader.__iter__()
    model = model.cuda(0)
    model.train()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.train()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=0.01)
    accumulation_steps = 1
    num_loops = 2
    total_steps = num_loops * accumulation_steps * 8

    grad_scaler = torch.cuda.amp.GradScaler(init_scale=2.0 ** 20, enabled=True)

    try:
        for i in range(total_steps):
            with timer:
                batch = iter.__next__()
                batch = {k: v.to(torch.device("cuda:0")) for k, v in batch.items()}
                take_training_step(batch, grad_scaler, model, criterion, accumulation_steps)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()
            print(f"Loop time: {timer.time}")
    except KeyboardInterrupt:
        return


def test_model_lamb_dataparallel(tokenizer, model_config):
    model, config, criterion = model_config
    with timer:
        data = dataloader.BertDataset("books-wiki-tokenized", False,
                                      max_seq_len=128, min_seq_len=30,
                                      truncate_strategy="truncate_second")
    batch_size = 512 * 2
    print(f"batch size is {batch_size}")
    loader = get_loader(data, tokenizer, batch_size)
    iter = loader.__iter__()
    model = model.cuda(0)
    model.train()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.train()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = FusedLAMBAMP(optimizer_grouped_parameters, lr=0.01)
    optimizer.setup_fp32_params()

    accumulation_steps = 8
    num_loops = 2
    total_steps = num_loops * accumulation_steps

    grad_scaler = torch.cuda.amp.GradScaler(init_scale=2.0 ** 20, enabled=True)

    try:
        for i in range(total_steps):
            with timer:
                batch = iter.__next__()
                batch = {k: v.to(torch.device("cuda:0")) for k, v in batch.items()}
                take_training_step(batch, grad_scaler, model, criterion, accumulation_steps)
            print(f"Loop time: {timer.time}")
            if not (i+1) % accumulation_steps:
                with timer:
                    take_optimizer_step(optimizer, grad_scaler)
                print(f"Optimizer step took time {timer.time}")
    except KeyboardInterrupt:
        return
    
