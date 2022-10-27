from typing import Final
from types import SimpleNamespace
import json
import math
from functools import partial
from pathlib import Path
import os

import numpy as np

import torch
import transformers

import args as Args
from models.bert import modeling
from schedulers import LinearWarmUpScheduler, PolyWarmUpScheduler
from lamb_amp_opt.fused_lamb import FusedLAMBAMP

from dataloader import get_wiki_books_loader, get_owt_loader
from simple_trainer.helpers import SimpleUpdateFunction
from simple_trainer.trainer import Trainer
from simple_trainer import functions


def dict_to(tensors, device):
    return {k: v.to(device) for k, v in tensors.items()}


custom_model = True


class BertUpdateFunction(SimpleUpdateFunction):
    def __init__(self, grad_scaler, lr_scheduler, grad_accumulation_steps):
        self._train = True
        self._grad_scaler = grad_scaler
        self._lr_scheduler = lr_scheduler
        self._grad_accumulation_steps = grad_accumulation_steps
        self._returns = ["loss", "scaled_loss", "total"]

    # Leaky Abstraction alert!
    # We need to know whether batch_num starts from 0 or 1, LOL
    # Have worked around it though
    def is_gradient_accumulation_step(self, batch_num):
        if batch_num == 0:
            self._batch_starts_from_zero = True
        else:
            self._batch_starts_from_zero = False
        if self._batch_starts_from_zero:
            batch_num = batch_num + 1
        return not (batch_num % self._grad_accumulation_steps)

    def __call__(self, batch, criterion, model, optimizer, **kwargs):
        batch = {k: model.to_(v) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=True):
            if custom_model:
                prediction_scores, seq_relationship_score = model(input_ids=batch['input_ids'],
                                                                  token_type_ids=batch['token_type_ids'],
                                                                  attention_mask=batch['attention_mask'],
                                                                  masked_lm_labels=batch['labels'])
                loss = criterion(prediction_scores, seq_relationship_score,
                                 batch['labels'], batch['next_sentence_labels'])
            else:
                outputs = model(input_ids=batch['input_ids'],
                                token_type_ids=batch['token_type_ids'],
                                attention_mask=batch['attention_mask'],
                                labels=batch['labels'])
                loss = criterion(outputs['prediction_logits'], outputs['seq_relationship_logits'],
                                 batch['labels'],
                                 batch['next_sentence_labels'])
            loss = loss / self._grad_accumulation_steps
        scaled_loss = self._grad_scaler.scale(loss)
        scaled_loss.backward()
        total = batch["input_ids"].shape[0]
        if self.is_gradient_accumulation_step(kwargs["batch_num"]):
            # if "trainer" in kwargs:
            #     kwargs["trainer"].logger.info("Taking backward step")
            # else:
            #     print("Taking backward step")
            self._lr_scheduler.step()  # learning rate warmup
            self._grad_scaler.step(optimizer)
            self._grad_scaler.update()
            optimizer.zero_grad(set_to_none=True)
        return {"loss": loss.detach().item(),
                "scaled_loss": scaled_loss.detach().item(),
                "total": total}


class BertPretrainingCriterion(torch.nn.Module):

    sequence_output_is_dense: Final[bool]

    def __init__(self, vocab_size, sequence_output_is_dense=False, nsp=True):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size
        self.sequence_output_is_dense = sequence_output_is_dense
        self.nsp = nsp

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels,
                next_sentence_labels=None):
        if self.sequence_output_is_dense:
            # prediction_scores are already dense
            masked_lm_labels_flat = masked_lm_labels.view(-1)
            mlm_labels = masked_lm_labels_flat[masked_lm_labels_flat != -1]
            masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size),
                                          masked_lm_labels.view(-1))
        if self.nsp:
            next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2),
                                              next_sentence_labels.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return masked_lm_loss


def get_model_and_config(model_name, config_file, sequence_output_is_dense=True):
    if custom_model:
        config = modeling.BertConfig.from_json_file(config_file)
    else:
        config = transformers.BertConfig.from_json_file(config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    if custom_model:
        model = modeling.BertForPreTraining(config,
                                            sequence_output_is_dense=sequence_output_is_dense)
    else:
        model = transformers.models.bert.BertForPreTraining(config)
    return model, config


def get_optimizer(optimizer_name, model, devices, warmup_proportion,
                  max_steps, learning_rate, init_loss_scale):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    if optimizer_name == "lamb":
        device = torch.device(devices[0])
        optimizer = FusedLAMBAMP(optimizer_grouped_parameters,
                                 lr=learning_rate, device=device)
        lr_scheduler = PolyWarmUpScheduler(optimizer,
                                           warmup=warmup_proportion,
                                           total_steps=max_steps,
                                           base_lr=learning_rate,
                                           device=device)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                     lr=learning_rate)
        lr_scheduler = LinearWarmUpScheduler(optimizer,
                                             warmup=warmup_proportion,
                                             total_steps=max_steps)
    else:
        raise ValueError(f"Unknown optmizer {optimizer_name}")
    grad_scaler = torch.cuda.amp.GradScaler(init_scale=init_loss_scale, enabled=True)
    return optimizer, lr_scheduler, grad_scaler


def resume_optimizer(optimizer, checkpoint, phase1, phase2_from_scratch,
                     resume, grad_scaler=None, lr=None):
    # For phase2 from scratch, need to reset the learning rate
    # and step count in the checkpoint. Else restore values in checkpoint.
    if phase2_from_scratch or resume:  # lr implies init_checkpoint
        for group in checkpoint['optimizer']['param_groups']:
            group['step'].zero_()
            group['lr'].fill_(lr)
    else:
        if 'grad_scaler' in checkpoint and phase1:
            grad_scaler.load_state_dict(checkpoint['grad_scaler'])
    optimizer.load_state_dict(checkpoint['optimizer'])  # , strict=False)


def load_grad_scaler_and_scheduler(self, saved_state, num_iters=0):
    self._grad_scaler.load_state_dict(saved_state["grad_scaler"])
    if "lr_scheduler" in saved_state:
        self._lr_scheduler.load_state_dict(saved_state["lr_scheduler"])
    else:
        if num_iters:
            self._lr_scheduler.last_epoch = num_iters
        else:
            max_epochs = saved_state["params"]["max_epochs"]
            if max_epochs == 1:
                raise ValueError("Must specify num_iters when steps based training")
            else:
                self._lr_scheduler.last_epoch = len(self.dataloaders["train"]) * (self.epoch+1)


def dump_extra_opts(self, **kwargs):
    if self.extra_opts:
        self.logger.info("Dumping extra opts")
        for k, v in self.extra_opts.items():
            with open(os.path.join(self.savedir, f"{k}.json"), "w") as f:
                json.dump(v, f)


def main():
    args = Args.parse_arguments()
    if not args.resume_dir:
        if args.train_config_file:
            with open(args.train_config_file) as f:
                default_config = SimpleNamespace(**json.load(f))
        else:
            default_config = SimpleNamespace()
        for k, v in default_config.__dict__.items():
            if k in args.__dict__ and not args.__dict__[k]:
                args.__dict__[k] = v
            elif k not in args.__dict__:
                args.__dict__[k] = v
    sequence_output_is_dense = args.dense_sequence_output
    model, config = get_model_and_config(args.model_name, args.model_config_file)
    model.model_name = args.model_name

    # NOTE: For DataParallel, for DDP a different approach will have to be used
    if args.start_phase2 or args.resume_phase2:
        max_seq_len = args.max_seq_len_phase2
        device_batch_size = args.device_batch_size_phase2
        num_datapoints = args.train_steps_phase2 * 256
        warmup_proportion = args.warmup_steps_phase2 / args.train_steps_phase2
        learning_rate = args.learning_rate_phase2
        phase2 = True
    else:
        device_batch_size = args.device_batch_size_phase1
        max_seq_len = args.max_seq_len_phase1
        num_datapoints = args.train_steps_phase1 * 256
        warmup_proportion = args.warmup_steps_phase1 / args.train_steps_phase1
        learning_rate = args.learning_rate_phase1
        phase2 = False
    if args.devices == "-1":
        devices = ["cpu"]
    else:
        devices = [*map(int, args.devices.split(","))]
    model = model.cuda(devices[0])
    loader_batch_size = device_batch_size * len(devices)
    # NOTE: Effective train_batch_size is:
    #       (device_batch_size * num_devices) * gradient_accumulation_steps
    #       For dataparallel we fetch loader_batch_size (device_batch_size * num_devices)
    #       which is the effective batch_size
    if args.optimizer == "adam":
        args.gradient_accumulation_steps = 1
    if args.gradient_accumulation_steps > 1:
        effective_batch_size = loader_batch_size * len(devices) * args.gradient_accumulation_steps
        print(f"loader_batch_size is {loader_batch_size}")
        print(f"Effective batch size for large batch training is {effective_batch_size}")
    if args.dataset == "books-wiki":
        loader = get_wiki_books_loader(loader_batch_size, args.num_workers, 8,
                                       shuffle=not args.testing, max_seq_len=max_seq_len,
                                       min_seq_len=30, truncate_strategy="truncate_second",
                                       mask_whole_words=args.mask_whole_words)
        max_steps = math.ceil(num_datapoints / len(loader.dataset) * len(loader))
    elif args.dataset == "owt":
        if args.train_strategy == "epoch" and args.dataset == "owt":
            raise ValueError("Epoch wise training not supported with OWT")
        loader = get_owt_loader(num_datapoints, loader_batch_size, args.num_workers, 8,
                                max_seq_len=max_seq_len,
                                min_seq_len=30, truncate_strategy="truncate_second",
                                mask_whole_words=args.mask_whole_words)
        max_steps = len(loader)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    optimizer, lr_scheduler, grad_scaler = get_optimizer(args.optimizer,
                                                         model, devices, warmup_proportion,
                                                         max_steps, learning_rate,
                                                         args.init_loss_scale)
    if args.optimizer == "lamb":
        optimizer.setup_fp32_params()
    elif args.optimizer == "adam":
        print("Will ignore gradient accumulation steps")
        args.gradient_accumulation_steps = 1
    else:
        raise ValueError(f"Unknown optmizer {args.optimizer}")

    if args.train_strategy == "epoch":
        num_epochs = math.ceil(max_steps / len(loader))
    elif args.train_strategy == "steps":
        num_epochs = 1
    criterion = BertPretrainingCriterion(config.vocab_size,
                                         sequence_output_is_dense=custom_model,
                                         nsp=args.next_sentence_prediction)

    update_function = BertUpdateFunction(grad_scaler, lr_scheduler,
                                         args.gradient_accumulation_steps)
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    trainer_params = {"gpus": devices, "cuda": True,
                      "seed": args.seed, "resume": True,
                      "use_prefetch": args.use_prefetch,
                      "metrics": ["loss", "scaled_loss", "total"], "val_frequency": 1,
                      "test_frequency": 5,
                      "log_frequency": 1 if args.testing else 5,
                      "max_epochs": num_epochs}
    trainer_name = "bert_trainer_" + ("phase2" if phase2 else "phase1")
    data = {"name": args.dataset, "train": loader.dataset}
    savedir = "-".join([args.model_name.replace("_", "-"), args.dataset])
    if args.mask_whole_words:
        savedir += "-whole-words"
    if args.savedir_suffix:
        savedir += f"-{args.savedir_suffix}"
    print(f"Output dir is {savedir}")
    trainer = Trainer(trainer_name, trainer_params, optimizer, model, data,
                      {"train": loader}, update_function, criterion,
                      savedir=savedir, logdir=savedir, ddp_params={},
                      extra_opts={"args": args.__dict__,
                                  "model_config": config.to_dict()})
    trainer._grad_scaler = grad_scaler
    trainer._lr_scheduler = lr_scheduler
    trainer.save_extra = lambda self: {"grad_scaler": self._grad_scaler.state_dict(),
                                       "lr_scheduler": self._lr_scheduler.state_dict()}
    trainer.load_extra = lambda self, saved_state: load_grad_scaler_and_scheduler(self, saved_state)
    desc = trainer.describe_hook("post_batch_hook")
    if not any("post_batch_progress" in x for x in desc):
        trainer.add_to_hook_at_end("post_batch_hook", functions.post_batch_progress)
    trainer.add_to_hook_at_end("post_batch_hook", partial(functions.update_metrics, quiet=True))
    if args.dataset == "books-wiki":
        trainer.add_to_hook_at_end("post_epoch_hook", lambda self: loader.dataset.reset())
    _save_checkpoint = partial(functions.post_batch_save_checkpoint, batch_num_for_saving=10000)
    trainer.add_to_hook_at_end("post_batch_hook", _save_checkpoint)
    trainer.add_to_hook_at_end("pre_training_hook", dump_extra_opts)
    # TODO:
    # 1. loader shuffle seed for deterministic load and save
    #    BUT if we resume and it should not shuffle from same seed
    if args.testing:
        trainer.test_loops(args.testing_iters)
    else:
        files = [*filter(lambda x: x.endswith("pth"), os.listdir(savedir))]
        if not any(f.startswith("checkpoint") for f in files):
            files.sort()
            if files:
                trainer._resume_path = Path(f"{savedir}/{files[-1]}")
        trainer.try_resume()
        trainer.train()


if __name__ == '__main__':
    main()
