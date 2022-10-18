from typing import Final
from types import SimpleNamespace
import json
import math

import torch

import args as Args
from bert import modeling
from schedulers import PolyWarmUpScheduler
from lamb_amp_opt.fused_lamb import FusedLAMBAMP

from dataloader import get_wiki_books_loader
from simple_trainer.helpers import SimpleUpdateFunction
from simple_trainer.trainer import Trainer
from simple_trainer import functions


def dict_to(tensors, device):
    return {k: v.to(device) for k, v in tensors.items()}


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
            prediction_scores, seq_relationship_score = model(input_ids=batch['input_ids'],
                                                              token_type_ids=batch['token_type_ids'],
                                                              attention_mask=batch['attention_mask'],
                                                              masked_lm_labels=batch['masked_lm_labels'])
            loss = criterion(prediction_scores, seq_relationship_score,
                             batch['masked_lm_labels'],
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

    def __init__(self, vocab_size, sequence_output_is_dense=False):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size
        self.sequence_output_is_dense = sequence_output_is_dense

    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
        if self.sequence_output_is_dense:
            # prediction_scores are already dense
            masked_lm_labels_flat = masked_lm_labels.view(-1)
            mlm_labels = masked_lm_labels_flat[masked_lm_labels_flat != -1]
            masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss


def get_model_and_config(config_file, sequence_output_is_dense):
    config = modeling.BertConfig.from_json_file(config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = modeling.BertForPreTraining(config, sequence_output_is_dense=sequence_output_is_dense)
    return model, config


def get_optimizer(optimizer_name, model, devices, warmup_proportion,
                  max_steps, learning_rate, init_loss_scale):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    if optimizer_name == "lamb":
        optimizer = FusedLAMBAMP(optimizer_grouped_parameters,
                                 lr=learning_rate)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                     lr=learning_rate)
    else:
        raise ValueError(f"Unknown optmizer {optimizer_name}")
    lr_scheduler = PolyWarmUpScheduler(optimizer,
                                       warmup=warmup_proportion,
                                       total_steps=max_steps,
                                       base_lr=learning_rate,
                                       device=torch.device(devices[0]))
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


def load_grad_scaler(self, saved_state):
    self._grad_scaler.load_state_dict(saved_state["grad_scaler"])


def save_checkpoint(self, **kwargs):
    batch_num = kwargs["batch_num"]
    if not (batch_num+1) % 100000:
        prefix = f"{batch_num:06}_"
        save_name = f"{prefix}_{self.checkpoint_name}"
        self.logger.info(f"Saving to {save_name}")
        self._save(save_name)


def main():
    args = Args.parse_arguments()
    with open("train_config.json") as f:
        default_config = SimpleNamespace(**json.load(f))

    for k, v in default_config.__dict__.items():
        if k in args.__dict__ and not args.__dict__[k]:
            args.__dict__[k] = v
    sequence_output_is_dense = args.dense_sequence_output
    model, config = get_model_and_config(args.model_config_file, sequence_output_is_dense)
    model.model_name = "bert_base"

    # NOTE: For DataParallel, for DDP a different approach will have to be used
    if args.start_phase2 or args.resume_phase2:
        max_seq_len = 512
        device_batch_size = 56
        max_steps = args.train_steps_phase2
        warmup_proportion = args.warmup_steps_phase2 / args.train_steps_phase2
        learning_rate = args.learning_rate_phase2
        phase2 = True
    else:
        device_batch_size = 480
        max_seq_len = 128
        max_steps = args.train_steps_phase1
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
    if args.gradient_accumulation_steps > 1:
        effective_batch_size = loader_batch_size * len(devices) * args.gradient_accumulation_steps
        print(f"Effective batch size is {effective_batch_size}")
    loader = get_wiki_books_loader(loader_batch_size, args.num_workers, 8,
                                   shuffle=not args.testing, max_seq_len=max_seq_len,
                                   min_seq_len=30, truncate_stragety="truncate_second")

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

    num_epochs = math.ceil(max_steps * 256 / len(loader) / loader.batch_size)
    criterion = BertPretrainingCriterion(config.vocab_size,
                                         sequence_output_is_dense=sequence_output_is_dense)

    update_function = BertUpdateFunction(grad_scaler, lr_scheduler,
                                         args.gradient_accumulation_steps)
    torch.autograd.set_detect_anomaly(True)
    trainer_params = {"gpus": devices, "cuda": True,
                      "seed": args.seed, "resume": True,
                      "metrics": ["loss", "scaled_loss", "total"], "val_frequency": 1,
                      "test_frequency": 5, "log_frequency": 5, "max_epochs": num_epochs}
    trainer_name = "bert_trainer_" + ("phase2" if phase2 else "phase1")
    data = {"name": "books-wiki", "train": loader.dataset}
    trainer = Trainer(trainer_name, trainer_params, optimizer, model, data,
                      {"train": loader}, update_function, criterion,
                      args.savedir, args.logdir, ddp_params={},
                      extra_opts={"args": args.__dict__})
    trainer._grad_scaler = grad_scaler
    # trainer.save_optimizer = lambda : None
    trainer.save_extra = lambda self: {"grad_scaler": self._grad_scaler.state_dict()}
    trainer.load_extra = lambda self, saved_state: load_grad_scaler(self, saved_state)
    desc = trainer.describe_hook("post_batch_hook")
    if not any("post_batch_progress" in x for x in desc):
        trainer.add_to_hook_at_end("post_batch_hook", functions.post_batch_progress)
    trainer.add_to_hook_at_end("post_epoch_hook", lambda self: loader.dataset.reset())
    trainer.add_to_hook_at_end("post_batch_hook", save_checkpoint)
    # TODO:
    # 1. loader shuffle seed for deterministic load and save
    #    BUT if we resume and it should not shuffle from same seed
    # 2. Save on steps instead of epochs (maybe)
    # trainer.test_loops()
    trainer.train()


if __name__ == '__main__':
    main()
