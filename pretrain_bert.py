from typing import Final

import torch

import args as Args
import modeling
from schedulers import PolyWarmUpScheduler
from lamb_amp_opt.fused_lamb import FusedLAMBAMP

from dataloader import get_wiki_books_loader
from simple_trainer.models import UpdateFunction


def dict_to(tensors, device):
    return {k: v.to(device) for k, v in tensors.items()}


class BertUpdateFunction(UpdateFunction):
    def __init__(self, grad_scaler, lr_scheduler, fp16, all_reduce_fp16):
        self._train = True
        self._grad_scaler = grad_scaler
        self._lr_scheduler = lr_scheduler
        self._returns = ["loss", "scaled_loss"]

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, x: bool):
        self._train = x

    @property
    def returns(self):
        return self._returns

    def __call__(self, batch, criterion, model, optimizer):
        batch = {k: model.to_(v) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=(self.fp16 and not self.all_reduce_fp16)):
            prediction_scores, seq_relationship_score = model(input_ids=batch['input_ids'],
                                                              token_type_ids=batch['token_type_ids'],
                                                              attention_mask=batch['attention_mask'],
                                                              masked_lm_labels=batch['masked_lm_labels'])
            loss = criterion(prediction_scores, seq_relationship_score,
                             batch['masked_lm_labels'],
                             batch['next_sentence_labels'])
        self._grad_scaler.scale(loss).backward()
        
        if grad_accumulation_step:
            self._lr_scheduler.step()  # learning rate warmup
            self._grad_scaler.step(optimizer)
            self._grad_scaler.update()
            optimizer.zero_grad(set_to_none=True)


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


def get_optimizer(args, model, devices):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = FusedLAMBAMP(optimizer_grouped_parameters,
                             lr=args.learning_rate)
    lr_scheduler = PolyWarmUpScheduler(optimizer,
                                       warmup=args.warmup_proportion,
                                       total_steps=args.max_steps,
                                       base_lr=args.learning_rate,
                                       device=torch.device(devices[0]))
    grad_scaler = torch.cuda.amp.GradScaler(init_scale=args.init_loss_scale, enabled=args.fp16)
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


def main():
    args = Args.parse_arguments()
    sequence_output_is_dense = args.dense_sequence_output
    model, config = get_model_and_config(args.config_file, sequence_output_is_dense)

    from types import SimpleNamespace
    default_args = SimpleNamespace(device_batch_size=64,        # changed
                                   device_batch_size_phase2=8,  # changed
                                   warmup_proportion=0.2843,
                                   warmup_proportion_phase2=0.128,
                                   phase1_end_step=7038,
                                   train_steps_phase2=1563,  # check
                                   accumulate_gradients=True,
                                   gradient_accumulation_steps=128,  # check
                                   gradient_accumulation_steps_phase2=512,  # check
                                   allreduce_post_accumulation=True,
                                   allreduce_post_accumulation_fp16=True,
                                   learning_rate=6e-3,        # check
                                   learning_rate_phase2=4e-3,  # check
                                   num_workers=4,              # check, changed
                                   masking="static"            # check
                                   ) 

    # NOTE: For DataParallel, for DDP a different approach will have to be used
    device_batch_size = args.device_batch_size
    if args.devices == "-1":
        devices = [torch.device("cpu")]
    else:
        devices = [*map(lambda x: torch.device(f"cuda:{devices[x]}"), args.devices.split(","))]
    loader_batch_size = device_batch_size * len(devices)

    # NOTE: Effective train_batch_size is args.train_batch_size
    # but we fetch loader_batch_size // gradient_accumulation_steps from dataloader
    if args.gradient_accumulation_steps > 1:
        effective_batch_size = loader_batch_size * args.gradient_accumulation_steps

    loader = get_wiki_books_loader(loader_batch_size, args.num_workers, 8)
    # If allreduce_post_accumulation_fp16 is not set, Native AMP Autocast is
    # used along with FP32 gradient accumulation and all-reduce
    if args.fp16 and args.allreduce_post_accumulation_fp16:
        model.half()

    if not args.disable_jit_fusions:
        model = torch.jit.script(model)

    optimizer, lr_scheduler, grad_scaler = get_optimizer(args, model, devices)

    model.checkpoint_activations(args.checkpoint_activations)

    optimizer.setup_fp32_params()
    criterion = BertPretrainingCriterion(config.vocab_size,
                                         sequence_output_is_dense=sequence_output_is_dense)

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
