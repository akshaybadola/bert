from typing import Final

import torch

import args as Args
import modeling
from schedulers import PolyWarmUpScheduler
from lamb_amp_opt.fused_lamb import FusedLAMBAMP
import dataprep


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


def main():
    args = Args.parse_arguments()
    sequence_output_is_dense = args.sequence_output_is_dense
    model, config = get_model_and_config(args.config_file, sequence_output_is_dense)

    data = dataprep.load_saved_data()
    # If allreduce_post_accumulation_fp16 is not set, Native AMP Autocast is
    # used along with FP32 gradient accumulation and all-reduce
    if args.fp16 and args.allreduce_post_accumulation_fp16:
        model.half()

    if not args.disable_jit_fusions:
        model = torch.jit.script(model)

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
                                       device=device)
    grad_scaler = torch.cuda.amp.GradScaler(init_scale=args.init_loss_scale, enabled=args.fp16)

    model.checkpoint_activations(args.checkpoint_activations)

    optimizer.setup_fp32_params()
    criterion = BertPretrainingCriterion(config.vocab_size, sequence_output_is_dense=sequence_output_is_dense)
