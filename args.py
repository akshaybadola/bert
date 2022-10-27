import os
import json
import sys
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--resume_dir", help="Resume training from directory")
    parser.add_argument("--model_name", help="The BERT model config")
    parser.add_argument("--model_config_file", help="The BERT model config")
    parser.add_argument("--dataset", help="dataset name, one of [books-wiki, owt]")
    parser.add_argument("--train_config_file", help="The training config")

    # Other parameters
    parser.add_argument("--mask_whole_words", action="store_true")
    parser.add_argument("--no_next_sentence_prediction", dest="next_sentence_prediction",
                        action="store_false")
    parser.add_argument("--device_batch_size_phase1", type=int)
    parser.add_argument("--train_strategy", default="epoch", help="One of [epochs, steps]")
    parser.add_argument("--use_prefetch", action="store_true")
    parser.add_argument("--optimizer", required=True,
                        help="Which optimizer to use. [adam, lamb]")
    parser.add_argument("--max_predictions_per_seq",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--learning_rate_phase1",
                        default=0.0,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate_phase2",
                        default=0.0,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_steps_phase1",
                        default=0.0,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for phase1. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_steps_phase2",
                        default=0.0,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for phase2. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--devices", default="0,1",
                        help="Comma separated devices. Use -1 for CPU")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=8,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Mixed precision training")
    parser.add_argument('--amp',
                        action='store_true',
                        help="Mixed precision training")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=int, default=1,
                        help='frequency of logging loss.')
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument('--start_phase2',
                        default=False,
                        action='store_true',
                        help="Whether to train with seq len 512")
    parser.add_argument('--resume_phase2',
                        action='store_true',
                        help="Whether to resume training with seq len 512")
    parser.add_argument('--train_steps_phase1',
                        type=int,
                        default=0,
                        help="Number of training steps in Phase1 - seq len 128")
    parser.add_argument('--train_steps_phase2',
                        type=int,
                        default=0,
                        help="Number of training steps in Phase2 - seq len 512")
    parser.add_argument('--init_loss_scale',
                        type=int,
                        default=2**20,
                        help="Initial loss scaler value")
    parser.add_argument('--steps_this_run', type=int, default=-1,
                        help='If provided, only run this many steps before exiting')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='number of DataLoader worker processes per rank')
    parser.add_argument("--savedir", default="saves")
    parser.add_argument("--logdir", default="logs")

    # optimizations controlled by command line arguments
    parser.add_argument("--no_dense_sequence_output",
                        dest="dense_sequence_output",
                        action='store_false',
                        help="Disable dense sequence output")
    parser.add_argument("--disable_jit_fusions",
                        default=False,
                        action='store_true',
                        help="Disable jit fusions.")
    parser.add_argument("--cuda_graphs",
                        default=False,
                        action='store_true',
                        help="Enable Cuda Graphs.")

    # misc params
    parser.add_argument("--testing", action="store_true")

    args = parser.parse_args()
    args.fp16 = args.fp16 or args.amp

    return check_args(args)


def check_args(args):
    assert (args.resume_dir or
            (args.model_name and
             args.model_config_file and
             args.dataset and
             args.train_config_file)), "Either resume dir or config files must be given"
    if args.resume_dir:
        try:
            with open(os.path.join(args.resume_dir, "args.json")) as f:
                _args = json.load(f)
            return _args
        except FileNotFoundError("args.json not found in resume dir"):
            sys.exit(1)
    else:
        model_configs = [x.split(".json")[0] for x in 
                         os.listdir(os.path.dirname(__file__) + "/configs")
                         if ("train" not in x.split(".json")[0] and
                             "eval" not in x.split(".json")[0])]
        assert args.model_name in model_configs, f"Model name {args.model_name} not in configs"
    return args
