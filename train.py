import os
import torch

from argparse import ArgumentParser

from stablediff.params import params_simple
from stablediff.learner import tfdiffLearner
from stablediff.models import tfdiff_Simple
from stablediff.dataset import from_path_modulation_holdout

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _train_impl(replica_id, model, dataset, params, val_dataset=None):
    opt = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)
    learner = tfdiffLearner(
        params.log_dir,
        params.model_dir,
        model,
        dataset,
        opt,
        params,
        val_dataset=val_dataset,
    )
    learner.is_master = (replica_id == 0)
    learner.restore_from_checkpoint()
    learner.train(max_iter=params.max_iter)


def train(params):
    dataset, val_dataset = from_path_modulation_holdout(
        params,
        test_per_mod=int(getattr(params, "test_per_mod", 1)),
        mods=tuple(getattr(params, "test_mods", ["BPSK", "QPSK", "8PSK"])),
        split_seed=int(getattr(params, "split_seed", 42)),
    )
    device = torch.device('cpu', 0)
    model = tfdiff_Simple(params).to(device)
    _train_impl(0, model, dataset, params, val_dataset=val_dataset)

def main(args):
    params = params_simple
    if args.batch_size is not None:
        params.batch_size = args.batch_size
    if args.model_dir is not None:
        params.model_dir = args.model_dir
    if args.data_dir is not None:
        params.data_dir = args.data_dir
    if args.log_dir is not None:
        params.log_dir = args.log_dir
    if args.max_iter is not None:
        params.max_iter = args.max_iter
    if args.animate_training:
        params.animate_after_training = True
    if args.animation_out is not None:
        params.training_animation_out = args.animation_out
    params.test_per_mod = args.test_per_mod
    params.test_mods = args.test_mods
    params.split_seed = args.split_seed
    train(params)


# python train.py  --model_dir [model_dir] --data_dir [data_dir]
# HF_ENV_NAME=py38-202207 hfai python train.py --model_dir [model_dir] --data_dir [data_dir] --max_iter [iter_num] --batch_size [batch_size] -- -n [node_num] --force
if __name__ == '__main__':
    parser = ArgumentParser(
        description='train (or resume training) a tfdiff model')
    parser.add_argument('--model_dir', default=None,
                        help='directory in which to store model checkpoints and training logs')
    parser.add_argument('--data_dir', default=None, nargs='+',
                        help='space separated list of directories from which to read csi files for training')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--max_iter', default=None, type=int,
                        help='maximum number of training iteration')
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--animate_training', action='store_true',
                        help='generate training animation automatically when training ends')
    parser.add_argument('--animation_out', default=None,
                        help='output path for training animation (.gif or .mp4)')
    parser.add_argument('--test_per_mod', default=1, type=int,
                        help='number of held-out test samples per modulation')
    parser.add_argument('--test_mods', default=['BPSK', 'QPSK', '8PSK'], nargs='+',
                        help='modulations to hold out for the test set')
    parser.add_argument('--split_seed', default=42, type=int,
                        help='random seed used for train/test holdout selection')
    main(parser.parse_args())
