import argparse
import random

import numpy as np
import torch

import pprint
import yaml


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def get_optimizer(optim, verbose=False):
    # Bind the optimizer
    if optim == 'rms':
        if verbose:
            print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        if verbose:
            print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        if verbose:
            print("Optimizer: Using AdamW")
        # optimizer = torch.optim.AdamW
        optimizer = 'adamw'
    elif optim == 'adamax':
        if verbose:
            print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        if verbose:
            print("Optimizer: SGD")
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Data Splits
    parser.add_argument("--train", default='cic-bart-ssa/flickr/splits/train.txt')
    parser.add_argument("--valid", default='cic-bart-ssa/flickr/splits/val.txt')
    parser.add_argument("--test", default='cic-bart-ssa/flickr/splits/test.txt')
    parser.add_argument('--test_only', action='store_true')

    parser.add_argument('--dataset_images', type=str, default='flickr30k/flickr30k-images/', help='Flickr-30k images folder')
    parser.add_argument('--precomp_features', type=str, default='precomputed_features.h5', help='Precomputed features for Flickr-30k images')

    parser.add_argument('--submit', action='store_true')

    # Quick experiments
    parser.add_argument('--train_topk', type=int, default=-1)
    parser.add_argument('--valid_topk', type=int, default=-1)

    # Checkpoint
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--load', type=str, default='snap/pretrain/VLBart/Epoch30', help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--from_scratch', action='store_true')

    # CPU/GPU
    parser.add_argument("--multiGPU", action='store_const', default=True, const=True)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument('--local_rank', type=int, default=0)

    # Model Config
    parser.add_argument('--backbone', type=str, default='facebook/bart-base')
    parser.add_argument('--tokenizer', type=str, default=None)

    parser.add_argument('--feat_dim', type=float, default=2048)
    parser.add_argument('--pos_dim', type=float, default=4)

    parser.add_argument('--use_vision', default=True, type=str2bool)
    parser.add_argument('--use_vis_order_embedding', default=True, type=str2bool)
    parser.add_argument('--use_vis_layer_norm', default=True, type=str2bool)
    parser.add_argument('--individual_vis_layer_norm', default=False, type=str2bool)
    parser.add_argument('--share_vis_lang_layer_norm', action='store_true')

    parser.add_argument('--n_boxes', type=int, default=80)
    parser.add_argument('--max_n_boxes', type=int, default=80)
    parser.add_argument('--max_text_length', type=int, default=200)

    # Training
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--valid_batch_size', type=int, default=None)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument("--losses", default='ground_caption', type=str)

    parser.add_argument('--log_train_accuracy', action='store_true')

    parser.add_argument('--n_ground', type=int, default=1)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.3, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate',default=0.15, type=float)

    # Inference
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--gen_max_length', type=int, default=200)

    # Data
    parser.add_argument('--caption_only', action='store_true')

    parser.add_argument('--do_lower_case', action='store_true')

    parser.add_argument('--prefix', type=str, default=None)

    # Pretraining
    parser.add_argument('--ground_upsample', type=int, default=1)
    parser.add_argument('--ground_weight', type=int, default=1)

    parser.add_argument("--classifier", action='store_true')

    # Controlled Captioning
    parser.add_argument("--use_entities_data", action='store_true', help='Use original entities data')
    parser.add_argument("--use_ssa_data", action='store_true', help='Use SSA data')

    # Visual genome objects/attribute classes names
    parser.add_argument("--vg_objects_name", default='vlt5_vocab/objects_vocab.txt', type=str)
    parser.add_argument("--vg_attrs_name", default='vlt5_vocab/attributes_vocab.txt', type=str)

    # Stanford pos tagger
    parser.add_argument("--spos_jar", default='stanford-postagger/stanford-postagger.jar', type=str)
    parser.add_argument("--spos_model", default='stanford-postagger/models/english-bidirectional-distsim.tagger', type=str)

    parser.add_argument("--glove_vectors", default='vectors/glove.6B.300d.txt', type=str)
    
    parser.add_argument("--entities_annotations", default='flickr30k/Annotations/', type=str)
    parser.add_argument("--entities_sentences", default='flickr30k/Sentences/', type=str)

    parser.add_argument("--cntl_dataset_dir_original", default='', type=str) # original
    parser.add_argument("--ssa_dataset", default="flickr30k/cntlImgCaptioning_Dataset/", type=str, help='Flickr-30k SSA data') 
    parser.add_argument("--original_control_dataset", default="ssa/", type=str, help='Flickr-30k data in ASG json format') 

    parser.add_argument('--diverse_length', action='store_true', help='Activate to randomly select the fine grained length value')
    parser.add_argument('--add_verbs', action='store_true', help='Activate for including the ground truth verbs in the control signal')
    
    # Etc.
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument("--dry", action='store_true')

    # Evaluation output files
    parser.add_argument("--results_csv", default="", type=str)
    parser.add_argument("--results_metrics", default="", type=str)
    parser.add_argument("--results_gruen", default="", type=str)
    parser.add_argument("--self_cider_path", default="", type=str)



    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    # For interative engironmnet (ex. jupyter)
    else:
        args = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

    # Bind optimizer class.
    verbose = False
    args.optimizer = get_optimizer(args.optim, verbose=verbose)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


if __name__ == '__main__':
    args = parse_args(True)
