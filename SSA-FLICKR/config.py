import argparse
import yaml
import pprint
import random
import numpy as np

def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    # flickr30k entities
    parser.add_argument("--flickr_sentences", default='C:/Users/k.basioti/Documents/CIC/AMR/Sentences/')
    parser.add_argument("--flickr_annotations", default='C:/Users/k.basioti/Documents/CIC/AMR/Annotations/')

    # SSA augmentation
    parser.add_argument("--gen_snts", default='vgAMRtoText.txt')
    parser.add_argument("--sampled_amrs", default='vgAMRs.txt')
    parser.add_argument("--vgamr_nodes", default='vgamr_bboxes/')
    parser.add_argument("--output", default='output/')
    parser.add_argument("--gruen_threshold", default=0.7, type=float)
    parser.add_argument("--seed", default=0, type=int)
    # glove parameters
    parser.add_argument("--glove", default='glove_vectors_folder/glove.6B.300d.txt')

    if parse:
        args = parser.parse_args()
    else:
        args = parser.parse_known_args()[0]

    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

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