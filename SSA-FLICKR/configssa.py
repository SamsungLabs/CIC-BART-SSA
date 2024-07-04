import argparse
import yaml
import pprint
import random
import numpy as np

def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    # coco captions amrs
    parser.add_argument("--caption_amrs", default='caption_amrs.txt')
    parser.add_argument("--flickr_sentences", default='flickr_annotations_folder/Sentences/')
    parser.add_argument("--flickr_annotations", default='flickr_annotations_folder/Annotations/')

    # output files
    parser.add_argument("--sampledamrs", default='vgamrs.txt')
    parser.add_argument("--meta_amrs", default='meta-vgamrs.txt')
    parser.add_argument("--nodevisualgroundinfo", default='vgamr_bboxes/')

    # probank
    parser.add_argument("--propbank", default='propbank-frames-folder/frames')

    # glove vectors file
    parser.add_argument("--glove", default='glove_vectors_folder/glove.6B.300d.txt')

    # SSA parameters
    parser.add_argument("--glove_threshold", default=0.5, type=float)
    parser.add_argument("--seed", default=10, type=int)

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