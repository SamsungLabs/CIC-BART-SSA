from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from cic_data import get_loader
from diversity_metrics import calc_div_ngram
from packaging import version
from param import parse_args
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from trainer_base import TrainerBase


class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        from cic_model import VLBartPretraining

        model_kwargs = {}
        model_class = VLBartPretraining

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        if 'bart' in self.args.tokenizer:
            num_added_toks = 0
            capLenTokens = 5
            if config.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1-capLenTokens, -1, -1)] + \
                    [f'<cap_len_level_{i}>' for i in range(capLenTokens, 0, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

                config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{i}>' for i in range(100)])

        self.model = self.create_model(model_class, config, **model_kwargs)

        if 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.model.model.shared.num_embeddings + num_added_toks)

        self.model.tokenizer = self.tokenizer

        self.glove_vectors_f = args.glove_vectors

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)

        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def evaluate_diversity_best5(self):
        LOSSES_NAME = self.args.LOSSES_NAME

        epoch_results = {}
        for loss_name in LOSSES_NAME:
            epoch_results[loss_name] = 0.
            epoch_results[f'{loss_name}_count'] = 0

        uid2ans = {}

        self.model.eval()
        with torch.no_grad():
            if self.verbose:

                pbar = tqdm(total=len(self.test_loader), ncols=250)
            
            corpus_score_n1 = []
            corpus_score_n2 = []

            g_snt_dict = {}
            average_coverage = {}
            for _, batch in enumerate(self.test_loader):

                if self.args.distributed:
                    results = self.model.module.valid_step(batch)
                else:
                    results = self.model.valid_step(batch)

                generated_sentences = results['ground_caption']
                uids = batch['uid']
                coverage = batch['bbox_coverage']


                for idx in range(len(generated_sentences)):
                    if uids[idx] in g_snt_dict:
                        g_snt_dict[uids[idx]] += "###" + generated_sentences[idx]
                        average_coverage[uids[idx]] += coverage[idx]
                    else:
                        g_snt_dict[uids[idx]] = generated_sentences[idx]
                        average_coverage[uids[idx]] = coverage[idx]

                dist.barrier()

            self_cider_dict = []
            coverage_d1 = np.zeros([10])
            coverage_d2 = np.zeros([10])
            coverage_NoE = np.zeros([10])
            idx = 0
            for img_id in g_snt_dict:
                img_sentences = g_snt_dict[img_id].split('###')
                if len(img_sentences) == 5:
                    temp_dict = {}
                    temp_dict['image_id'] = int(img_id.split('_')[0])
                    temp_captions = []
                grouped_sentences_tokenized = []
                for snt in img_sentences:
                    if len(img_sentences) == 5:
                        temp_captions.append(snt)
                    grouped_sentences_tokenized += snt.split()
                if len(img_sentences) == 5:
                    cov_band = average_coverage[img_id]/5
                    cov_idx = int(np.floor(cov_band/10))

                    temp_dict['captions'] = temp_captions.copy()
                    self_cider_dict.append(temp_dict)

                    d1 = calc_div_ngram(temp_captions,1)
                    d2 = calc_div_ngram(temp_captions,2)
                    coverage_d1[cov_idx] += d1
                    coverage_d2[cov_idx] += d2
                    coverage_NoE[cov_idx] += 1
                    corpus_score_n1.append(calc_div_ngram(temp_captions,1))
                    corpus_score_n2.append(calc_div_ngram(temp_captions,2))

            distinct_n1_mean = np.array(corpus_score_n1).mean()
            distinct_n2_mean = np.array(corpus_score_n2).mean()

            print('Distinct 1 is ' + str(distinct_n1_mean) + '\n Distinct 2 is ' + str(distinct_n2_mean))

            import json
            with open(args.self_cider_path, 'w') as outfile:
                json.dump(self_cider_dict,outfile)

            try:
                fl = open(args.results_metrics, 'w')
                fl.write(f'Distinct 1: {distinct_n1_mean}\n')
                fl.write(f'Distinct 2: {distinct_n2_mean}\n')
                fl.close()
            except Exception as e:
                print(str(e))

            if self.verbose:
                pbar.close()
            dist.barrier()

            if 'qa' not in self.args.losses:
                uid2ans = None

            return epoch_results, uid2ans

def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building test loader at GPU {gpu}')
    test_loader = get_loader(
        args,
        split=args.test, mode='test', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.valid_topk,)

    trainer = Trainer(args, test_loader=test_loader, train=False)

    # trainer.train()
    trainer.evaluate_diversity_best5()

if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    if args.local_rank in [0, -1]:
        print(args)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
    if args.local_rank in [0, -1]:
        print(LOSSES_NAME)
    LOSSES_NAME.append('total_loss') # total loss

    args.LOSSES_NAME = LOSSES_NAME

    comments = []
    dsets = []
    if 'coco' in args.train:
        dsets.append('COCO')
    if 'vg' in args.train:
        dsets.append('VG')
    if 'flickr' in args.train:
        dsets.append('Flickr-30k')
    comments.append(''.join(dsets))
    if args.backbone:
        comments.append(args.backbone)
    comments.append(''.join(args.losses.split(',')))
    if args.comment != '':
        comments.append(args.comment)
    comment = '_'.join(comments)

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M')

    project_dir = Path(__file__).resolve().parent.parent

    if args.local_rank in [0, -1]:
        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'
        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)