from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from dist_utils import reduce_dict
from noun_iou import NounIoU
from packaging import version
from param import parse_args
from speaksee.evaluation import Bleu, Cider, Meteor, PTBTokenizer, Rouge, Spice
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from utils import LossMeter
from nltk.tag import StanfordPOSTagger
from nltk import word_tokenize


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
                # include the spatial and length control tokens
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1-capLenTokens, -1, -1)] + \
                    [f'<cap_len_level_{i}>' for i in range(capLenTokens, 0, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

                config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{i}>' for i in range(100)])

        self.model = self.create_model(model_class, config, **model_kwargs)

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

    def train(self):
        LOSSES_NAME = self.args.LOSSES_NAME

        if self.verbose:
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
            best_eval_loss = 9595.

            src_dir = Path(__file__).resolve().parent
            src_dir = str(src_dir)

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        for epoch in range(self.args.epochs):
            if self.start_epoch is not None:
                epoch += self.start_epoch
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            # Train
            self.model.train()

            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=250)

            epoch_results = {}
            for loss_name in LOSSES_NAME:
                epoch_results[loss_name] = 0.
                epoch_results[f'{loss_name}_count'] = 0

            for _, batch in enumerate(self.train_loader):

                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']

                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

                if self.args.fp16 and _use_native_amp:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()
                # self.model.zero_grad()
                for param in self.model.parameters():
                    param.grad = None
                global_step += 1

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose:
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'

                    for _, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.verbose:
                pbar.close()

            dist.barrier()

            dist.barrier()

            # Validation
            _, valid_uid2ans = self.evaluate_epoch(epoch=epoch)

            if self.verbose:
                valid_loss = results['loss']

            if 'qa' in self.args.losses:
                dset2score, dset2cnt, score, cnt = self.val_loader.dataset.evaluator.evaluate(valid_uid2ans)

                if len(dset2score) == 0:
                    dset2score = {'vqa': 0, 'gqa': 0, 'visual7w': 0}
                    dset2cnt = {'vqa': 1, 'gqa': 1, 'visual7w': 1}
                    cnt = 3
                    score = 0

                dset2score = reduce_dict(dset2score, average=False)
                dset2cnt = reduce_dict(dset2cnt, average=False)
                score_cnt_dict = reduce_dict({'score': score, 'cnt': cnt}, average=False)

                if self.args.gpu == 0:
                    score = score_cnt_dict['score']
                    cnt = score_cnt_dict['cnt']
                    accu = score / cnt
                    dset2accu = {}
                    for dset in dset2cnt:
                        dset2accu[dset] = dset2score[dset] / dset2cnt[dset]
                    accu_str = "Overall QA Acc %0.4f" % (accu)
                    sorted_keys = sorted(dset2accu.keys())
                    for key in sorted_keys:
                        accu_str += ", %s Acc %0.4f" % (key, dset2accu[key])
                    print(accu_str)
                    accu_str += '\n\n'

            dist.barrier()

            if self.verbose:
                # Save
                if valid_loss < best_eval_loss:
                    best_eval_loss = valid_loss
                self.save("Epoch%02d" % (epoch + 1))

            dist.barrier()

    def evaluate_epoch(self, epoch):
        LOSSES_NAME = self.args.LOSSES_NAME

        log_file = self.args.output + 'log_file.txt'

        epoch_results = {}
        for loss_name in LOSSES_NAME:
            epoch_results[loss_name] = 0.
            epoch_results[f'{loss_name}_count'] = 0

        uid2ans = {}

        self.model.eval()
        with torch.no_grad():
            if self.verbose:
                loss_meter = LossMeter()
                loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]

                pbar = tqdm(total=len(self.val_loader), ncols=250)

            bleu_4 = []
            meteor = []; rouge = []; cider = []; spice = []

            noun_iou = NounIoU(pre_comp_file=self.glove_vectors_f)

            scores_iou = []
            m_scores_iou = []
            m_scores_iou_real = []

            m_scores_hal = []
            m_scores_hal_real = []

            m_scores_mis = []
            m_scores_mis_real = []

            coverage = []

            for _, batch in enumerate(self.val_loader):

                if self.args.distributed:
                    results = self.model.module.valid_step(batch)
                else:
                    results = self.model.valid_step(batch)

                generated_sentences = results['ground_caption']
                postags = pos_tagger.tag_sents([word_tokenize(sent) for sent in generated_sentences])
                target_sentences = batch['target_text']
                cntl_names = batch['cntl_names']

                # IoU, Missing Nouns/Adjs, Hallucinations.
                for idx in range(len(generated_sentences)):
                    if batch['bbox_coverage'][idx] == -1:
                        continue

                    score = noun_iou.score(target_sentences[idx], generated_sentences[idx])
                    scores_iou.append(score)

                    try:
                        score = noun_iou.score_modified(cntl_names[idx], generated_sentences[idx], postags=postags[idx])
                        t_iou_score, t_gtmi_percent, t_prmi_percent, t_r_iou_score, t_r_gtmi_percent, t_r_prmi_percent, _, _ = score

                        m_scores_iou.append(t_iou_score)
                        m_scores_iou_real.append(t_r_iou_score)

                        m_scores_hal.append(t_prmi_percent)
                        m_scores_hal_real.append(t_r_prmi_percent)

                        m_scores_mis.append(t_gtmi_percent)
                        m_scores_mis_real.append(t_r_gtmi_percent)

                        coverage.append(batch['bbox_coverage'][idx])

                    except Exception as e:
                        print(str(e))

                gts_t = PTBTokenizer.tokenize(target_sentences)
                gen_t = PTBTokenizer.tokenize(generated_sentences)

                val_bleu, _ = Bleu(n=4).compute_score(gts_t, gen_t)
                bleu_4.append(val_bleu[3])
                    
                val_meteor, _ = Meteor().compute_score(gts_t, gen_t)
                meteor.append(val_meteor)

                val_rouge, _ = Rouge().compute_score(gts_t, gen_t)
                rouge.append(val_rouge)

                val_cider, _ = Cider().compute_score(gts_t, gen_t)
                cider.append(val_cider)

                val_spice, _ = Spice().compute_score(gts_t, gen_t)
                spice.append(val_spice)

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose:
                    desc_str = f'Valid Epoch {epoch} |'
                    for _, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):
                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)
                dist.barrier()

            if self.verbose:
                pbar.close()
            dist.barrier()

            if 'qa' not in self.args.losses:
                uid2ans = None

            try:
                fl = open(log_file, 'a')
                fl.write(f"Epoch {epoch}\n")
                fl.write(f'Test Bleu 4: {np.array(bleu_4).mean()}\n')
                fl.write(f'Test Meteor: {np.array(meteor).mean()}\n')
                fl.write(f'Test Rouge: {np.array(rouge).mean()}\n')
                fl.write(f'Test Cider: {np.array(cider).mean()}\n')
                fl.write(f'Test Spice: {np.array(spice).mean()}\n')
                fl.write(f'Ours Noun IoU {np.mean(m_scores_iou)}\n')
                fl.write(f'Noun Hallucinations {np.mean(m_scores_hal)}\n')
                fl.write(f'Missing Nouns {np.mean(m_scores_mis)}\n')
                fl.write('\n\n\n')
                fl.close()
            except Exception as e:
                print(str(e))

            return epoch_results, uid2ans

def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building train loader at GPU {gpu}')
    train_loader = get_loader(
        args,
        split=args.train, mode='train', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.train_topk,)

    print(f'Building val loader at GPU {gpu}')
    val_loader = get_loader(
        args,
        split=args.test, mode='val', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.valid_topk,)
    
    trainer = Trainer(args, train_loader, val_loader, train=True)

    trainer.train()

if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    if args.local_rank in [0, -1]:
        print(args)

    print('Init Stanford POS tagger')
    pos_tagger = StanfordPOSTagger(args.spos_model, args.spos_jar)

    from cic_data import get_loader

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

    main_worker(args.local_rank, args)
