import json
from multiprocessing import Pool
from pathlib import Path

import h5py
import nltk
import nltk.tokenize as nt
import numpy as np
import preprocess
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (BartTokenizer)
from utils import get_annotations, get_sentence_data, bb_intersection_over_union, coverage_computation
from nltk.tag import StanfordPOSTagger
from nltk import word_tokenize

project_dir = Path(__file__).resolve().parent.parent # VLT5
workspace_dir = project_dir.parent

def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx)

def get_datum(datum):
    data = []
    args = datum['args']

    img_id = datum['img_id']
    caption_id = datum['caption_id']
    img_controldata = datum['img_controldata']

    n_total_bboxes = datum['n_img_bboxes'] 
    n_cntl_bboxes = datum['n_cntl_bboxes']
    cntl_bboxes = datum['cntl_bboxes']
    coverage = datum['coverage']
    sentence = datum['sentence']

    if datum['ground_caption']:
        for i in range(args.ground_upsample):
            new_datum = {
                'uid': make_uid(img_id, 'ground_caption', i),
                'img_id': img_id,
                'caption_id': caption_id,
                'img_controldata': img_controldata,
                'img_source': f"{args.dataset_images}{img_id}.jpg",
                'task': 'ground_caption',
                'text_source': 'ground_caption',
                'sent': None,
                'label': None,
                'n_img_bboxes': n_total_bboxes,
                'n_cntl_bboxes': n_cntl_bboxes,
                'coverage': coverage,
                'cntl_bboxes': cntl_bboxes, 
                'sentence': sentence
            }
            data.append(new_datum)

    return data

class PretrainDataset(Dataset): 
    def __init__(self, split='train', rank=-1, topk=-1, verbose=True, args=None, is_train=True, mode='train', raw_dataset=None):

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.vg_classes = []
        with open(args.vg_objects_name) as f:
            for obj in f.readlines():
                self.vg_classes.append(obj.split(',')[0].lower().strip())

        self.vg_attrs = []
        with open(args.vg_attrs_name) as f:
            for attr in f.readlines():
                self.vg_attrs.append(attr.split(',')[0].lower().strip())

        self.postagger = StanfordPOSTagger(args.spos_model, args.spos_jar)

        self.mode = mode

        self.diverse_length = args.diverse_length
        self.add_verbs = args.add_verbs

        self.entities = 'entities'

        # check if image has entry.
        self.source_to_h5 = {
            'gt_flickr-30k': args.precomp_features
        }
        gt_source = 'gt_flickr-30k'
        f = self.source_to_h5[gt_source]
        if  isinstance(f, str):
            f = h5py.File(f, 'r')
            self.source_to_h5[gt_source] = f

        # Loading datasets to data
        self.source = [split]
        if self.verbose:
            print('Data source: ', self.source)

        self.img_ids_to_source = {}
        self.img_ids_to_control = {}

        losses = args.losses.split(',')

        data = []
        for img_source in self.source:
            data_info_path = img_source
            _data = (np.loadtxt(data_info_path)).astype(int)
            if self.verbose:
                print(f"Loaded {len(_data)} data from", img_source)
            datum = {}
            for img_id in _data:
                # Check that we have the detectron2 features
                try:
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    attr_ids = f[f'{img_id}/attr_id'][()]
                    boxes = f[f'{img_id}/boxes'][()]
                    if len(obj_ids)<1 and len(attr_ids)<1 and len(boxes)<1:
                        continue
                    n_total_bboxes = len(boxes)
                except Exception as e:
                    print(str(e))
                    continue

                try:
                    snt_data = get_sentence_data(self.args.entities_sentences + str(img_id) + '.txt')
                    ann_data = get_annotations(self.args.entities_annotations + str(img_id) + '.xml')

                    count = 0
                    for snt in snt_data:
                        cntl_bboxes = []
                        for phrase in snt['phrases']:
                            phrase_id = phrase['phrase_id']

                            # find the bboxes of the phrase id
                            if phrase_id in ann_data['boxes'].keys():
                                phrase_boxes = ann_data['boxes'][phrase_id]
                                for phrase_box in phrase_boxes:
                                    temp_box = np.array([phrase_box[0],phrase_box[1],phrase_box[2],phrase_box[3]]).astype(dtype='float32')
                                    # import ipdb; ipdb.set_trace()
                                    if len(cntl_bboxes) == 0:
                                        cntl_bboxes.append(temp_box)
                                    elif not np.any(np.all(temp_box == cntl_bboxes, axis=1)):
                                        cntl_bboxes.append(temp_box)

                            elif phrase_id in ann_data['scene']:
                                temp_scene = np.array([-1,-1,-1,-1]).astype(dtype='float32')
                                if len(cntl_bboxes) == 0:
                                    cntl_bboxes.append(temp_scene)
                                elif not np.any(np.all(temp_scene == cntl_bboxes, axis=1)):
                                    cntl_bboxes.append(temp_scene)
                        n_cntl_bboxes = len(cntl_bboxes)
                        if self.args.use_entities_data:
                            datum['img_id'] = img_id
                            datum['caption_id'] = str(img_id) + '.' + str(count)
                            datum['img_controldata'] = self.entities
                            self.img_ids_to_source[datum['img_id']] = args.dataset_images + str(datum['img_id']) + '.jpg'
                            self.img_ids_to_control[datum['img_id']] = self.entities
                            datum['args'] = args
                            datum['is_train'] = is_train
                            datum['caption_only'] = args.caption_only
                            datum['lm'] = 'lm' in losses
                            datum['qa'] = 'qa' in losses
                            datum['ground_caption'] = 'ground_caption' in losses
                            datum['refer'] = 'refer' in losses
                            datum['itm'] = 'itm' in losses
                            datum['caption'] = 'caption' in losses
                            datum['backbone'] = self.args.backbone
                            datum['n_img_bboxes'] = n_total_bboxes
                            datum['n_cntl_bboxes'] = n_cntl_bboxes
                            datum['cntl_bboxes'] = cntl_bboxes
                            datum['sentence'] = snt['sentence']
                            if n_total_bboxes > 0 and n_cntl_bboxes > 0:
                                tmp_coverage = coverage_computation(np.ones([int(boxes[0][2]), int(boxes[0][3])]), cntl_bboxes)
                                datum['coverage'] = tmp_coverage
                            else: 
                                # you should not be here! 
                                continue

                            data.append(datum.copy())
                            count += 1

                except Exception as e:
                    pass

                if self.args.use_ssa_data and ('train' in split):
                    count = 0
                    try:
                        with open(f"{args.ssa_dataset}{img_id}.json", "r") as outfile:
                            control_data = json.load(outfile)
                            
                        for control_signal in control_data:
                            postags = nltk.pos_tag(nt.word_tokenize(control_data[control_signal]['phrase']))
                            has_nouns = False
                            for tag in postags:
                                if 'NN' in tag[1]:
                                    has_nouns = True
                                    break
                            if not has_nouns:
                                tmp_cap = control_data[control_signal]['phrase']
                                print(f'NLTK: {tmp_cap}: Pos Tags {str(postags)}')
                                postags = self.pos_tagger.tag(word_tokenize(control_data[control_signal]['phrase']))
                                has_nouns = False
                                for tag in postags:
                                    if 'NN' in tag[1]:
                                        has_nouns = True
                                        break
                                if not has_nouns:
                                    print(f'STANFORD: {tmp_cap}: Pos Tags {str(postags)}')
                                    continue

                            cntl_bboxes = []
                            for obj in control_data[control_signal]['objects']:
                                if obj['object_id'] == 'scene':
                                    bbox = np.array([-1,-1,-1,-1]).astype(dtype='float32')
                                else:
                                    bbox = np.array([obj['xmin'],obj['ymin'],obj['xmax'],obj['ymax']]).astype(dtype='float32')
                                if len(cntl_bboxes) == 0:
                                    cntl_bboxes.append(bbox)
                                elif not np.any(np.all(bbox == cntl_bboxes, axis=1)):
                                    cntl_bboxes.append(bbox)
                            if len(cntl_bboxes) < 1:
                                continue

                            datum['img_id'] = img_id
                            datum['caption_id'] = control_signal
                            datum['img_controldata'] = f"{args.ssa_dataset}{img_id}.json"
                            self.img_ids_to_source[datum['img_id']] = args.dataset_images + str(datum['img_id']) + '.jpg'
                            self.img_ids_to_control[datum['img_id']] = args.ssa_dataset + str(datum['img_id']) + '.json'
                            datum['args'] = args
                            datum['is_train'] = is_train
                            datum['caption_only'] = args.caption_only
                            datum['lm'] = 'lm' in losses
                            datum['qa'] = 'qa' in losses
                            datum['ground_caption'] = 'ground_caption' in losses
                            datum['refer'] = 'refer' in losses
                            datum['itm'] = 'itm' in losses
                            datum['caption'] = 'caption' in losses
                            datum['backbone'] = self.args.backbone
                            datum['n_img_bboxes'] = n_total_bboxes
                            n_cntl_bboxes = len(cntl_bboxes)
                            datum['n_cntl_bboxes'] = n_cntl_bboxes
                            if n_total_bboxes > 0 and n_cntl_bboxes > 0:
                                tmp_coverage = coverage_computation(np.ones([int(boxes[0][2]), int(boxes[0][3])]), cntl_bboxes)
                                datum['coverage'] = tmp_coverage
                            else: 
                                continue

                            datum['cntl_bboxes'] = cntl_bboxes
                            datum['sentence'] = control_data[control_signal]['phrase']

                            data.append(datum.copy())
                    except Exception as e:
                        pass


        if self.verbose:
            print("# images:", len(data))

        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")
        
        with Pool(8) as pool:
            if self.verbose:
                data = [datum for data in tqdm(
                    pool.imap(get_datum, data), total=len(data), ncols=100, desc="Creating pretrainig data examples") for datum in data]
            else:
                data = [datum for data in pool.imap(
                    get_datum, data) for datum in data]
                
        self.data = data
        self.n_data = len(self.data)
        print(f"!!!The size of data for {split} is equal to {self.n_data}")

        if self.verbose and is_train:
            from collections import Counter
            task_counter = Counter()
            for datum in data:
                try:
                    task_counter.update([datum['task']])
                except KeyError:
                    print(datum)
                    exit()

            print(task_counter)
            for k, v in task_counter.items():
                print(k, f'{v/len(data)*100:.1f}%')

        if self.verbose:
            print("# examples:", len(data))

        self.n_boxes = args.n_boxes

        if 'bart' in self.args.backbone:
            capLenTokens = 5
            self.tokenizer = BartTokenizer.from_pretrained(args.backbone)
            additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1-capLenTokens, -1, -1)] + \
                    [f'<cap_len_level_{i}>' for i in range(capLenTokens, 0, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
            special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        out_dict = {}
        out_dict['args'] = self.args

        source = 'flickr-30k'
        gt_source = 'gt_flickr-30k'

        try:
            f = self.source_to_h5[gt_source]
            if  isinstance(f, str):
                f = h5py.File(f, 'r')
                self.source_to_h5[gt_source] = f
        except Exception as e:
            print(str(e))

        datum = self.data[idx]

        ###### Image ######
        datum = self.data[idx]
        img_id = datum['img_id']

        uid = datum['uid']
        out_dict['uid'] = uid
            
        if 'bart' in self.args.backbone:

            task = datum['task'] # ground caption

            loss_weight = 1

            # T5 Corrupt span
            if task == 'ground_caption':
                obj_ids = f[f'{img_id}/obj_id'][()]
                attr_ids = f[f'{img_id}/attr_id'][()]
                boxes = f[f'{img_id}/boxes'][()]
                img_h = f[f'{img_id}/img_h'][()]
                img_w = f[f'{img_id}/img_w'][()]

                try:
                    feats = np.array(f[f'{img_id}/features'], np.float32)
                    feats = torch.from_numpy(feats)
                except KeyError:
                    print(uid)
                    print(source)
                    print(img_id)
                    exit()

                captions = []
                all_objs = []
                all_attrs = []
                for obj_id, attr_id in zip(obj_ids, attr_ids):
                    obj = self.vg_classes[obj_id]
                    attr = self.vg_attrs[attr_id]

                    all_objs.append(obj)
                    all_attrs.append(attr)

                    caption = f'{attr} {obj}'
                    captions.append(caption)

                if self.args.use_entities_data or self.args.use_ssa_data:
                    cntl_caption = []
                    ml_boxes = []
                    cntl_obj_adj = []
                    cntl_signal = []
                    obj_adj_names = ''
                    for box_i in range(len(datum['cntl_bboxes'])):
                        if np.any(np.all(datum['cntl_bboxes'][box_i] == np.array([-1,-1,-1,-1]).astype(dtype='float32'), axis=0)):
                            ml_boxes.append(0)
                            obj_adj_names += ' ' + all_objs[0]
                        else:
                            iou = []
                            for det2_box in boxes:
                                iou.append(bb_intersection_over_union(datum['cntl_bboxes'][box_i],det2_box))
                            match_bbox = np.array(iou).argmax()
                            ml_boxes.append(np.array(iou).argmax())
                            if all_objs[match_bbox] not in obj_adj_names:
                                obj_adj_names += ' ' + all_objs[match_bbox]

                    ml_boxes = [*set(ml_boxes)]
                    if len(ml_boxes) > 0:
                        cntl_caption.append(datum['sentence'])
                        cntl_signal.append(ml_boxes)
                        cntl_obj_adj.append(obj_adj_names)

                visual_emb_boxes = cntl_signal
                if 0 not in visual_emb_boxes[0]:
                    visual_emb_boxes[0].append(0)
                visual_emb_boxes[0].sort()

                boxes = boxes[tuple(visual_emb_boxes)]
                out_dict['vis_feats'] = feats[tuple(visual_emb_boxes)]
                out_dict['boxes_raw'] = torch.from_numpy(boxes).clone()

                # Normalize the boxes (to 0 ~ 1)
                boxes[:, (0, 2)] /= img_w
                boxes[:, (1, 3)] /= img_h
                np.testing.assert_array_less(boxes, 1+1e-5)
                np.testing.assert_array_less(-boxes, 0+1e-5)
                boxes = torch.from_numpy(boxes)
                boxes.clamp_(min=0.0, max=1.0)
                out_dict['boxes'] = boxes


                if 0 in cntl_signal:
                    cntl_signal = [np.array(range(len(boxes)))]
                else:
                    cntl_signal = [np.array(range(1,len(boxes)))]

                prefix = "caption region:"
                if len(cntl_caption) > 0:
                    diverse = False
                    if self.diverse_length:
                        diverse = True
                    add_verbs = False
                    if self.add_verbs:
                        add_verbs = True
                    source_text, target_text, cntl_names, length_level = preprocess.ground_cntl_caption(
                        cntl_obj_adj, cntl_caption, cntl_signal, self.args.n_ground, prefix=prefix, diverse=diverse, add_verbs=add_verbs)
                else:
                    source_text, target_text, cntl_names = preprocess.ground_caption(
                        captions, self.args.n_ground, prefix=prefix, sort=False)
                                        
                sent = source_text

                loss_weight = self.args.ground_weight

            input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
            target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

            out_dict['input_ids'] = torch.LongTensor(input_ids)
            out_dict['input_length'] = len(input_ids)
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)
            out_dict['source_text'] = source_text
            out_dict['target_text'] = target_text
            out_dict['cntl_names'] = cntl_names
            out_dict['length_level'] = length_level
            out_dict['task'] = task
            out_dict['sent'] = sent
            out_dict['loss_weight'] = loss_weight
            out_dict['n_img_bboxes'] = datum['n_img_bboxes']
            out_dict['n_cntl_bboxes'] = datum['n_cntl_bboxes']
            out_dict['coverage'] = datum['coverage']

            return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        args = self.args


        max_bbox = 0
        for i, entry in enumerate(batch):
            if len(entry['boxes']) > max_bbox:
                max_bbox = len(entry['boxes'])
        V_L = max_bbox

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        feat_dim = batch[0]['vis_feats'].shape[-1]

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
        vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

        loss_weights = torch.ones(B, dtype=torch.float)

        sentences = []
        ans = []
        uids = []
        tasks = []

        source_text = []
        target_text = []
        cntl_names = []
        length_level = []

        n_img_boxes = []
        n_snt_boxes = []
        bbox_coverage = []
        boxes_raw = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            boxes[i,:len(entry['boxes'])] += entry['boxes']
            vis_feats[i,:len(entry['vis_feats'])] += entry['vis_feats']

            if 'ans' in entry:
                ans.append(entry['ans'])

            if 'task' in entry:
                tasks.append(entry['task'])

            sentences.append(entry['sent'])
            uids.append(entry['uid'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
            if 'target_text' in entry:
                target_text.append(entry['target_text'])
            if 'cntl_names' in entry:
                cntl_names.append(entry['cntl_names'])
            if 'length_level' in entry:
                length_level.append(entry['length_level'])

            if 'boxes_raw' in entry:
                boxes_raw.append(entry['boxes_raw'])


            if 'n_img_bboxes' in entry:
                n_img_boxes.append(entry['n_img_bboxes'])
            if 'n_cntl_bboxes' in entry:
                n_snt_boxes.append(entry['n_cntl_bboxes'])
            if 'coverage' in entry:
                bbox_coverage.append(entry['coverage'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

        assert 't5' in args.backbone or 'bart' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks
        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text
        batch_entry['cntl_names'] = cntl_names
        batch_entry['length_level'] = length_level
        batch_entry['input_ids'] = input_ids
        batch_entry['target_ids'] = target_ids
        batch_entry['boxes'] = boxes
        batch_entry['vis_feats'] = vis_feats
        batch_entry['loss_weights'] = loss_weights
        batch_entry['uid'] = uids
        batch_entry['sent'] = sentences
        batch_entry['n_img_bboxes'] = n_img_boxes
        batch_entry['snt_boxes'] = n_snt_boxes
        batch_entry['bbox_coverage'] = bbox_coverage
        batch_entry['boxes_raw'] = boxes_raw

        return batch_entry


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    verbose = (gpu == 0)
    
    dataset = PretrainDataset(
        split,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        is_train=(mode == 'train'),
        )

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    

    if mode == 'train':
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(sampler is None),
            num_workers=workers, 
            pin_memory=True, 
            sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, 
            pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    return loader