# coding=utf-8

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from detectron2_proposal_maxnms import DIM, NUM_OBJECTS, collate_fn, extract
from torch.utils.data import DataLoader, Dataset


class COCODataset(Dataset):
    def __init__(self, image_dir, args):
        self.image_dir = image_dir
        # self.image_path_list = list(tqdm(image_dir.iterdir()))
        # self.n_images = len(self.image_path_list)
        self.coco_entities = args.coco_entities

        f_cocoentities = open(args.coco_entities)
        self.annotations = json.load(f_cocoentities)

        self.data_indices = []
        for img_id in self.annotations:
            self.data_indices.append(img_id)


        # self.transform = image_transform

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        try:
            img_id = self.data_indices[idx]
            image_path = self.image_dir.joinpath(img_id.zfill(12) + '.jpg')
            img = cv2.imread(str(image_path))
            (h,w,c) = img.shape[:3]

            total_bboxes = []
            bbox = np.array([0.,0.,w,h]).astype(dtype='float32')
            total_bboxes.append(bbox)
            bboxes = []
            bbox = str(0) + ' ' + str(0) + ' ' + str(w) + ' ' + str(h)
            bboxes.append(bbox)

            img_ann = self.annotations[img_id]
            for caption in img_ann:
                caption_ann = img_ann[caption]
                for det_id in caption_ann['detections']:
                    for bbox_id in caption_ann['detections'][det_id]:
                        bbox = str(bbox_id[1][0]) + ' ' + str(bbox_id[1][1]) + ' ' + str(bbox_id[1][2]) + ' ' + str(bbox_id[1][3])
                        if bbox not in bboxes:
                            bboxes.append(bbox)
                            bbox = np.array([bbox_id[1][0],bbox_id[1][1],bbox_id[1][2],bbox_id[1][3]]).astype(dtype='float32')
                            total_bboxes.append(bbox)

            return {
                'img_id': img_id,
                'img': img,
                'bboxes': total_bboxes
            }
        except Exception as e:
            print(str(e))
            return {'img_id': None,
                    'img': None,
                    'bboxes': None}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
    parser.add_argument('--cocoroot', type=str,
                        default='MSCOCO/images/train2017/', help='MSCOCO images folder')
    parser.add_argument("--coco_entities", default='coco_entities/coco_entities_release.json', type=str, help='COCO entities dataset')
    parser.add_argument('--split', type=str, default='coco', choices=[ 'trainval', 'test2017', 'test2018', 'coco'])
    # parser.add_argument('')

    args = parser.parse_args()

    SPLIT2DIR = {
        'coco': 'coco_images',
    }

    coco_dir = Path(args.cocoroot).resolve()

    dataset_name = 'COCO'

    out_dir = coco_dir.joinpath('features')
    if not out_dir.exists():
        out_dir.mkdir()

    print('Load images from', coco_dir)
    print('# Images:', len(list(coco_dir.iterdir())))

    dataset = COCODataset(coco_dir, args)

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, collate_fn=collate_fn, num_workers=80)

    output_fname = out_dir.joinpath(f'{args.split}_coco_precomputed_features.h5')
    print('features will be saved at', output_fname)

    desc = f'{dataset_name}_{args.split}_{(NUM_OBJECTS, DIM)}'

    extract(output_fname, dataloader, desc)
