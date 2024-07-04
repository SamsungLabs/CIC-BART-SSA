# coding=utf-8

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from detectron2_proposal_maxnms import DIM, NUM_OBJECTS, collate_fn, extract
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Flickr30KDataset(Dataset):
    def __init__(self, image_dir, args):
        self.image_dir = image_dir
        self.image_path_list = list(tqdm(image_dir.iterdir()))
        self.n_images = len(self.image_path_list)
        self.originalSentences = args.cntl_dataset_dir_original

        # self.transform = image_transform

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        try:
            image_path = self.image_path_list[idx]
            image_id = image_path.stem
            img = cv2.imread(str(image_path))
            (h,w,c) = img.shape[:3]

            total_bboxes = []
            bbox = np.array([0.,0.,w,h]).astype(dtype='float32')
            total_bboxes.append(bbox)
            bboxes = []
            bbox = str(0) + ' ' + str(0) + ' ' + str(w) + ' ' + str(h)
            bboxes.append(bbox)


            try:
                with open(f"{self.originalSentences}{image_id}.json", "r") as outfile:
                            original_data = json.load(outfile)
                for control_signal in original_data:
                    objects = original_data[control_signal]['objects']
                    for obj in objects:
                        if obj["name"] != "scene":
                            bbox = str(obj['xmin']) + ' ' + str(obj['ymin']) + ' ' + str(obj['xmax']) + ' ' + str(obj['ymax'])
                            if bbox not in bboxes:
                                bboxes.append(bbox)
                                bbox = np.array([obj['xmin'],obj['ymin'],obj['xmax'],obj['ymax']]).astype(dtype='float32')
                                total_bboxes.append(bbox)
            except Exception as e:
                print(str(e))


            return {
                'img_id': image_id,
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
    parser.add_argument('--flickrroot', type=str,
                        default='/group-volume/IVU/flickr30k/')
    parser.add_argument("--cntl_dataset_dir_original", default='/group-volume/IVU/flickr30k/cntlImgCaptioning_flickr30kCaptions/', type=str)
    parser.add_argument('--split', type=str, default='flickr30k', choices=['flickr30k', 'trainval', 'test2017', 'test2018'])
    # parser.add_argument('')

    args = parser.parse_args()

    SPLIT2DIR = {
        'flickr30k': 'flickr30k_images',
    }

    flickr_dir = Path(args.flickrroot).resolve()
    flickr_img_dir = flickr_dir.joinpath('flickr30k-images/')
    # flickr_img_dir = flickr_dir.joinpath('flickr30k_images/').joinpath(SPLIT2DIR[args.split])

    dataset_name = 'Flickr30K'

    out_dir = flickr_dir.joinpath('features')
    if not out_dir.exists():
        out_dir.mkdir()

    print('Load images from', flickr_img_dir)
    print('# Images:', len(list(flickr_img_dir.iterdir())))

    dataset = Flickr30KDataset(flickr_img_dir, args)

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    output_fname = out_dir.joinpath(f'{args.split}_GTboxes_v2.h5')
    print('features will be saved at', output_fname)

    desc = f'{dataset_name}_{args.split}_{(NUM_OBJECTS, DIM)}'

    extract(output_fname, dataloader, desc)
