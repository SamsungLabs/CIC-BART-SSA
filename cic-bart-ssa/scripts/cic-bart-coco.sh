#!/bin/bash
chmod +x cic-bart-ssa/coco/cic.py
cd cic-bart-ssa
conda activate <the cic-bart-ssa conda environment>

python -m torch.distributed.launch \
    coco/cic.py  \
        --distributed --multiGPU --fp16 --use_entities_data --use_ssa_data \
        --num_workers 80 \
        --load <VL-BART pretrained checkpoint> \
        --ssa_dataset <SSA dataset folder> \
        --output <your output folder location> \
        --epochs 20 \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --backbone 'facebook/bart-base' \
        --individual_vis_layer_norm False \
        --num_beams 5 \
        --batch_size 80 \