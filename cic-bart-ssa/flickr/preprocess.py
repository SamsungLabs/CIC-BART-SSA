import random
from copy import deepcopy

import nltk
import nltk.tokenize as nt
import numpy as np
import torch


def corrupt_spans(text, mask_ratio=0.15, prefix=None):
    """T5-style Masked Language Modeling with corrupted span prediction
    Args:
        text

    Returns:
        source_text (masked_text)
        target_text

    Ex) (in vocab ids)
    input
        In this tutorial, we’ll explore how to preprocess your data using Transformers. The main tool for this is what we call a tokenizer.

    masked_text
        <extra_id_0> this tutorial, we’ll explore how to preprocess your data <extra_id_1> Transformers. The main tool for this is what <extra_id_2> call a tokenizer.
    target_text
    """

    tokens = text.split()

    n_tokens = len(tokens)

    n_mask = int(max(mask_ratio * n_tokens, 1))
    mask_indices = torch.randperm(n_tokens)[:n_mask].sort().values

    assert len(mask_indices) > 0, text

    mask_indices = mask_indices.tolist()
    span = [mask_indices[0], mask_indices[0]+1]
    spans = []

    for i, mask_index in enumerate(mask_indices):
        # if current mask is not the last one & the next mask is right after current mask
        if i < len(mask_indices) - 1 and mask_indices[i+1] == mask_index + 1:
            contiguous = True
        else:
            contiguous = False

        if contiguous:
            span[1] += 1

        else:
            # non contiguous -> output current span
            spans.append(span)
            # if current mask is not the last one -> create next span
            if i < len(mask_indices) - 1:
                span = [mask_indices[i+1], mask_indices[i+1]+1]

    masked_tokens = deepcopy(tokens)

    target_tokens = []
    cum_span_length = 0
    for i, span in enumerate(spans):
        start, end = span

        masked_tokens[start-cum_span_length+i: end -
                      cum_span_length+i] = [f'<extra_id_{i}>']

        target_tokens.append(f'<extra_id_{i}>')
        target_tokens.extend(tokens[start:end])

        cum_span_length += (end - start)

    # target_tokens.append(f'<extra_id_{i+1}>')
    # target_tokens.append(f'</s>')

    masked_text = " ".join(masked_tokens)

    if prefix is None:
        source_text = masked_text
    else:
        source_text = f"{prefix} {masked_text}"

    target_text = " ".join(target_tokens)

    return source_text, target_text


def corrupt_bart(input_text, mask_ratio=0.30, prefix="denoise text:"):
    """BART-style Masked Language Modeling with corrupted span prediction
    Args:
        text

    Returns:
        source_text (masked_text)
        target_text

    Ex) (in vocab ids)
    input
        In this tutorial, we’ll explore how to preprocess your data using Transformers. The main tool for this is what we call a tokenizer.

    masked_text
        denoise text: In <mask> we’ll explore how to preprocess your data <mask> Transformers. <mask> main <mask> for this is what we <mask> a tokenizer.
    target_text
        same is input text
    """

    tokens = input_text.split()

    n_tokens = len(tokens)

    n_mask = int(max(mask_ratio * n_tokens, 1))
    mask_indices = torch.randperm(n_tokens)[:n_mask].sort().values

    assert len(mask_indices) > 0, input_text

    mask_indices = mask_indices.tolist()
    span = [mask_indices[0], mask_indices[0]+1]
    spans = []

    for i, mask_index in enumerate(mask_indices):
        # if current mask is not the last one & the next mask is right after current mask
        if i < len(mask_indices) - 1 and mask_indices[i+1] == mask_index + 1:
            contiguous = True
        else:
            contiguous = False

        if contiguous:
            span[1] += 1

        else:
            # non contiguous -> output current span
            spans.append(span)
            # if current mask is not the last one -> create next span
            if i < len(mask_indices) - 1:
                span = [mask_indices[i+1], mask_indices[i+1]+1]

    masked_tokens = deepcopy(tokens)

    cum_span_length = 0
    for i, span in enumerate(spans):
        start, end = span

        masked_tokens[start-cum_span_length +
                      i: end-cum_span_length+i] = ['<mask>']

        cum_span_length += (end - start)

    masked_text = " ".join(masked_tokens)

    if prefix is None:
        source_text = masked_text
    else:
        source_text = f"{prefix} {masked_text}"

    target_text = input_text

    return source_text, target_text


def ground_cntl_caption(cntl_obj_adj, cntl_captions, cntl_signal, n_ground=1, prefix="describe visual inputs:", diverse=False, add_verbs=False):
    """
    For VG

    n_boxes = 36 image regions from Detectron2

    Args:
        captions
        n_ground

    Returns:
        source_text
        target_text

    Ex) (in vocab ids)
    captions
        ['Yellow banana', 'red crayon', 'black cow', 'blue sky']

    n_ground > 1

    ground_indices
        [1, 0, 2]
    source_text
        describe visual inputs: <vis_extra_id_1> <vis_extra_id_0> <vis_extra_id_2>
    target_text
        <extra_id_0> red crayon <extra_id_1> Yellow banana <extra_id_2> black cow

    n_ground == 1

    source_text
        describe visual inputs: <vis_extra_id_1>
    target_text
        red crayon
    """

    levela = 10
    levelb = 20
    levelc = 30
    leveld = 40
    levele = 50

    d_idx = random.choice(range(0,len(cntl_signal)))
    sel_phrase = cntl_captions[d_idx]
    sel_control = cntl_signal[d_idx]
    sel_control_name = cntl_obj_adj[d_idx]

    source_text = [prefix]
    target_text = []
    cntl_names = []
    length_level = []

    if add_verbs:
        postags = nltk.pos_tag(nt.word_tokenize(sel_phrase))
        verbs = ''
        for tag in postags:
            if 'VB' in tag[1]:
                verbs += ' ' + tag[0]
        source_text.append(verbs)

    if n_ground == 1:
        # add target sentence length
        tokens = sel_phrase.split()
        snt_len = len(tokens)

        if diverse:
            np.random.shuffle(sel_control)

            snt_len = random.choice(range(5,30))

            # shift = random.choice(range(-10,10))
            # snt_len += shift
            # if snt_len < 5:
            #     snt_len= 5

        if snt_len < levela:
            if diverse:
                snt_len = random.choice(range(5,levela))
            source_text.append(f'<cap_len_level_{1}> {snt_len}')
        elif snt_len < levelb:
            if diverse:
                snt_len = random.choice(range(levela,levelb))
            source_text.append(f'<cap_len_level_{2}> {snt_len}')
        elif snt_len < levelc:
            if diverse:
                snt_len = random.choice(range(levelb,levelc))
            source_text.append(f'<cap_len_level_{3}> {snt_len}')
        elif snt_len < leveld:
            if diverse:
                snt_len = random.choice(range(levelc,leveld))
            source_text.append(f'<cap_len_level_{4}> {snt_len}')
        else:
            if diverse:
                snt_len = random.choice(range(leveld,leveld+5))
            source_text.append(f'<cap_len_level_{5}> {snt_len}')

        for idx in range(len(sel_control)):
            source_text.append(f'<vis_extra_id_{sel_control[idx]}>')

        target_text.append(f'{sel_phrase}')
        cntl_names.append(sel_control_name)
        length_level.append(str(snt_len))

    # target_text.append('</s>')

    source_text = " ".join(source_text)
    target_text = " ".join(target_text)
    cntl_names = " ".join(cntl_names)
    length_level = " ".join(length_level)

    # return ground_indices, source_text, target_text
    return source_text, target_text, cntl_names, length_level

def ground_caption(captions, n_ground=1, prefix="describe visual inputs:", sort=True):
    """
    For VG

    Args:
        captions
        n_ground

    Returns:
        source_text
        target_text

    Ex) (in vocab ids)
    captions
        ['Yellow banana', 'red crayon', 'black cow', 'blue sky']

    n_ground > 1

    ground_indices
        [1, 0, 2]
    source_text
        describe visual inputs: <vis_extra_id_1> <vis_extra_id_0> <vis_extra_id_2>
    target_text
        <extra_id_0> red crayon <extra_id_1> Yellow banana <extra_id_2> black cow

    n_ground == 1

    source_text
        describe visual inputs: <vis_extra_id_1>
    target_text
        red crayon
    """

    n_boxes = len(captions)

    if sort:
        ground_indices = torch.randperm(n_boxes)[:n_ground].sort().values
    else:
        ground_indices = torch.randperm(n_boxes)[:n_ground]

    ground_indices = ground_indices.tolist()

    source_text = [prefix]
    target_text = []

    if n_ground == 1:
        idx = ground_indices[0]
        source_text.append(f'<vis_extra_id_{idx}>')
        target_text.append(f'{captions[idx]}')
    else:
        for j, idx in enumerate(ground_indices):
            source_text.append(f'<vis_extra_id_{idx}>')

            target_text.append(f'<extra_id_{j}>')
            target_text.append(f'{captions[idx]}')

    # target_text.append('</s>')

    source_text = " ".join(source_text)
    target_text = " ".join(target_text)

    # return ground_indices, source_text, target_text
    return source_text, target_text, target_text


def refer_expression(captions, n_ground=1, prefix="refer expressions:", sort=True):
    """

    n_ground > 1

    ground_indices
        [1, 0, 2]
    source_text
        refer expressions: <extra_id_0> red crayon <extra_id_1> Yellow banana <extra_id_2> black cow
    target_text
        <vis_extra_id_1> <vis_extra_id_0> <vis_extra_id_2>

    n_ground == 1

    source_text
        refer expressions: red crayon
    target_text
        <vis_extra_id_1>
    """
    n_boxes = len(captions)

    if sort:
        ground_indices = torch.randperm(n_boxes)[:n_ground].sort().values
    else:
        ground_indices = torch.randperm(n_boxes)[:n_ground]

    ground_indices = ground_indices.tolist()

    source_text = [prefix]
    target_text = []

    if n_ground == 1:
        idx = ground_indices[0]
        source_text.append(f'{captions[idx]}')
        target_text.append(f'<vis_extra_id_{idx}>')
    else:

        for j, idx in enumerate(ground_indices):
            source_text.append(f'<extra_id_{j}>')
            source_text.append(f'{captions[idx]}')

            target_text.append(f'<vis_extra_id_{idx}>')

    # target_text.append('</s>')

    source_text = " ".join(source_text)
    target_text = " ".join(target_text)

    # return ground_indices, source_text, target_text
    return source_text, target_text
