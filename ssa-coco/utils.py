#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import codecs
import json
from itertools import chain
import re

import inflect
import ipdb
import nltk
import nltk.tokenize as nt
from nltk.tokenize import word_tokenize
import numpy as np
import penman
from scipy.spatial.distance import cosine

p = inflect.engine()

__all__ = ["ngrams"]

entities = ['man', 'person', 'girl', 'child', 'woman', 'lady', 'boy', 'toddler', 'baby', 'guy']

def read_amr_meta(name):
    with open(name, 'r') as f:
        data = f.readlines()

    info_dict = {}
    total_info_dict = {}
    for line in data:
        # print (line)
        if line.strip() == '':
            if len(info_dict) == 0:
                continue
            else:
                total_info_dict[info_dict['id']] = info_dict
                info_dict = {}

        if line.startswith('#'):
            fields = line.split('::')
            for field in fields[1:]:
                tokens = field.split()
                info_name = tokens[0]
                info_body = ' '.join(tokens[1:])
                info_dict[info_name] = info_body

    return total_info_dict

def has_numbers(inputString):
     return any(char.isdigit() for char in inputString)

def glove_cosine_similarity(node1, node2, glove_vecs):
    try:
        glove_node2 = glove_vecs[node2]
        glove_node1 = glove_vecs[node1]
    except Exception as ex:
        print(str(ex))
        return -1   
    return 1 - min(1,cosine(glove_node1, glove_node2))

def load(input_filename, coco_entities, trim=False):
    '''
    :param input_filename:
    :param trim: trim based on edges
    :return:
    '''
    def _get_annotations(img_id, sentence):
        if img_id not in coco_entities:
            ipdb.set_trace()
            return None
        annotations = coco_entities[img_id]
        for ann_snt in annotations:
            if ann_snt[-1] == ' ':
                if sentence in ann_snt:
                    annotations4amr = annotations[ann_snt]
                    annotations4amr['sentence'] = ann_snt
                    return annotations4amr
            if ann_snt == sentence:
                annotations4amr = annotations[ann_snt]
                annotations4amr['sentence'] = ann_snt
                return annotations4amr
        return None

    def _get_nodes_edges_attrs(pg):
        '''
        :param pg: decoded penman string
        :return:
        '''
        nonlocal doc_nodes, doc_root_nodes, doc_edges, doc_attrs, doc_node2idx, node_indices, doc_align
        pg_nodes, pg_edges, pg_attrs = pg.instances(), pg.edges(), pg.attributes()
        tmp = {}  # variable concept mapping
        first_node = True
        for nodes in pg_nodes:
            tmp[nodes.source] = nodes.target
            if not tuple((nodes.target,)) in doc_node2idx:
                doc_node2idx.update({tuple((nodes.target,)): node_indices})
                doc_nodes.update({node_indices: tuple((nodes.target,))})

                # get alignments between sentence words and amr nodes
                for epi in pg.epidata:
                    if epi[0] == nodes[0] and epi[1] == nodes[1] and epi[2] == nodes[2]:
                        doc_align[node_indices] = pg.epidata[epi][0].indices[0]
                        break
                node_indices += 1
            if first_node:
                doc_root_nodes.update({doc_node2idx[tuple((nodes.target,))]: tuple((nodes.target,))})
                first_node = False     

        for edge in pg_edges:
            
            try:
                v_source, v_role, v_target = edge.source, edge.role, edge.target
            except:
                print("No dictionary entry for role " + edge.role)
                v_source, v_role, v_target = edge.source, edge.role, edge.target

            source_node, target_node = tmp[v_source], tmp[v_target]
            source_node_idx, target_node_idx = doc_node2idx[tuple((source_node,))], doc_node2idx[tuple((target_node,))]
            if not tuple((source_node_idx, target_node_idx)) in doc_edges:
            # if not (tuple((source_node_idx, target_node_idx)) in doc_edges or tuple((target_node_idx, source_node_idx)) in doc_edges):
                doc_edges.update({tuple((source_node_idx, target_node_idx)): v_role})

        if trim is False:
            for attr in pg_attrs:
                v_source, v_role, v_target = attr.source, attr.role, attr.target
                source_node = tmp[v_source]
                source_node_idx = doc_node2idx[tuple((source_node,))]
                if tuple((source_node_idx,)) not in doc_attrs:
                    doc_attrs[tuple((source_node_idx,))] = [(source_node, v_target, v_role)]
                else:
                    doc_attrs[tuple((source_node_idx,))].append((source_node, v_target, v_role))
        return

    graph_str = ''
    amr_str0 = ''
    info_dict = {}

    doc_filename = ''
    corpus = {}  # filename -> (nodes, root_nodes, edges, exp_edges)
    amr_str = {}
    annotations = {}
    info_dict_all = {}

    with codecs.open(input_filename, 'r', 'utf-8') as infile:
        for line in infile:
            line = line.rstrip()

            if line == '' or line == '\n':
                if len(info_dict) > 0:
                    amr_str[info_dict['id']] = amr_str0
                    amr_str0 = '' 
                    info_dict_all[info_dict['id']] = info_dict.copy()

                if graph_str == '':
                    info_dict = {}
                    continue

                filename = info_dict['id']
                if filename != doc_filename:
                    if doc_filename != '':
                        ann_coco = _get_annotations(doc_filename.split('.')[0], info_dict_all[doc_filename]['snt'])
                        if ann_coco == None:
                            doc_filename = filename
                            doc_nodes = {}
                            doc_align = {}
                            doc_root_nodes = {}
                            doc_edges = {}
                            doc_attrs = {}
                            doc_node2idx = {}
                            node_indices = 0

                        ann_nodes = {}
                        ann_nodes['sentence'] = ann_coco['sentence']
                        if info_dict_all[doc_filename]['tok'] != info_dict_all[doc_filename]['snt']:
                            tok_1 = nt.word_tokenize(info_dict_all[doc_filename]['tok'])
                            tok_2 = nt.word_tokenize(info_dict_all[doc_filename]['snt'])
                            ii = 0
                            jj = 0
                            for tok in tok_1:
                                if tok != tok_2[ii]:
                                    if tok in tok_2[ii]:
                                        for iii in range(len(doc_align)):
                                            if doc_align[iii] == jj:
                                                doc_align[iii] = ii
                                        if tok_1[jj+1] == tok_2[ii+1]:
                                            ii += 1
                                    else:
                                        ii += 1
                                else:
                                    if ii != jj:
                                        for iii in range(len(doc_align)):
                                            if doc_align[iii] == jj:
                                                doc_align[iii] = ii
                                    ii += 1
                                jj += 1

                        tag = nltk.pos_tag(word_tokenize(ann_nodes['sentence']))
                        for node in doc_nodes:
                            node_label = doc_nodes[node][0]
                            res = re.findall("-[0-9]+", node_label)
                            if len(res)== 0 and 'NN' in tag[doc_align[node]][1]:
                                if ann_coco['det_sequences'][doc_align[node]] != None:
                                    noun_name = ann_coco['det_sequences'][doc_align[node]]
                                    if noun_name == '_':
                                        noun_name = 'unknown'
                                        ann_nodes[node] = (noun_name, noun_name)
                                    else:
                                        bndbox = ann_coco['detections'][noun_name]
                                        ann_nodes[node] = (noun_name, bndbox)
                                
                        annotations[doc_filename] = ann_nodes
                        corpus[doc_filename] = (doc_nodes, doc_root_nodes, doc_edges, doc_attrs)

                    doc_filename = filename
                    doc_nodes = {}
                    doc_align = {}
                    doc_root_nodes = {}
                    doc_edges = {}
                    doc_attrs = {}
                    doc_node2idx = {}
                    node_indices = 0
                    
                pg = penman.decode(graph_str)
                _get_nodes_edges_attrs(pg)
                graph_str = ''
                info_dict = {}
                continue

            if line.startswith('#'):
                amr_str0 += line
                amr_str0 += '\n'
                fields = line.split('::')
                for field in fields[1:]:
                    tokens = field.split()
                    info_name = tokens[0]
                    info_body = ' '.join(tokens[1:])
                    info_dict[info_name] = info_body
                continue
            graph_str += line
            amr_str0 += line
            amr_str0 += '\n'

        corpus[doc_filename] = (doc_nodes, doc_root_nodes, doc_edges, doc_attrs)
        ann_coco = _get_annotations(doc_filename.split('.')[0],info_dict_all[doc_filename]['snt'])
        if ann_coco == None:
            return corpus, amr_str, annotations
        ann_nodes = {}
        ann_nodes['sentence'] = ann_coco['sentence']
        tag = nltk.pos_tag(word_tokenize(ann_nodes['sentence']))
        for node in doc_nodes:
            if 'NN' in tag[doc_align[node]][1]:
                if ann_coco['det_sequences'][doc_align[node]] != None:
                    noun_name = ann_coco['det_sequences'][doc_align[node]]
                    if noun_name == '_':
                        noun_name = 'unknown'
                        ann_nodes[node] = (noun_name, noun_name)
                    else:
                        bndbox = ann_coco['detections'][noun_name]
                        ann_nodes[node] = (noun_name, bndbox)

        annotations[doc_filename] = ann_nodes

    return corpus, amr_str, annotations

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    """
    Returns a padded sequence of items before ngram extraction.

        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ['<s>', 1, 2, 3, 4, 5, '</s>']
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        ['<s>', 1, 2, 3, 4, 5]
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [1, 2, 3, 4, 5, '</s>']

    :param sequence: the source data to be padded
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence

def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    For example:

        >>> from nltk.util import ngrams
        >>> list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    Wrap with list for a list version of this function.  Set pad_left
    or pad_right to true in order to get additional ngrams:

        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]


    :param sequence: the source data to be converted into ngrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

def load_glove_vecs(fp):
    dic={}
    if not fp:
        return dic
    with open(fp, "r", encoding="utf8") as f:
        for line in f:
            ls = line.split()
            word = ls[0]
            vec = np.array([float(x) for x in ls[1:]])
            dic[word] = vec
    return dic 