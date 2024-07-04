import codecs
import json
import os
import re
from ast import literal_eval

import inflect
import nltk
import numpy as np
import penman
import scipy
from utils import get_sentence_data
from nltk.tokenize import word_tokenize
from utils import glove_cosine_similarity, load_glove_vecs

from config import parse_args


p = inflect.engine()

excempt_nouns = 'outside person spending steps'

grouped_edges = {':prep-with': ':accompanier', ':dayperiod': ':time', ':prep-on': ":location", ":unit": ":quant", ":prep-in": ":location", ":prep-by": ":location", ":prep-against": ":compared-to", \
":prep-to": ":direction", ":duration": ":time", ":prep-as": ":purpose", ":prep-from": ":source", ":frequency": ":time", ":prep-into": ":consist", ":prep-for": ":purpose", ":prep-under": ":location", \
":prep-in-addition-to": ":accompanier", ":prep-on-top": ":location", ":prep-at": ":time", ":weekday": ":time", ":prep-amid": ":location", ":prep-among": ":location", ":prep-along-with": ":accompanier", \
":prep-near": ":location", ":prep-side-by": ":location", ":prep-over": ":location", ":prep-up": ":location", ":value": ":quant", ":timezone": ":time", ":prep-next-to": ":location", ":prep-along-side": ":prep-along-side", \
":prep-all-on": ":location", ":prep-behind": ":location", ":prep-in-front": ":location", ":prep-on-to": ":location", ":prep-up-to": ":location", ":prep-upon": ":location", ":day": ":time" }

def load(flickr_sentences, input_filename, sentences, vgamr_nodes, output, glove_vecs, trim=False, grouped_edges={}, gruen_threshold=0):
    '''
    :param input_filename:
    :param trim: trim based on edges
    :return:
    '''

    def getannotatedsentences(img_id, most_info=False):
        sen_full_path = os.path.abspath(flickr_sentences+img_id+".txt")
        snt = get_sentence_data(sen_full_path)

        most_info_phrase_id = {}
        for s in snt:
            for p in s['phrases']:
                if p['phrase_id'] in most_info_phrase_id.keys():
                    if most_info:
                        if len(word_tokenize(p['phrase'])) > len(word_tokenize(most_info_phrase_id[p['phrase_id']])):
                            most_info_phrase_id[p['phrase_id']] = p['phrase']
                    else:
                        if len(word_tokenize(most_info_phrase_id[p['phrase_id']])) == 1:
                            most_info_phrase_id[p['phrase_id']] = p['phrase']
                        if len(word_tokenize(p['phrase'])) == 1:
                            continue
                        if len(word_tokenize(p['phrase'])) < len(word_tokenize(most_info_phrase_id[p['phrase_id']])):
                            most_info_phrase_id[p['phrase_id']] = p['phrase']            
                else:
                    most_info_phrase_id[p['phrase_id']] = p['phrase']
        for s in snt:
            offset = 0
            for p in s['phrases']:
                new_phrase_len = len(word_tokenize(most_info_phrase_id[p['phrase_id']]))
                old_phrase_len = len(word_tokenize(p['phrase']))
                old_phrase_word_index = p['first_word_index']
                s['sentence'] = s['sentence'].replace(p['phrase'], most_info_phrase_id[p['phrase_id']])
                p['phrase'] = most_info_phrase_id[p['phrase_id']]
                p['first_word_index'] = old_phrase_word_index + offset
                offset += new_phrase_len - old_phrase_len
        return snt

    def getsentences(snts):
        snt = ['lala']
        sentences = {}
        missing_nouns = ''
        try:
            while True:
                line_c = snts.readline()

                if not line_c:
                    break

                [snt, temp] = line_c.split('GRUEN')
                [gruen, amrid] = temp.split('CAPTIONID')
                if float(gruen) > gruen_threshold:
                    # get nouns from ground truth sentences
                    img_id = amrid.split('_')[0]
                    gtsnt = getannotatedsentences(img_id)
                    gt_nouns = ''
                    gt_all_words = ''
                    for s in gtsnt:
                        gt_all_words += ' ' + s['sentence']
                        tags = nltk.pos_tag(word_tokenize(s['sentence']))
                        for tag in tags:
                            if 'NN' in tag[1]:
                                res = p.singular_noun(tag[0])
                                if res == False:
                                    gt_nouns += ' ' + tag[0]
                                else: 
                                    gt_nouns += ' ' + res

                    # get nouns from generated sentence.
                    tags = nltk.pos_tag(word_tokenize(snt))

                    mis_noun_flag = False
                    
                    for tag in tags:
                        if 'NN' in tag[1]:
                            res = p.singular_noun(tag[0])
                            if res == False:
                                res = tag[0]
                            if res not in excempt_nouns and res not in gt_nouns and res not in gt_all_words:
                                words = gt_all_words.split()
                                try:
                                    synonym_flag = False
                                    for word in words:
                                        cos_sim = glove_cosine_similarity(word, res, glove_vecs)
                                        if cos_sim > 0.5:
                                            synonym_flag = True
                                            break
                                    if not synonym_flag:
                                        missing_nouns += ' ' + tag[0]
                                        mis_noun_flag = True
                                except:
                                    print('no glove vec')
                    if mis_noun_flag:
                        continue
                    
                    amrid = amrid.split('\n')[0]
                    if amrid not in sentences:
                        if len(sentences )>0:
                            if snt in sentences.values():
                                continue
                        sentences[amrid] = snt
                    else:
                        print('Double entry for amr id: ' + str(amrid))
        except Exception as e:
            print(str(e))

        return sentences

    def _get_nodes_edges_attrs(pg, grouped_edges):
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
            if not tuple((nodes.source,)) in doc_node2idx:
                node_idx = int(nodes.source.split('z')[1])
                doc_node2idx.update({tuple((nodes.source,)): node_idx})
                doc_nodes.update({node_idx: tuple((nodes.target,))})
                node_indices += 1
            if first_node:
                doc_root_nodes.update({doc_node2idx[tuple((nodes.source,))]: tuple((nodes.target,))})
                first_node = False     

        for edge in pg_edges:
            
            try:
                v_source, v_role, v_target = edge.source, edge.role, edge.target
            except:
                print("No dictionary entry for role " + edge.role)
                v_source, v_role, v_target = edge.source, edge.role, edge.target

            source_node, target_node = tmp[v_source], tmp[v_target]
            source_node_idx, target_node_idx = doc_node2idx[tuple((v_source,))], doc_node2idx[tuple((v_target,))]
            if not tuple((source_node_idx, target_node_idx)) in doc_edges:
                doc_edges.update({tuple((source_node_idx, target_node_idx)): v_role})

        if trim is False:
            for attr in pg_attrs:
                v_source, v_role, v_target = attr.source, attr.role, attr.target
                source_node = tmp[v_source]
                source_node_idx = doc_node2idx[tuple((v_source,))]
                if tuple((source_node_idx,)) not in doc_attrs:
                    doc_attrs[tuple((source_node_idx,))] = [(source_node, v_target, v_role)]
                else:
                    doc_attrs[tuple((source_node_idx,))].append((source_node, v_target, v_role))

        return

    sentences = getsentences(sentences)
    print(f'Total Number of filtered sentences {len(sentences)}')
    
    graph_str = ''
    amr_str0 = ''
    info_dict = {}

    doc_filename = ''
    corpus = {}  # filename -> (nodes, root_nodes, edges, exp_edges)
    amr_str = {}
    info_dict_all = {}

    img_dict = {}
    img_id = None
    caption_id = 0

    with codecs.open(input_filename, 'r', 'utf-8') as infile:
        for line in infile:
            # ipdb.set_trace()
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
                        if img_id == None:
                            img_id = int(doc_filename.split('_')[0])
                        if img_id != int(doc_filename.split('_')[0]):
                            # we changed img, save json file
                            with open(f"{output}{img_id}.json", "w") as outfile:
                                json.dump(img_dict, outfile) 

                            caption_id = 0
                            img_id = int(doc_filename.split('_')[0])
                            img_dict = {}

                        corpus[doc_filename] = (doc_nodes, doc_root_nodes, doc_edges, doc_attrs)
                        if doc_filename in sentences:
                            subsnt = sentences[doc_filename]

                            subamr_id = doc_filename.split('_')[0]
                            with open(f"{vgamr_nodes}{subamr_id}.json", "r") as outfile:
                                bboxes = json.load(outfile)
                            objects = []
                            relationships = []
                            for node in doc_nodes:
                                if node in doc_root_nodes:
                                    # print('Found the main relation!')
                                    single_relationship = {}
                                    single_relationship['name'] = doc_nodes[node][0]
                                    single_relationship['relationship_id'] = len(relationships)
                                    # add relationship
                                    relationships.append(single_relationship)
                                    continue
                                res = re.findall("-[0-9]+", doc_nodes[node][0])
                                if len(res)>0:
                                    res = []
                                    #do nothing for now.
                                else:
                                    for bbox_name in bboxes:
                                        [node_id, node_name] = bbox_name.split('_')
                                        if int(node_id) == node:
                                            if bboxes[bbox_name] == 'scene':
                                                single_object = {}
                                                single_object['object_id'] = 'scene'
                                                single_object['name'] = str(node_name)
                                                single_object['attributes'] = []
                                                # dummy coordinates since it refers to complete image
                                                single_object['xmin'] = 0
                                                single_object['ymin'] = 0
                                                single_object['xmax'] = 0
                                                single_object['ymax'] = 0
                                                if single_object not in objects:
                                                    objects.append(single_object)
                                                continue
                                            bbox = literal_eval(bboxes[bbox_name])
                                            for single_bbox in bbox:
                                                single_object = {}
                                                single_object['object_id'] = int( str(single_bbox[0]) + str(single_bbox[1]) + str(single_bbox[2]) + str(single_bbox[3]))
                                                single_object['name'] = str(node_name)
                                                single_object['attributes'] = []
                                                single_object['xmin'] = single_bbox[0]
                                                single_object['ymin'] = single_bbox[1]
                                                single_object['xmax'] = single_bbox[2]
                                                single_object['ymax'] = single_bbox[3]
                                                # add in object list
                                                if single_object not in objects:
                                                    objects.append(single_object)
                            tmp = {}
                            tmp["objects"] = objects.copy()
                            tmp["relationships"] = relationships.copy()
                            tmp["region_id"] = int(caption_id)
                            tmp["image_id"] = int(subamr_id)
                            tmp["phrase"] = str(subsnt)
                            img_dict[str(caption_id)] = tmp.copy()
                            caption_id += 1

                    doc_filename = filename
                    doc_nodes = {}
                    doc_align = {}
                    doc_root_nodes = {}
                    doc_edges = {}
                    doc_attrs = {}
                    doc_node2idx = {}
                    node_indices = 0
                    
                pg = penman.decode(graph_str)
                _get_nodes_edges_attrs(pg, grouped_edges)
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

    return corpus, amr_str

if __name__ == '__main__':
    print('Loading args')
    args = parse_args(True)

    print('Loading glove vectors')
    glove_vecs = load_glove_vecs(args.glove)

    sentences = open(args.gen_snts)
    # loading the visual grounded sampled amrs and their generated sentenses, quality filtering, and save them in the ASG json format.
    body_corpus, amr_text = load(args.flickr_sentences, args.sampled_amrs, sentences, args.vgamr_nodes, args.output, glove_vecs=glove_vecs, gruen_threshold=args.gruen_threshold)

    sentences.close()