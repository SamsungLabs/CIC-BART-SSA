import codecs
import json
import os
import re
from ast import literal_eval

import inflect
import nltk
import penman
from nltk.tokenize import word_tokenize
from utils import glove_cosine_similarity, load_glove_vecs
from config import parse_args

from nltk.tag import StanfordPOSTagger
from nltk import word_tokenize

# initializing punctuations string
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
# frequent hallucinating words from SPRING parsers.
excempt_nouns = 'outside person spending steps'
hallucination_words = ['emperor', 'beltran', 'beltrn', 'enzyme', 'bowel', 'bowels', 'bactorial', 'obama', 'blah', 'jewish', 'korea', 'korean']
grouped_edges = {'':''}

def load(input_filename, sentences, vgamr_nodes, output, glove_vecs, trim=False, gruen_threshold=0):
    '''
    :param input_filename: sampled visual grounded AMRs (vgAMRs)
    :param input_filename: generated sentences from the sampled vgAMRs using AMR-to-text parsers
    :param trim: trim based on edges
    :return:
    '''
    def getannotatedsentences(img_id):
        if img_id not in coco_entities:
            return None
        annotations = coco_entities[img_id]
        snts = []
        for ann_snt in annotations:
            snts.append(ann_snt)
        return snts
        
    def getsentences(snts):
        snt = ['temp']
        sentences = {}
        missing_nouns = ''
        try:
            while True:
                line_c = snts.readline()

                if not line_c:
                    break

                [snt, temp] = line_c.split('GRUEN')
                [gruen, amrid] = temp.split('CAPTIONID')
                # remove merged
                if len(amrid.split('_'))<3:
                    continue
                snt = snt.lower()
                snt2return = snt.lower()
                if float(gruen) > gruen_threshold:
                    # get nouns from ground truth sentences
                    img_id = amrid.split('_')[0]
                    gtsnt = getannotatedsentences(img_id)
                    gt_all_words = ''
                    for s in gtsnt:
                        s = s.lower()
                        # remove punctuation symbols
                        for ele in s:
                            if ele in punc:
                                s = s.replace(ele, " ")
                        gt_all_words += ' ' + s
                    
                    # remove punctuation symbols
                    for ele in snt:
                            if ele in punc:
                                snt = snt.replace(ele, " ")

                    # check for SPRING/IBM common hallucinations.
                    found_hallucinations = False
                    for hal_word in hallucination_words:
                        if hal_word in snt:
                            found_hallucinations = True
                            break
                    if found_hallucinations:
                        continue

                    # get nouns from generated sentence.
                    tags = nltk.pos_tag(word_tokenize(snt))
                    has_nouns = False
                    for tag in tags:
                        if 'NN' in tag[1]:
                            has_nouns = True
                            break
                    if not has_nouns:
                        tags = pos_tagger.tag(word_tokenize(snt))

                    mis_noun_flag = False
                    has_nouns = False
                    for tag in tags:
                        if 'NN' in tag[1]:
                            has_nouns = True
                            res = tag[0]
                            if res not in excempt_nouns and res not in gt_all_words:
                                words = gt_all_words.split()
                                try:
                                    synonym_flag = False
                                    for word in words:
                                        cos_sim = glove_cosine_similarity(word, res, glove_vecs)
                                        if res == 'couch' or res == 'table': # frequent hallucinating SPRING words.
                                            if cos_sim > 0.8:
                                                synonym_flag = True
                                                break
                                        else:
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

                    if not has_nouns:
                        continue
                    
                    amrid = amrid.split('\n')[0]
                    if amrid not in sentences:
                        if len(sentences )>0:
                            if snt in sentences.values():
                                continue
                        sentences[amrid] = snt2return
                    else:
                        print('Double entry for amr id: ' + str(amrid))
        except Exception as e:
            print(str(e))

        return sentences

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
    print(f'Total Number of sentences {len(sentences)}')
    
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
            line = line.rstrip()

            if line == '' or line == '\n':
                if len(info_dict) > 0:
                    amr_str[info_dict['id']] = amr_str0
                    amr_str0 = '' 
                    info_dict_all[info_dict['id']] = info_dict.copy()

                if graph_str == '':
                    info_dict = {}
                    continue

                filename = info_dict['id'].replace('#^#', '-')
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
                            with open(f"{vgamr_nodes}{doc_filename.replace('---','#^#')}.json", "r") as outfile:
                                bboxes = json.load(outfile)
                            objects = []
                            relationships = []
                            not_fully_grounded = False
                            for node in doc_nodes:
                                if not_fully_grounded:
                                    break
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
                                else:
                                    for bbox_name in bboxes:
                                        [node_id, node_name] = bbox_name.split('_')
                                        if int(node_id) == node:
                                            if bboxes[bbox_name] == 'unknown':
                                                not_fully_grounded = True
                                                break
                                            bbox = literal_eval(bboxes[bbox_name])
                                            for single_bbox in bbox:
                                                single_object = {}
                                                single_object['object_id'] = int( str(int(single_bbox[1][0])) + str(int(single_bbox[1][1])) \
                                                                                 + str(int(single_bbox[1][2])) + str(int(single_bbox[1][3])))
                                                single_object['name'] = str(node_name)
                                                single_object['attributes'] = []
                                                single_object['xmin'] = single_bbox[1][0]
                                                single_object['ymin'] = single_bbox[1][1]
                                                single_object['xmax'] = single_bbox[1][2]
                                                single_object['ymax'] = single_bbox[1][3]
                                                # add in object list
                                                if single_object not in objects:
                                                    objects.append(single_object)
                            if not not_fully_grounded:
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

    return corpus, amr_str

if __name__ == '__main__':
    print('Loading args')
    args = parse_args(True)

    print('Setting up pos tagger')
    # Stanford POS tagger initialization
    os.environ['JAVAHOME'] = args.java_path
    p = inflect.engine()
    pos_tagger = StanfordPOSTagger(args.model_postager, args.jar_postager)

    print('Loading coco entities dataset')
    with open(args.coco_entities, 'r') as fp:  
        coco_entities = json.load(fp)

    print('Loading glove vectors')
    glove_vecs = load_glove_vecs(args.glove)

    sentences = open(args.gen_snts)
    # loading the visual grounded sampled amrs and their generated sentenses, quality filtering, and save them in the ASG json format.
    body_corpus, amr_text = load(args.sampled_amrs, sentences, args.vgamr_nodes, args.output, glove_vecs=glove_vecs, gruen_threshold=args.gruen_threshold)

    sentences.close()