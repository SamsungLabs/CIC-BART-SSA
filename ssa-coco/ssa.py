from __future__ import annotations

import copy
import json
import logging
import logging.config
import os
import random
import re

import networkx
import nltk
import nltk.tokenize as nt
import numpy as np
import penman
import scipy
import utils
from amrlib.graph_processing.amr_plot import AMRPlot
from penman.graph import Graph
from scipy.spatial.distance import cosine
from tqdm import tqdm
from utils import (bb_intersection_over_union, load)

from configssa import parse_args

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True
})
import smatch as sm

filtered_edges=':ARG1 :mod :ARG0 :ARG2 :location :quant :consist :part :time :manner :direction :poss :accompanier ' \
               ':degree :instrument :purpose :topic :domain :ARG3 :ARG4'.split()
grouped_edges = {'':''}

def AMR_pipeline(body_file, output_sampledamrs, output_meta_amrs, output_nodevginfo, propbank, glove, coco_entities_fname, glove_threshold=0.5, trim=False, hyper=None):
    '''
    :param body_corpus: input body corpus, amrs with node alignment from dataset original captions
    :param output_sampledamrs: output file for event-focused sampled meta-vgamrs
    :param output_meta_amrs: output file for meta-vgamrs
    :param output_nodevginfo: meta-vgamr node visual grounding information
    :param propbank: propbank folder
    :param glove: glove vector file
    :param glovethreshold: threshold for the cosine similarity of two glove embeddings
    :return:
    '''

    def _graph(doc_nodes, doc_edges, doc_attrs, doc_root_nodes=None, hyper_path=None):
        '''
        :param doc_nodes:
        :param doc_edges:
        :param doc_attrs:
        :return: string for penman encode, nodes and edges set as well
        '''
        graph_str = []
        nodes = []
        edges = []
        hyper_data = None
        if not hyper_path is None:
            hyper_data = json.load(open(hyper_path))

        for ix, node in doc_nodes.items():
            if hyper_data is None:
                graph_str.append(('z' + str(ix), 'instance', node[0]))
            elif node[0] not in hyper_data or len(hyper_data.get(node[0])) == 0:
                graph_str.append(('z' + str(ix), 'instance', node[0]))
            else:
                graph_str.append(('z' + str(ix), 'instance', hyper_data.get(node[0])[0]))
            nodes.append(ix)

        for ix, rela in doc_edges.items():
            graph_str.append(('z' + str(ix[0]), rela, 'z' + str(ix[1])))
            edges.append((ix[0], ix[1]))

        for ix, attrs in doc_attrs.items():
            for attr in attrs:
                graph_str.append(('z' + str(ix[0]), attr[2], attr[1]))

        new_doc_roots = {}
        if doc_root_nodes is not None and hyper_data is not None:
            for k, v in doc_root_nodes.items():
                if v[0] not in hyper_data or len(hyper_data.get(v[0])) == 0:
                    new_doc_roots.update({k: v})
                else:
                    new_doc_roots.update({k: tuple((hyper_data[v[0]][0],))})
        else:
            new_doc_roots = doc_root_nodes
        return graph_str, nodes, edges, new_doc_roots

    def _nodes_edges_attrs(pg):
        '''
        :param pg: decoded penman string
        :return:
        '''
        doc_nodes = {}
        doc_edges = {}
        doc_root_nodes = {}
        doc_attrs = {}

        pg_nodes, pg_edges, pg_attrs = pg.instances(), pg.edges(), pg.attributes()
        first_node = True
        for nodes in pg_nodes:
            node_indices = int(nodes.source[1:])
            doc_nodes.update({node_indices: tuple((nodes.target,))})
            if first_node:
                doc_root_nodes.update({node_indices: tuple((nodes.target,))})
                first_node = False

        for edge in pg_edges:
            v_source, v_role, v_target = edge.source, edge.role, edge.target
            source_node_idx, target_node_idx = int(v_source[1:]), int(v_target[1:])
            doc_edges.update({tuple((source_node_idx, target_node_idx)): v_role})

        for attr in pg_attrs:
            v_source, v_role, v_target = attr.source, attr.role, attr.target
            source_node_idx = int(v_source[1:])
            if tuple((source_node_idx,)) not in doc_attrs:
                doc_attrs[tuple((source_node_idx,))] = [(v_source, v_target, v_role)]
            else:
                doc_attrs[tuple((source_node_idx,))].append((v_source, v_target, v_role))

        return doc_nodes, doc_edges, doc_attrs, doc_root_nodes

    def _disjointAMR_v2(g_str, nodes, edges):
        '''
        :param num: filename
        :param g_str:
        :param nodes:
        :param edges:
        :return: all connected subgraphs
        '''
        # nonlocal doc_root_nodes
        dis_graph = networkx.DiGraph()
        dis_graph.add_nodes_from(nodes)
        dis_graph.add_edges_from(edges)

        if networkx.is_weakly_connected(dis_graph):
            g = Graph(g_str)
            return g
        else:
            return null

    def _disjointAMR(num, g_str, nodes, edges, doc_root_nodes):
        '''
        :param num: filename
        :param g_str:
        :param nodes:
        :param edges:
        :return: all connected subgraphs
        '''
        # nonlocal doc_root_nodes
        dis_graph = networkx.DiGraph()
        dis_graph.add_nodes_from(nodes)
        dis_graph.add_edges_from(edges)
        ret = {}

        disc = False

        if networkx.is_weakly_connected(dis_graph):
            g = Graph(g_str)
            ret[str(num) + '-0'] = (g, doc_root_nodes)
        else:
            ix = 0
            for sub_nodes in networkx.weakly_connected_components(dis_graph):
                if len(sub_nodes) < 3: # it should cover more than one triplet
                    continue
                sub_dict = {}
                for idx, number in enumerate(sub_nodes):
                    sub_dict[number] = idx

                sub_graph, sub_doc_edge, sub_doc_node = [], {}, {}
                for li in g_str:
                    assert len(tuple(li)) == 3, 'len of line is three'
                    arg1, arg2 = tuple(li)[0], tuple(li)[2]
                    if not (arg2.startswith('z') and utils.has_numbers(arg2) and '-' not in arg2):
                        if int(arg1[1:]) in sub_dict:
                            new_li = ('z' + str(sub_dict[int(arg1[1:])]), li[1], li[2])
                            sub_graph.append(new_li)
                            if li[1] == ':instance':
                                sub_doc_node[sub_dict[int(arg1[1:])]] = li[2]
                    else:
                        if int(arg1[1:]) in sub_dict and int(arg2[1:]) in sub_dict:
                            new_li = ('z' + str(sub_dict[int(arg1[1:])]), li[1], 'z' + str(sub_dict[int(arg2[1:])]))
                            sub_graph.append(new_li)
                            sub_doc_edge[tuple((sub_dict[int(arg1[1:])], sub_dict[int(arg2[1:])]))] = li[1][1:]  # [1:]

                g = Graph(sub_graph)
                sub_root_nodes = {}
                for root in doc_root_nodes.keys():
                    if sub_dict.get(root, -1) != -1:
                        sub_root_nodes.update({sub_dict[root]: tuple((doc_root_nodes[root], ))})

                ret[str(num) + '-' + str(ix)] = (g, sub_root_nodes)
                ix += 1
                disc = True

        return ret, disc

    def _revert_edges(gra, roots, root_ix=0, max_time=5, flag_remove_redundant_paths=False, glove_vecs={}, ann = {}):
        '''
        revert edges for mulitple roots and possible cycles
        '''

        def _get_all_paths(G, node1, node2):
            number_of_paths = 0; shortest_path = {}; second_shortest_path = {}; length_shortest_path = 0; length_second_shortest_path = 0

            paths = networkx.all_simple_paths(G, source=node1, target=node2)

            for path in paths:
                number_of_paths += 1
                len_path = len(path)
                # print(path)
                if length_shortest_path == 0:
                    length_shortest_path = len_path
                    shortest_path = path
                    continue
                if len_path < length_shortest_path:
                    length_shortest_path = len_path
                    shortest_path = path
                    continue
                if len_path == length_shortest_path:
                    length_second_shortest_path = len_path
                    second_shortest_path = path
            return number_of_paths, shortest_path, second_shortest_path, length_shortest_path, length_second_shortest_path

        def _merge_similar_attributes(G, doc_edges, doc_nodes, glove_vecs, ann):

            nodes2remove = {}

            for node1 in G.nodes:

                in_edges = G.in_edges(node1)
                out_edges = G.out_edges(node1)

                adj_nodes = {}
                for in_edge in in_edges:
                    if 'of' in doc_edges[in_edge]:
                        node = in_edge[0]
                        tag = nltk.pos_tag(nt.sent_tokenize(doc_nodes[node][0]))
                        if tag[0][1] == 'JJ' or tag[0][1] == 'JJR' or tag[0][1] == 'JJS':
                            adj_nodes[node] = node


                for out_edge in out_edges:
                    if 'of' not in doc_edges[out_edge]:
                        node = out_edge[1]
                        tag = nltk.pos_tag(nt.sent_tokenize(doc_nodes[node][0]))
                        if tag[0][1] == 'JJ' or tag[0][1] == 'JJR' or tag[0][1] == 'JJS':
                            adj_nodes[node] = node

                if len(adj_nodes) > 0:
                    for adj_node1 in adj_nodes:
                        if adj_node1 in nodes2remove.keys():
                                continue
                        for adj_node2 in adj_nodes:
                            if adj_node2 in nodes2remove.keys():
                                continue
                            if adj_node1 != adj_node2:
                                dist = glove_cosine_similarity(doc_nodes[adj_node1][0], doc_nodes[adj_node2][0], glove_vecs)
                                if dist >= 0.5:
                                    nodes2remove[adj_node2] = adj_node1

            for node in nodes2remove:
                doc_nodes.pop(node)
                G.remove_node(node)
                if node in ann.keys():
                    del ann[node]

                edges2remove = {}

                for edge in doc_edges:
                    if edge[0] == node:
                        edges2remove[edge] = 0

                    if edge[1] == node:
                        edges2remove[edge] = 1

                for edge in edges2remove:

                    # ipdb.set_trace()
                    if edges2remove[edge] == 0:
                        G.add_edge(nodes2remove[node], edge[1])
                        new_doc_edges[tuple((nodes2remove[node], edge[1]))] = doc_edges[edge]
                                            
                    if edges2remove[edge] == 1:
                        G.add_edge(edge[0], nodes2remove[node])
                        new_doc_edges[tuple((edge[0], nodes2remove[node]))] = doc_edges[edge]
                                            
                    new_doc_edges.pop(edge)

            return G, doc_edges, doc_nodes, ann
        
        def _remove_redundant_paths(G, new_doc_edges, doc_nodes, ann):
            for node1 in G.nodes:
                for node2 in G.nodes:
                    try:
                        number_of_paths, shortest_path, second_shortest_path, length_shortest_path, length_second_shortest_path =  _get_all_paths(G, node1, node2)

                        if number_of_paths > 1:
                            if length_shortest_path < 3:
                                for idx in range(length_shortest_path-1):
                                    edge = new_doc_edges[tuple((shortest_path[idx], shortest_path[idx+1]))]
                                    if ('ARG' not in edge) and ('mod' not in edge) and ('snt' not in edge) and ('subset' not in edge):
                                        new_doc_edges.pop(tuple((shortest_path[idx], shortest_path[idx+1])))
                                        G.remove_edge(shortest_path[idx], shortest_path[idx+1])

                                number_of_paths, shortest_path, second_shortest_path, length_shortest_path, length_second_shortest_path =  _get_all_paths(G, node1, node2)
                    except:
                        continue
            return G, new_doc_edges, doc_nodes, ann
       
        def _remove_cycles(dis_graph, new_doc_edges):
            # nonlocal dis_graph, new_doc_edges
            cnt = 0
            while True:
                try:
                    cyc = networkx.find_cycle(dis_graph, orientation="original")
                    # ipdb.set_trace()
                except:
                    break
                cnt += 1
                if len(cyc) == 1:
                    _ = new_doc_edges.pop(tuple((cyc[-1][0], cyc[-1][1])))
                    dis_graph.remove_edge(cyc[-1][0], cyc[-1][1]) 
                    continue
                if cnt >= max_time:
                    _ = new_doc_edges.pop(tuple((cyc[-1][0], cyc[-1][1])))
                    dis_graph.remove_edge(cyc[-1][0], cyc[-1][1])
                else:
                    _ = new_doc_edges.pop(tuple((cyc[-1][0], cyc[-1][1])))
                    if _.endswith('-of'):
                        new_doc_edges[tuple((cyc[-1][1], cyc[-1][0]))] = _[:-3]
                    else:
                        new_doc_edges[tuple((cyc[-1][1], cyc[-1][0]))] = _ + '-of'
                    dis_graph.remove_edge(cyc[-1][0], cyc[-1][1])
                    dis_graph.add_edge(cyc[-1][1], cyc[-1][0])
            return dis_graph, new_doc_edges

        try:
            graph = penman.decode(penman.encode(gra, top='z'+str(list(roots.keys())[root_ix])))
        except:
            print ('='*20)
            print (gra)
            print ('z', str(list(roots.keys())[root_ix]), ' is not good as root node.')
            print ('='*20)
            return False, {}, {}, {}, {}, {}

        doc_nodes, _, doc_attrs, doc_root_nodes = _nodes_edges_attrs(graph)
        # convert edge order by the graph
        new_doc_edges = {}
        tmp_triples = graph.triples
        dis_graph = networkx.DiGraph()
        for tmp in tmp_triples:
            arg2 = tmp[-1]
            if (arg2.startswith('z') and utils.has_numbers(arg2) and '-' not in arg2):
                try:
                    if penman.layout.appears_inverted(graph, tmp):
                        if tmp[1][1:].endswith('-of'):
                            new_doc_edges[tuple((int(tmp[-1][1:]), int(tmp[0][1:])))] = ':' + tmp[1][1:-3]
                            dis_graph.add_edge(int(tmp[-1][1:]), int(tmp[0][1:]))
                        else:
                            new_doc_edges[tuple((int(tmp[-1][1:]), int(tmp[0][1:])))] = ':' + tmp[1][1:] + '-of'
                            dis_graph.add_edge(int(tmp[-1][1:]), int(tmp[0][1:]))
                    else:
                        new_doc_edges[tuple((int(tmp[0][1:]), int(tmp[-1][1:])))] = ':' + tmp[1][1:]
                        dis_graph.add_edge(int(tmp[0][1:]), int(tmp[-1][1:]))
                except Exception as e:
                    print('EXCEPTION in edge revert: ' + str(e))
                    new_doc_edges[tuple((int(tmp[0][1:]), int(tmp[-1][1:])))] = ':' + tmp[1][1:]
                    dis_graph.add_edge(int(tmp[0][1:]), int(tmp[-1][1:]))

        dis_graph, new_doc_edges = _remove_cycles(dis_graph, new_doc_edges)

        if flag_remove_redundant_paths: 
            dis_graph, new_doc_edges, doc_nodes, ann = _merge_similar_attributes(dis_graph, new_doc_edges, doc_nodes, glove_vecs, ann)
            dis_graph, new_doc_edges, doc_nodes, ann = _remove_redundant_paths(dis_graph, new_doc_edges, doc_nodes, ann)
            return True, doc_nodes, new_doc_edges, doc_attrs, doc_root_nodes, ann
        
        return True, doc_nodes, new_doc_edges, doc_attrs, doc_root_nodes, ann

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

    def hierarchical_clustering_with_smatch_score(image_captions_ids, amr_text):
        similarity_matrix = np.zeros([len(image_captions_ids), len(image_captions_ids)])
        for i in range(len(image_captions_ids)):
            for j in range(len(image_captions_ids)):
                amr1 = amr_text[image_captions_ids[i]]
                amr2 = amr_text[image_captions_ids[j]]

                f1 = open("f1.txt",'w')
                f1.write(amr1)
                f2 = open("f2.txt",'w')
                f2.write(amr2)

                f1.close()
                f2.close()

                f1 = open("f1.txt",'r')
                f2 = open("f2.txt",'r')

                for (_, _, best_f_score) in sm.score_amr_pairs(f1, f2):
                    similarity_matrix[i,j] = best_f_score


        distance_matrix = 1-similarity_matrix
        distance_matrix_condensedformat = scipy.spatial.distance.squareform(np.around(distance_matrix,4), checks=False)
        z = scipy.cluster.hierarchy.linkage(distance_matrix_condensedformat, method='average')
        return z
         
    def renumber_graph(len_node1, len_node2, doc_root_nodes2, doc_nodes2, doc_edges2, doc_attrs2, ann2):
        
        doc_nodes2_new = {}
        doc_root_nodes2_new = {}
        ann2_new = {}

        ann2_new['sentence'] = ann2['sentence']
        
        doc_root_nodes2_new[len_node1] = doc_root_nodes2[0]
        
        for i in range(len_node2):
            doc_nodes2_new[i+len_node1] = doc_nodes2[i]
            if i in ann2.keys():
                ann2_new[i+len_node1] = ann2[i]

            keys2update = {}
            for edge in doc_edges2:   
                if i == edge[0]:
                    keys2update[edge] = 0
                if i == edge[1]:
                    keys2update[edge] = 1

            for key in keys2update:
                if keys2update[key] == 0:
                    doc_edges2.update({tuple((i+len_node1, key[1])): doc_edges2[key]})
                else:
                    doc_edges2.update({tuple((key[0], i+len_node1)): doc_edges2[key]})
                    
        keys2remove = {}
        for edge in doc_edges2:   
            if edge[0]<len_node1 or edge[1]<len_node1:
                keys2remove[edge] = edge
        for key in keys2remove:
            del doc_edges2[key]

        new_doc_attrs2 = {}
        for attr in doc_attrs2:
            new_doc_attrs2[tuple((attr[0]+len_node1,))] = doc_attrs2[attr]
        return doc_root_nodes2_new, doc_nodes2_new, doc_edges2, new_doc_attrs2, ann2_new

    def glove_cosine_similarity_matrix(doc_nodes1, doc_nodes2, glove_vecs, len_node1, len_node2):

        dist = np.zeros([len_node2, len_node1])

        ind2 = 0
        for node2 in doc_nodes2:

            try:
                if doc_nodes2[node2][0] in glove_vecs:
                    glove_node2 = glove_vecs[doc_nodes2[node2][0]]
                else:
                    dist[ind2,:] = -1
                    ind2 += 1 
                    continue 
            except:
                dist[ind2,:] = -1
                ind2 += 1 
                continue

            ind1 = 0
            for node1 in doc_nodes1:

                try:
                    if doc_nodes1[node1][0] in glove_vecs:
                        glove_node1 = glove_vecs[doc_nodes1[node1][0]]
                    else:
                        dist[ind2,ind1] = -1
                        ind1 +=1
                        continue
                except:
                    dist[ind2,ind1] = -1
                    ind1 +=1
                    continue

                dist[ind2,ind1] = 1 - min(1,cosine(glove_node1, glove_node2))
                ind1 += 1

            ind2 += 1
        return dist

    def glove_cosine_similarity(node1, node2, glove_vecs):
        try:
            if node2 in glove_vecs and node1 in glove_vecs:
                glove_node2 = glove_vecs[node2]
                glove_node1 = glove_vecs[node1]
            else:
                return -1
        except:
            return -1   
        return 1 - min(1,cosine(glove_node1, glove_node2))

    def get_entity_parents(doc_nodes, node, doc_edges, depth):
        parents = {}
        parent_edges = {}
        node_temp = {}
        node_temp[node] = node

        for _ in range(depth):
            next_node_temp = {}

            for edge in doc_edges:
                if '-of' in doc_edges[edge]:
                    if edge[0] in node_temp:
                        parent_edges[edge] = edge
                        next_node_temp[edge[1]] = edge[1]
                        parents[edge[1]] = edge[1]
                else:
                    if edge[1] in node_temp:
                        parent_edges[edge] = edge
                        next_node_temp[edge[0]] = edge[0]
                        parents[edge[0]] = edge[0]

                # if the parent node is a predicate, look for its args
                for n in node_temp:
                    res = re.findall("-[0-9]+", doc_nodes[n][0])
                    if len(res) != 0:
                        # ipdb.set_trace()
                        if '-of' in doc_edges[edge]:
                            if edge[1] == n:
                                if 'ARG' in doc_edges[edge]:
                                    parent_edges[edge] = edge
                                    next_node_temp[edge[0]] = edge[0]
                                    parents[edge[0]] = edge[0]
                        else:
                            if edge[0] == n:
                                if 'ARG' in doc_edges[edge]:
                                    parent_edges[edge] = edge
                                    next_node_temp[edge[1]] = edge[1]
                                    parents[edge[1]] = edge[1]

            node_temp = next_node_temp

        if len(parents) == 0:
            return parents

        remove_predicate_parent = {}
        for parent in parents:
            if '-' in doc_nodes[parent][0]:
                remove_predicate_parent[parent] = parent

        for remove_pred in remove_predicate_parent:
            del parents[remove_pred]

        return parents

    def get_predicate_parentARGnodes(doc_nodes, node, doc_edges, depth):
        parents = {}
        children = {}
        node_temp = {}
        node_temp[node] = node

        for _ in range(depth):
            next_node_temp = {}

            for edge in doc_edges:
                if '-of' in doc_edges[edge]:
                    if edge[1] in node_temp:
                        next_node_temp[edge[0]] = edge[0]
                        children[doc_edges[edge].split('-of')[0]] = edge[0]
                    if edge[0] in node_temp:
                        parents[doc_edges[edge].split('-of')[0]] = edge[1]
                else:
                    if edge[0] in node_temp:
                        next_node_temp[edge[1]] = edge[1]
                        children[doc_edges[edge]] = edge[1]
                    if edge[1] in node_temp:
                        parents[doc_edges[edge]] = edge[0]

            node_temp = copy.copy(next_node_temp)

        return parents, children

    def subset_edges(nodes, edges, ann, glove_vecs):
        for n1 in nodes:
            for n2 in nodes:
                if n1 == n2:
                    continue
                if n1 in ann.keys() and n2 in ann.keys():
                    if ann[n1][0] != 'unknown' and ann[n2][0] != 'unknown':
                        id1 = ann[n1][1]
                        id2 = ann[n2][1]
                        if id1 != 'unknown' and id2 != 'unknown' and id1 != id2:
                            bb_counter = 0
                            for bb1 in id1:
                                for bb2 in id2:
                                    if bb1[1] != 'unknown' and bb2[1] != 'unknown':
                                        iou = bb_intersection_over_union(bb1[1], bb2[1])
                                        if iou == 1:
                                            bb_counter += 1

                            if bb_counter > 0:
                                label1 = nodes[n1][0].split('#^#')
                                label2 = nodes[n2][0].split('#^#')
                                g_dist = -1
                                tag1 = ''
                                tag2 = ''
                                try:
                                    for l1 in label1:
                                        for l2 in label2:
                                            temp = glove_cosine_similarity(l1, l2, glove_vecs)
                                            t1 = nltk.pos_tag(nt.sent_tokenize(l1))
                                            t2 = nltk.pos_tag(nt.sent_tokenize(l2))
                                            tag1 = tag1 + t1[0][1]
                                            tag2 = tag2 + t2[0][1]
                                            if temp > g_dist:
                                                g_dist = copy.copy(temp)
                                    if 'NN' in tag1 and 'NN' in tag2 and g_dist > 0.34:
                                        if len(id1) > len(id2):
                                            edges.update({tuple((n1, n2)): ":subset"})
                                        else:
                                            edges.update({tuple((n2, n1)): ":subset"})
                                except:
                                    print('Exception in subset edges')

        return nodes, edges
    
    def max_similarity_multilabel_nodes(labels1, labels2):
        max_sim = -1
        labels1 = labels1.split('#^#')
        labels2 = labels2.split('#^#')
        for label1 in labels1:
            for label2 in labels2:
                if label2 == 'and' or labels1 == 'and' or label1 == 'or' or label2 == 'or' or 'multi-sentence' in label1 or 'multi-sentence' in label2:
                    continue
                try:
                    if label1 == label2:
                        max_sim = 1
                        continue
                    res1 = re.findall("-[0-9]+", label1)
                    res2 = re.findall("-[0-9]+", label2)
                    if len(res1) > 0:
                        label1 = label1.split(res1[0])[0]
                    if len(res2) > 0:
                        label2 = label2.split(res2[0])[0]
                    g_sim = glove_cosine_similarity(label1, label2, glove_vecs)
                    if g_sim>max_sim:
                        max_sim = copy.deepcopy(g_sim)
                except:
                    continue
        return max_sim

    def pairwise_merge(doc_nodes1, doc_root_nodes1, doc_edges1, doc_attrs1, ann1, doc_nodes2, doc_root_nodes2, doc_edges2, doc_attrs2, ann2, glove_vecs, sim_score):
        
        len_node1 = len(doc_nodes1)
        len_node2 = len(doc_nodes2)

        try:
            doc_root_nodes2, doc_nodes2, doc_edges2, doc_attrs2, ann2 = renumber_graph(len_node1, len_node2, doc_root_nodes2, doc_nodes2, doc_edges2, doc_attrs2, ann2)
        except:
            print('Exception in pairwise merge')
        
        dist = glove_cosine_similarity_matrix(doc_nodes1, doc_nodes2, glove_vecs, len_node1, len_node2)

        doc_node1_val = doc_nodes1.values()
        doc_node2_val = doc_nodes2.values()

        doc_nodes1_init = doc_nodes1.copy() 
        doc_nodes2_init = doc_nodes2.copy() 

        doc_edges1_init = doc_edges1.copy()
        doc_edges2_init = doc_edges2.copy()

        ann1_init = ann1.copy()
        ann2_init = ann2.copy()

        common_counter = 0
        nodes2del = {}
        ind1 = 0
        for node1 in doc_nodes1_init:

            keys = {}
            ind2 = 0

            if doc_nodes1_init[node1][0] == 'include-91':
                continue

            for node2 in doc_nodes2_init:

                if doc_nodes2_init[node2][0] == 'include-91':
                    continue

                if node2 in nodes2del:
                    ind2 += 1
                    continue

                if ('and' == doc_nodes2_init[node2][0] and 'and' == doc_nodes1_init[node1][0]) or ('or' == doc_nodes2_init[node2][0] and 'or' == doc_nodes1_init[node1][0]):
                    if doc_nodes2_init[node2] in doc_root_nodes2 or doc_nodes1_init[node1] in doc_root_nodes1:
                        keys[node2] = node2
                        ind2 += 1
                        print(f'Merge and nodes')
                        continue
                    else:
                        parents_node2 = get_entity_parents(doc_nodes2_init, node2, doc_edges2_init,2)
                        parents_node1 = get_entity_parents(doc_nodes1_init, node1, doc_edges1_init,2)

                    if len(parents_node1) > 0 and len(parents_node2) > 0:
                        found_match = False
                        for par_1 in parents_node1:
                            if found_match:
                                break
                            for par_2 in parents_node2:
                                g_dist = max_similarity_multilabel_nodes(doc_nodes1_init[par_1][0], doc_nodes2_init[par_2][0])
                                if g_dist >= glove_threshold  and ('and' != doc_nodes1_init[par_1][0]) and ('and' != doc_nodes2_init[par_2][0]) :
                                    keys[node2] = node2
                                    found_match = True
                                    print('Merge and nodes')
                                    break
                    ind2 += 1
                    continue

                # do not try to merge amr-specific role with other nodes.
                if 'and' == doc_nodes2_init[node2][0] or 'and' == doc_nodes1_init[node1][0] or 'or' == doc_nodes2_init[node2][0] or 'or' == doc_nodes1_init[node1][0]:
                    ind2 +=1
                    continue

                # are the examined nodes visual grounded?
                if node1 in ann1.keys() and node2 in ann2.keys():

                    (obj_class_1, n_bb_1) = ann1_init[node1]
                    (obj_class_2, n_bb_2) = ann2_init[node2]

                    if n_bb_1 == n_bb_2:
                        g_dist = max_similarity_multilabel_nodes(doc_nodes1_init[node1][0], doc_nodes2_init[node2][0])
                        if n_bb_1 == n_bb_2 and obj_class_1 != 'unknown':
                            if g_dist > 0.3:
                                print(f'Grounded nouns {doc_nodes1_init[node1][0]}#{doc_nodes2_init[node2][0]}')
                                keys[node2] = node2
                                common_counter += 1
                                ind2 += 1
                                continue
                        elif obj_class_1 == 'unknown' and obj_class_2 == 'unknown':
                            if g_dist > 0.5:
                                print(f'Grounded (unknown) nouns {doc_nodes1_init[node1][0]}#{doc_nodes2_init[node2][0]}')
                                keys[node2] = node2
                                common_counter += 1
                                ind2 += 1
                                continue
                    ind2 += 1
                    continue
                
                # if one of the nodes is a visually grounded noun, do not try to merge them with other nodes.
                if (node1 in ann1.keys()) or (node2 in ann2.keys()):
                    ind2 += 1
                    continue

                g_dist = max_similarity_multilabel_nodes(doc_nodes1_init[node1][0], doc_nodes2_init[node2][0])
                # for same label nodes (not nouns, not amr-specific)
                if g_dist == 1:
                    res = re.findall("-[0-9]+", doc_nodes1_init[node1][0])
                    # if they are predicate nodes
                    if len(res) != 0:
                        parents_node2 = get_entity_parents(doc_nodes2_init, node2, doc_edges2_init,1)
                        parents_node1 = get_entity_parents(doc_nodes1_init, node1, doc_edges1_init,1)

                        if len(parents_node1) > 0 and len(parents_node2) > 0:
                            found_match = False
                            for par_1 in parents_node1:
                                if found_match:
                                    break
                                for par_2 in parents_node2:
                                    dist_temp = max_similarity_multilabel_nodes(doc_nodes1_init[par_1][0], doc_nodes2_init[par_2][0])
                                    if dist_temp >= glove_threshold and (doc_nodes2_init[node2][0] not in doc_nodes1_init[par_1][0]) and (doc_nodes1_init[node1][0] not in doc_nodes2_init[par_2][0]):  
                                            keys[node2] = node2
                                            common_counter += 1
                                            found_match = True
                                            print(f'Predicate nodes {doc_nodes1_init[node1][0]}#{doc_nodes2_init[node2][0]}')
                                            break
                        else:
                            print(f'Grounded nouns {doc_nodes1_init[node1][0]}#{doc_nodes2_init[node2][0]}')
                            common_counter += 1
                            keys[node2] = node2
                    else:
                        # a non-predicate node
                        print(f'Merge non-predicate nodes {doc_nodes1_init[node1][0]}#{doc_nodes2_init[node2][0]}')
                        keys[node2] = node2
                        common_counter += 1
                    ind2 += 1
                    continue

                # merge similar meaning nodes, when they do not re-appear in the "other" graph.
                if dist[ind2,ind1] >= 0.7:
                    if (doc_nodes2_init[node2] not in doc_node1_val) and (doc_nodes1_init[node1] not in doc_node2_val):

                        parents_node2 = get_entity_parents(doc_nodes2_init, node2, doc_edges2_init, 1)
                        parents_node1 = get_entity_parents(doc_nodes1_init, node1, doc_edges1_init, 1)

                        if len(parents_node1) > 0 and len(parents_node2) > 0:
                            found_match = False
                            for par_1 in parents_node1:
                                if found_match:
                                    break
                                for par_2 in parents_node2:
                                    dist_temp = max_similarity_multilabel_nodes(doc_nodes1_init[par_1][0], doc_nodes2_init[par_2][0])
                                    if dist_temp >= glove_threshold and (doc_nodes2_init[node2][0] not in doc_nodes1_init[par_1][0]) and (doc_nodes1_init[node1][0] not in doc_nodes2_init[par_2][0]):  
                                            keys[node2] = node2
                                            common_counter += 1
                                            found_match = True
                                            print(f'Merge semantic similar nodes {doc_nodes1_init[node1][0]}#{doc_nodes2_init[node2][0]} with similarity {dist[ind2, ind1]}')
                                            break
                        else:
                            keys[node2] = node2
                            common_counter += 1 
                            print(f'Merge semantic similar nodes {doc_nodes1_init[node1][0]}#{doc_nodes2_init[node2][0]} with similarity {dist[ind2, ind1]}')

                        ind2 +=1
                        continue


                # Merge predicate nodes based on ARGx and glove vecs similarity
                res1 = re.findall("-[0-9]+", doc_nodes1_init[node1][0])
                res2 = re.findall("-[0-9]+", doc_nodes2_init[node2][0])

                # merge predicate nodes
                if len(res1) > 0 and len(res2) > 0:
                    parents_node2, children_node2 = get_predicate_parentARGnodes(doc_nodes2_init, node2, doc_edges2_init,1)
                    parents_node1, children_node1 = get_predicate_parentARGnodes(doc_nodes1_init, node1, doc_edges1_init,1)

                    g_sim = max_similarity_multilabel_nodes(doc_nodes1_init[node1][0], doc_nodes2_init[node2][0])

                    if ':ARG0' in children_node1 and ':ARG0' in children_node2:
                        if children_node1[':ARG0'] == children_node2[':ARG0']:
                            if ':ARG1' in children_node1 and ':ARG1' in children_node2:
                                if children_node1[':ARG1'] == children_node2[':ARG1']:
                                    if ':ARG2' in children_node1 and ':ARG2' in children_node2:
                                        if children_node1[':ARG2'] == children_node2[':ARG2']:
                                            print('Connect labels ' + label1 + ' and ' + label2)
                                            keys[node2] = node2
                                            common_counter += 1
                                            print(f'Merge predicates {doc_nodes1_init[node1][0]}#{doc_nodes2_init[node2][0]} with similar semantic structure')
                                        else:
                                            if g_sim > 0.7:
                                                print('Connect labels ' + label1 + ' and ' + label2)
                                                keys[node2] = node2
                                                common_counter += 1
                                                print(f'Merge predicates {doc_nodes1_init[node1][0]}#{doc_nodes2_init[node2][0]} with similar semantic structure')

                                    else:
                                        if g_sim > 0.8:
                                            print('Connect labels ' + label1 + ' and ' + label2)
                                            keys[node2] = node2
                                            common_counter += 1
                                            print(f'Merge predicates {doc_nodes1_init[node1][0]}#{doc_nodes2_init[node2][0]} with similar semantic structure')

                            elif ':ARG1' not in children_node1 or ':ARG1' not in children_node2:
                                if g_sim > 0.8:
                                    print('Connect labels ' + label1 + ' and ' + label2)
                                    keys[node2] = node2
                                    common_counter += 1
                                    print(f'Merge predicates {doc_nodes1_init[node1][0]}#{doc_nodes2_init[node2][0]} with similar semantic structure')
                ind2 += 1
                

            for node2 in keys:
                label1 = doc_nodes1[node1][0].split('#^#')
                label2 = doc_nodes2[node2][0].split('#^#')

                isEqual = False
                for l1 in label1:
                    for l2 in label2:
                        if l1 == l2:
                            isEqual = True

                if not isEqual:
                    new_label = doc_nodes1[node1][0] + '#^#' + doc_nodes2[node2][0]
                    print('The new label is ' + new_label)
                    doc_nodes1.update({node1: tuple((new_label,))})

                doc_nodes2[node1] = doc_nodes1[node1]

                keys2update = {}
                attr2update = {}

                for attr in doc_attrs2:
                    # ipdb.set_trace()
                    if attr[0] == node2:
                        attr2update[attr] = attr

                for key in attr2update:
                    if key in doc_attrs2:
                        doc_attrs2[tuple((node1,))] = doc_attrs2[key]
                        del doc_attrs2[key]

                for edge in doc_edges2:
                    if node2 == edge[0]:
                        keys2update[edge] = 0
                    if node2 == edge[1]:
                        keys2update[edge] = 1

                for key in keys2update:
                    if keys2update[key] == 0:
                        doc_edges2.update({tuple((node1, key[1])): doc_edges2[key]})
                    else:
                        doc_edges2.update({tuple((key[0], node1)): doc_edges2[key]})

                for key in keys2update:
                    del doc_edges2[key]
                nodes2del[node2] = node2

            ind1 += 1

        for node2 in nodes2del:    
            del doc_nodes2[node2]
            if node2 in ann2.keys():
                del ann2[node2]

        if common_counter == 0:
            meta_root = {}; meta_nodes = {}; meta_edges = {}; meta_attrs = {}; meta_ann = {}

            if doc_root_nodes1[0] == tuple(('multi-sentence',)):

                meta_root = doc_root_nodes1.copy()

                meta_nodes = doc_nodes1.copy()

                meta_ann = ann1.copy()
                meta_ann['sentence'] = ann1['sentence'] + ' ' + ann2['sentence']

                max_snt = 0
                for edge in doc_edges1:
                    if 'snt' in doc_edges1[edge]:
                        tmp_snt = int(doc_edges1[edge].split('nt')[1])
                        if tmp_snt > max_snt:
                            max_snt = tmp_snt

                meta_edges = doc_edges1.copy()

                meta_edges[tuple((0,len_node1))] = ':snt' + str(max_snt+1)

                meta_attrs = doc_attrs1.copy()

                for node in doc_nodes2:
                    meta_nodes[node] = doc_nodes2[node]
                    if node in ann2.keys():
                        meta_ann[node] = ann2[node]

                for edge in doc_edges2:
                    meta_edges[tuple((edge[0],edge[1]))] = doc_edges2[edge]

                for attr in doc_attrs2:
                    meta_attrs[tuple((attr[0],))] = doc_attrs2[attr]

                return meta_root, meta_nodes, meta_edges, meta_attrs, meta_ann

            meta_root = doc_root_nodes1.copy()
            meta_root[0] = tuple(('multi-sentence',))

            meta_ann['sentence'] = ann1['sentence'] + ' ' + ann2['sentence']

            for node in doc_nodes1:
                meta_nodes[node+1] = doc_nodes1[node]
                if node in ann1.keys():
                    meta_ann[node+1] = ann1[node]

            meta_nodes[0] = meta_root[0]

            for edge in doc_edges1:
                meta_edges[tuple((edge[0]+1,edge[1]+1))] = doc_edges1[edge]

            meta_edges[tuple((0,1))] = ':snt1'

            for attr in doc_attrs1:
                meta_attrs[tuple((attr[0]+1,))] = doc_attrs1[attr]

            for node in doc_nodes2:
                meta_nodes[node+1] = doc_nodes2[node]
                if node in ann2.keys():
                    meta_ann[node+1] = ann2[node]

            for edge in doc_edges2:
                meta_edges[tuple((edge[0]+1,edge[1]+1))] = doc_edges2[edge]

            meta_edges[tuple((0,len_node1+1))] = ':snt2'

            for attr in doc_attrs2:
                meta_attrs[tuple((attr[0]+1,))] = doc_attrs2[attr]

            
            meta_nodes, meta_edges = subset_edges(meta_nodes, meta_edges, meta_ann, glove_vecs)

            return meta_root, meta_nodes, meta_edges, meta_attrs, meta_ann

        meta_root = doc_root_nodes1 # keep as root the one coming from the largest graph.
        meta_nodes = doc_nodes1 | doc_nodes2 
        meta_edges = doc_edges1 | doc_edges2   
        meta_attrs = doc_attrs1 | doc_attrs2
        ann1['sentence'] = ann1['sentence'] + ' ' + ann2['sentence']
        ann2['sentence'] = ann1['sentence']
        meta_ann = ann1 | ann2

        print(">>>Common nodes in two graphs " + str(common_counter))

        meta_root, meta_nodes, meta_edges, meta_attrs, meta_ann = renumber_meta_graph(meta_root, meta_nodes, meta_edges, meta_attrs, meta_ann)

        meta_nodes, meta_edges = subset_edges(meta_nodes, meta_edges, meta_ann, glove_vecs)

        g, meta_nodes, meta_edges, meta_root = _graph(meta_nodes, meta_edges, meta_attrs, meta_root, hyper_path=hyper)
        g = _disjointAMR_v2(g, meta_nodes, meta_edges)

        _, meta_nodes, meta_edges, meta_attrs, meta_root, meta_ann = _revert_edges(g, meta_root, root_ix=0, ann = meta_ann)

        return meta_root, meta_nodes, meta_edges, meta_attrs, meta_ann

    def renumber_meta_graph(meta_root, meta_nodes, meta_edges, meta_attrs, meta_ann):

        for i in range(len(meta_nodes)):
            if i not in meta_nodes.keys():
                for j in range(100): ##TODO: change to max key value of the meta-amr
                    if j>i and (j in meta_nodes.keys()):

                        if i == 0:
                            meta_root[i] = meta_root[j]
                            del meta_root[j]
                            if j in meta_ann.keys():
                                meta_ann[i] = meta_ann[j]
                                del meta_ann[j]

                        meta_nodes[i] = meta_nodes[j]
                        del meta_nodes[j]
                        
                        if j in meta_ann.keys():
                            meta_ann[i] = meta_ann[j]
                            del meta_ann[j]

                        if tuple((j,)) in meta_attrs:
                            meta_attrs[tuple((i,))] = meta_attrs[tuple((j,))]
                            del meta_attrs[tuple((j,))]

                        keys2update = {}
                        for edge in meta_edges:   
                            if j == edge[0]:
                                keys2update[edge] = 0
                            if j == edge[1]:
                                keys2update[edge] = 1

                        for key in keys2update:
                            if keys2update[key] == 0:
                                meta_edges.update({tuple((i, key[1])): meta_edges[key]})
                            else:
                                meta_edges.update({tuple((key[0], i)): meta_edges[key]})

                        keys2remove = {}
                        for edge in meta_edges:   
                            if edge[0] == j or edge[1] == j:
                                keys2remove[edge] = edge
                        for key in keys2remove:
                            del meta_edges[key]

                        break

        return meta_root, meta_nodes, meta_edges, meta_attrs, meta_ann
    
    def save_amr_graph(pair_fname, numcaptions, doc_nodes, doc_edges, doc_attrs, doc_root_nodes, amr_id, id, separate_meta_corpus, flag_remove_redundant_paths=False, glove_vecs = {}, ann = {}, sampling_fname = None):
        
        try:
            if flag_remove_redundant_paths == True:
                g, nodes, edges, doc_root_nodes = _graph(doc_nodes, doc_edges, doc_attrs, doc_root_nodes, hyper_path=hyper)
                g = _disjointAMR_v2(g, nodes, edges)
                _, sub_nodes, sub_edges, sub_attrs, sub_root_nodes, ann = _revert_edges(g, doc_root_nodes, root_ix=0, flag_remove_redundant_paths=True, glove_vecs=glove_vecs, ann=ann)


                g_str, nodes, edges, doc_root_nodes = _graph(sub_nodes, sub_edges, sub_attrs, sub_root_nodes, hyper_path=hyper)
                g_str, doc_edges = sampling(sampling_fname, str(id), g_str, nodes, edges, doc_root_nodes, ann)
                sub_graphs, disc = _disjointAMR(amr_id, g_str, nodes, edges, doc_root_nodes)

                # 3. Prepare new meta data
                flag = False
                new_meta = {}
                snt_list = []

                for i in range(numcaptions):
                    v_sepa = separate_meta_corpus[str(amr_id.split('.')[0]) + '.' + str(i)]
                    snt_list.append(v_sepa['snt'])
                    if 'status' not in v_sepa:
                        flag = True
                new_meta['snt'] = '##'.join(snt_list)
                new_meta['id'] = str(id)
                if not flag:
                    new_meta['status'] = 'ParsedStatus.OK'
                else:
                    new_meta['status'] = 'ParsedStatus.hasDummys'
                # new_meta['source'] = v_sepa['file']
                new_meta['snt-type'] = 'amr-merge'


                for _, r_g in sub_graphs.items():
                    g, _ = r_g
                    '''
                    merged one
                    '''
                    pen_graph = penman.encode(g)
                    for ind in range(5):
                        pair_fname.write(' '.join(['# ::status'] + [new_meta['status']]) + '\n')
                        pair_fname.write(' '.join(['# ::id'] + [new_meta['id'].split('.')[0] + '.' + str(ind)]) + '\n')
                        pair_fname.write(' '.join(['# ::snt-type'] + [new_meta['snt-type']]) + '\n')
                        pair_fname.write(' '.join(['# ::snt'] + [new_meta['snt']]) + '\n')
                        pair_fname.write(pen_graph + '\n')
                        pair_fname.write('\n')

                    # visualize_amr_graph(pen_graph, new_meta['id'])
            else:
                # split into several sub-graph if not fully connected
                g_str, nodes, edges, doc_root_nodes = _graph(doc_nodes, doc_edges, doc_attrs, doc_root_nodes, hyper_path=hyper)

                # augmentations saving....
                augmentations_graph = sample_node_label(copy.deepcopy(g_str))
                visuallyGround_AMRnodes(augmentations_graph, ann, id)
                save_augmentations_graph(sampling_fname, augmentations_graph, 'amr-merge', id)

                # 2. Get sub-graphs with its own root ndoes
                sub_graphs, disc = _disjointAMR(amr_id, g_str, nodes, edges, doc_root_nodes)

                if disc:
                    return

                # 3. Prepare new meta data
                flag = False
                new_meta = {}
                snt_list = []

                for i in range(numcaptions):
                    v_sepa = separate_meta_corpus[str(amr_id.split('.')[0]) + '.' + str(i)]
                    snt_list.append(v_sepa['snt'])
                    if 'status' not in v_sepa:
                        flag = True
                new_meta['snt'] = '##'.join(snt_list)
                new_meta['id'] = str(id)
                if not flag:
                    new_meta['status'] = 'ParsedStatus.OK'
                else:
                    new_meta['status'] = 'ParsedStatus.hasDummys'
                # new_meta['source'] = v_sepa['file']
                new_meta['snt-type'] = 'amr-merge'

                for _, r_g in sub_graphs.items():
                    g, _ = r_g
                    '''
                    merged one
                    '''
                    pen_graph = penman.encode(g)
                    pair_fname.write(' '.join(['# ::status'] + [new_meta['status']]) + '\n')
                    pair_fname.write(' '.join(['# ::id'] + [new_meta['id']]) + '\n')
                    pair_fname.write(' '.join(['# ::snt-type'] + [new_meta['snt-type']]) + '\n')
                    pair_fname.write(' '.join(['# ::snt'] + [new_meta['snt']]) + '\n')
                    pair_fname.write(pen_graph + '\n')
                    pair_fname.write('\n')

                    # visualize_amr_graph(pen_graph, new_meta['id'])
        except Exception as e:
            print("EXCEPTION in save amr graph" + str(e))

    def visualize_amr_graph(amr_graph, img_name):
        plot = AMRPlot(format='eps', render_fn='show')
        plot.build_from_graph(amr_graph, debug=False)
        plot.view(img_name)
        return

    # when amr node has multiple labels sampled one of them.
    def sample_node_label(doc_nodes):
        for ii in range(len(doc_nodes)):
            try:
                label = doc_nodes[ii][2]
                mlabel = label.split('#^#')
                len_mlabel = len(mlabel)
                if len_mlabel > 1:
                    idx = random.randint(0, len_mlabel-1)
                    doc_nodes[ii] = tuple((doc_nodes[ii][0], doc_nodes[ii][1], mlabel[idx]))
            except Exception as ex:
                try:
                    # was it a single node?
                    label = doc_nodes[2]
                    mlabel = label.split('#^#')
                    len_mlabel = len(mlabel)
                    if len_mlabel > 1:
                        idx = random.randint(0, len_mlabel-1)
                        doc_nodes = tuple((doc_nodes[0], doc_nodes[1], mlabel[idx]))
                    return doc_nodes
                except Exception as ex:
                    print(str(ex))
        return doc_nodes

    def sample_edges(g, edges):
        g_final = g.copy()
        for ii in range(len(g)):
            try:
                role = g[ii][1]
                if role not in ':instance' and 'snt' not in role and 'ARG' not in role and 'quant' not in role:
                    n1 = g[ii][0].split('z')[1]
                    n2 = g[ii][2].split('z')[1]
                    for jj in range(len(edges)):
                        if edges[jj] == tuple((int(n1),int(n2))):
                            edges.remove(edges[jj])
                            g_final.remove(g[ii])
                            break
            except Exception as ex:
                print(str(ex))
        return g_final, edges

    def sample_from_predicate_node(pred_node, g_str, filterPredRoles, counter, fname, list_predicates, pred, ann):

        pred_node_index = g_str.index(pred_node)
        g_str = sample_node_label(g_str)
        pred_node = g_str[pred_node_index]

        flag = True
        sampled_graph = []
        sampled_graph.append(pred_node)
        new_connections = []
        new_connections.append(pred_node[0])
        
        cur_connections = []
        visited_connections = []

        edgeroles = {}

        #shuffle graph connections
        random.shuffle(g_str)
        if filterPredRoles is not None:
            cur_connections = new_connections.copy()
            new_connections = []
            for idx_cur_conn in range(len(cur_connections)):
                for connection in g_str:
                    if connection[1] != 'instance':
                        if '-of' in connection[1]:
                            if connection[1].split('-of')[0] in filterPredRoles:
                                if cur_connections[idx_cur_conn] == connection[2]:
                                    if connection in sampled_graph:
                                        continue

                                    if connection[2] not in edgeroles:
                                        sampled_graph.append(connection)
                                        edgeroles[connection[2]] = [connection[1].split('-of')[0]]

                                        if connection[0] not in visited_connections:
                                            new_connections.append(connection[0])
                                    else:
                                        if connection[1].split('-of')[0] not in edgeroles[connection[2]]:
                                            sampled_graph.append(connection)
                                            edgeroles[connection[2]].append(connection[1].split('-of')[0])

                                            if connection[0] not in visited_connections:
                                                new_connections.append(connection[0])
                        else:
                            if connection[1] in filterPredRoles:
                                if cur_connections[idx_cur_conn] == connection[0]:
                                    if connection in sampled_graph:
                                        continue

                                    if connection[0] not in edgeroles:
                                        sampled_graph.append(connection)
                                        edgeroles[connection[0]] = [connection[1]]

                                        if connection[2] not in visited_connections:
                                            new_connections.append(connection[2])
                                    else:
                                        if connection[1] not in edgeroles[connection[0]]:
                                            sampled_graph.append(connection)
                                            edgeroles[connection[0]].append(connection[1])

                                            if connection[2] not in visited_connections:
                                                new_connections.append(connection[2])

                visited_connections.append(cur_connections[idx_cur_conn])
            for new_c in new_connections:
                visited_connections.append(new_c)

            for node in visited_connections:
                for connection in g_str:
                    if node == connection[0] and 'instance' == connection[1]:
                        if connection not in sampled_graph:
                            sampled_graph.append(connection)
                    if node == connection[0] and connection[1] == ':quant':
                        if connection not in sampled_graph:


                            if connection[0] in edgeroles:
                                if ':quant' not in edgeroles[connection[0]]:
                                    sampled_graph.append(connection)

                                    edgeroles[connection[0]].append(connection[1])

                                    if 'z' in connection[2]:
                                        for quant in g_str:
                                            if connection[2] == quant[0] and 'instance' == quant[1]:
                                                if quant not in sampled_graph:
                                                    sampled_graph.append(quant)
                            else:
                                sampled_graph.append(connection)
                                edgeroles[connection[0]] = [connection[1]]
                                if 'z' in connection[2]:
                                    for quant in g_str:
                                        if connection[2] == quant[0] and 'instance' == quant[1]:
                                            if quant not in sampled_graph:
                                                sampled_graph.append(quant)

                    if node == connection[0] and connection[2] in visited_connections:
                        if connection not in sampled_graph:
                            sampled_graph.append(connection)
        else:

            while flag:
                cur_connections = new_connections.copy()
                new_connections = []
                for idx_cur_conn in range(len(cur_connections)):
                    for connection in g_str:
                        
                        if connection[1] != 'instance':
                            if '-of' in connection[1]:
                                if cur_connections[idx_cur_conn] == connection[2]:
                                    if connection in sampled_graph:
                                        continue

                                    if connection[2] not in edgeroles:
                                        sampled_graph.append(connection)
                                        edgeroles[connection[2]] = [connection[1].split('-of')[0]]
                                            
                                        if connection[0] not in visited_connections:
                                            new_connections.append(connection[0])
                                    else:
                                        if connection[1].split('-of')[0] not in edgeroles[connection[2]]:
                                            sampled_graph.append(connection)
                                            edgeroles[connection[2]].append(connection[1].split('-of')[0])

                                            if connection[0] not in visited_connections:
                                                new_connections.append(connection[0])
                            else:
                                if cur_connections[idx_cur_conn] == connection[0]:
                                    if connection in sampled_graph:
                                        continue

                                    if connection[0] not in edgeroles:
                                        sampled_graph.append(connection)
                                        edgeroles[connection[0]] = [connection[1]]

                                        if connection[2] not in visited_connections:
                                            new_connections.append(connection[2])
                                    else:
                                        if connection[1] not in edgeroles[connection[0]]:
                                            sampled_graph.append(connection)
                                            edgeroles[connection[0]].append(connection[1])

                                            if connection[2] not in visited_connections:
                                                new_connections.append(connection[2])

                    visited_connections.append(cur_connections[idx_cur_conn])
                if len(new_connections) == 0:
                    flag = False

            for node in visited_connections:
                for connection in g_str:
                    if node == connection[0] and 'instance' == connection[1]:
                        if connection not in sampled_graph:
                            sampled_graph.append(connection)
        if len(visited_connections) < 3:
            return None
        
        amr_g_id = img_id.split('.')[0] + '_' + list_predicates[pred][0] + '_' + list_predicates[pred][2] + '_' + str(counter)
        amr_type = list_predicates[pred][2]
        visuallyGround_AMRnodes(sampled_graph, ann, amr_g_id)
        save_augmentations_graph(fname=fname, sampled_graph=sampled_graph,amr_type=amr_type, amr_g_id=amr_g_id)

        return sampled_graph
        
    def find_predicate_ARGx_roles(pred_node, g_str):

        AMR_ARGx_list = [':ARG0', ':ARG1', ':ARG2', ':ARG3', ':ARG4', ':ARG5']

        predicate_ARGx_roles = []

        cur_connections = []
        cur_connections.append(pred_node[0])

        for idx_cur_conn in range(len(cur_connections)):
            for connection in g_str:
                if connection[1].split('-of')[0] in AMR_ARGx_list:
                    if '-of' in connection[1]:
                        if cur_connections[idx_cur_conn] == connection[2]:
                            if connection[1].split('-of')[0] not in predicate_ARGx_roles:
                                    predicate_ARGx_roles.append(connection[1].split('-of')[0])
                    else:
                        if cur_connections[idx_cur_conn] == connection[0]:
                                if connection[1] not in predicate_ARGx_roles:
                                    predicate_ARGx_roles.append(connection[1])
        predicate_ARGx_roles.sort()
        return predicate_ARGx_roles
        
    def save_augmentations_graph(fname, sampled_graph, amr_type, amr_g_id):

        g = Graph(sampled_graph)
        pen_sampled_graph = penman.encode(g)

        fname.write(' '.join(['# ::status'] + ['ok']) + '\n')
        fname.write(' '.join(['# ::id'] + [amr_g_id]) + '\n')
        fname.write(' '.join(['# ::snt-type'] + [amr_type]) + '\n')
        fname.write(' '.join(['# ::snt'] + [amr_type]) + '\n')
        fname.write(pen_sampled_graph + '\n')
        fname.write('\n')

        return

    def sampling(fname, img_id, g_str, doc_nodes, doc_edges, doc_root_nodes, ann):
        list_predicates = {}

        AMR_ARGx_list = [':ARG0', ':ARG1', ':ARG2', ':ARG3', ':ARG4', ':ARG5']

        for ii in range(len(g_str)):
            if g_str[ii][1] == 'instance':
                label = g_str[ii][2]
                res = re.findall("-[0-9]+", label)
                if len(res) > 0:
                    list_predicates[ii] = g_str[ii]

        for pred in list_predicates:

            try:
                if list_predicates[pred][2] == 'include-91':
                    continue
                pred_roleid = re.findall("-[0-9]+", list_predicates[pred][2])
                pred_name = list_predicates[pred][2].split(pred_roleid[0])[0]
                if pred_name + '.xml' in propbank_predicate_list:
                    pred_ARGs = find_predicate_ARGx_roles(list_predicates[pred], g_str)
                    counter = 0
                    if len(pred_ARGs) > 0:

                        sample_from_predicate_node(list_predicates[pred], copy.deepcopy(g_str), AMR_ARGx_list[:3], counter, fname, list_predicates, pred, ann)
                        counter += 1
                        if AMR_ARGx_list[3] in pred_ARGs:
                            sample_from_predicate_node(list_predicates[pred], copy.deepcopy(g_str), AMR_ARGx_list[:4], counter, fname, list_predicates, pred, ann)
                            counter += 1
                        if AMR_ARGx_list[4] in pred_ARGs:
                            sample_from_predicate_node(list_predicates[pred], copy.deepcopy(g_str), AMR_ARGx_list[:5], counter, fname, list_predicates, pred, ann)
                            counter += 1
                        if AMR_ARGx_list[5] in pred_ARGs:
                            sample_from_predicate_node(list_predicates[pred], copy.deepcopy(g_str), AMR_ARGx_list, counter, fname, list_predicates, pred, ann)
                            counter += 1

                    sample_from_predicate_node(list_predicates[pred], copy.deepcopy(g_str), None, counter, fname, list_predicates, pred, ann)
                    counter += 1
            except Exception as e:
                print(str(e))
                continue

        g_str, doc_edges = sample_edges(g_str, doc_edges)
        return g_str, doc_edges

    def visuallyGround_AMRnodes(doc_nodes, ann, id):
        nodes_bboxes = {}
        for node in doc_nodes:
            if 'instance' in node[1]:
                node_id = int(node[0].split('z')[1])
                if node_id in ann:
                    node_ann = ann[node_id]
                    node_phraseID = node_ann[0]
                    if node_phraseID != 'unknown':
                        node_bboxes =  str(node_ann[1])
                        nodes_bboxes[str(node_id) + '_' + node[2]] = node_bboxes
                    elif node_phraseID == 'unknown':
                        nodes_bboxes[str(node_id) + '_' + node[2]] = 'unknown'
        with open(f"{output_nodevginfo}{id}.json", "w") as outfile:
            json.dump(nodes_bboxes, outfile)           

    def caption_merging(img_id, hierchical_clusters, image_captions_ids, body_corpus, separate_meta_corpus):
        # initializations ...
        meta_root = {}; meta_nodes = {}; meta_edges = {}; meta_attrs = {}; meta_ann = {}; ann = {}

        cluster_index = len(image_captions_ids)

        for cluster in hierchical_clusters:
            cluster_0 = int(cluster[0])
            cluster_1 = int(cluster[1])
            if cluster_0<len(image_captions_ids):
                key1 = image_captions_ids[cluster_0]
                doc_nodes1, doc_root_nodes1, doc_edges1, doc_attrs1 = body_corpus[key1]
                annotations_1 = annotations[key1]
            else:
                key1 = 'meta' + str(cluster_0)
                doc_nodes1, doc_root_nodes1, doc_edges1, doc_attrs1 = meta_nodes[cluster_0], meta_root[cluster_0], meta_edges[cluster_0], meta_attrs[cluster_0]
                annotations_1 = meta_ann[cluster_0]
            if cluster_1 < len(image_captions_ids):
                key2 = image_captions_ids[cluster_1]
                annotations_2 = annotations[key2]
                doc_nodes2, doc_root_nodes2, doc_edges2, doc_attrs2 = body_corpus[key2]
            else:
                key2 = 'meta' + str(cluster_1)
                doc_nodes2, doc_root_nodes2, doc_edges2, doc_attrs2 = meta_nodes[cluster_1], meta_root[cluster_1], meta_edges[cluster_1], meta_attrs[cluster_1]
                annotations_2 = meta_ann[cluster_1]

                                         
            try:
                if len(doc_nodes1) >= len(doc_nodes2):
                    meta_root[cluster_index], meta_nodes[cluster_index], meta_edges[cluster_index], meta_attrs[cluster_index], meta_ann[cluster_index] = \
                        pairwise_merge(doc_nodes1, doc_root_nodes1, doc_edges1, doc_attrs1, annotations_1,\
                        doc_nodes2, doc_root_nodes2, doc_edges2, doc_attrs2, annotations_2, glove_vecs, cluster[2])
                else:
                    meta_root[cluster_index], meta_nodes[cluster_index], meta_edges[cluster_index], meta_attrs[cluster_index], meta_ann[cluster_index]  = \
                        pairwise_merge(doc_nodes2, doc_root_nodes2, doc_edges2, doc_attrs2, annotations_2,\
                        doc_nodes1, doc_root_nodes1, doc_edges1, doc_attrs1, annotations_1, glove_vecs, cluster[2])

                meta_root[cluster_index], meta_nodes[cluster_index], meta_edges[cluster_index], meta_attrs[cluster_index], meta_ann[cluster_index] = \
                    renumber_meta_graph(meta_root[cluster_index], meta_nodes[cluster_index], meta_edges[cluster_index], meta_attrs[cluster_index], meta_ann[cluster_index])

                merge_amr_id = str(img_id) + '_' + str(cluster_0) + str(cluster_1)
                save_amr_graph(f_meta_amrs, len(image_captions_ids), meta_nodes[cluster_index], meta_edges[cluster_index],  meta_attrs[cluster_index], meta_root[cluster_index], \
                            image_captions_ids[0], merge_amr_id, separate_meta_corpus, ann=meta_ann[cluster_index], sampling_fname=f_sampled_amrs)
                
            except Exception as e:
                print(str(e))

            cluster_index += 1
                   
        doc_nodes = meta_nodes[cluster_index-1]
        doc_edges = meta_edges[cluster_index-1]
        doc_attrs = meta_attrs[cluster_index-1]
        doc_root_nodes = meta_root[cluster_index-1]
        ann = meta_ann[cluster_index-1]

        try:
            # for predicate based sampling augmentations
            save_amr_graph(f_meta_amrs, len(image_captions_ids), doc_nodes, doc_edges, doc_attrs, doc_root_nodes, \
                        image_captions_ids[0], str(image_captions_ids[0]), separate_meta_corpus, flag_remove_redundant_paths=True, glove_vecs=glove_vecs, ann=ann, sampling_fname=f_sampled_amrs)
        except Exception as e:
            print(str(e))
        return doc_root_nodes, doc_nodes, doc_edges, doc_attrs, ann

    glove_vecs = load_glove_vecs(glove)

    with open(coco_entities_fname, 'r') as fp:  
        coco_entities = json.load(fp)

    propbank_predicate_list = os.listdir(propbank)
    f_meta_amrs = open(output_meta_amrs, 'w')
    f_sampled_amrs = open(output_sampledamrs, 'w')

    body_corpus, amr_text, annotations = load(body_file, coco_entities=coco_entities, trim=trim)
    separate_meta_corpus = utils.read_amr_meta(body_file) # information about AMR, id, annotator, sentence etc "#" entr

    image_captions_ids = []
    img_id = None
    for curr_filename in tqdm(sorted(list(body_corpus.keys()))):
        if img_id == None:
            img_id = curr_filename.split('.')[0]
            image_captions_ids.append(curr_filename)
            continue
        if img_id != curr_filename.split('.')[0]:
            try:
                hierchical_clusters = hierarchical_clustering_with_smatch_score(image_captions_ids, amr_text)
                caption_merging(img_id, hierchical_clusters, image_captions_ids, body_corpus, separate_meta_corpus)
            except Exception as e:
                print(str(e))

            img_id = curr_filename.split('.')[0]
            image_captions_ids = []
            image_captions_ids.append(curr_filename)
        else:
            image_captions_ids.append(curr_filename)
            continue

    f_meta_amrs.close()
    f_sampled_amrs.close()


if __name__ == '__main__':
    print('Loading args')
    args = parse_args(True)

    AMR_pipeline(args.caption_amrs, args.sampledamrs, args.meta_amrs, args.nodevisualgroundinfo, args.propbank, args.glove, args.glove_threshold, args.coco_entities)