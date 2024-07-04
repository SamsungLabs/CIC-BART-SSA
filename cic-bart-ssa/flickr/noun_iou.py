import torch
import torch.nn.functional as F
import pickle as pkl
import munkres
import nltk
from nltk import word_tokenize
import inflect
import numpy as np

p = inflect.engine()


class NounIoU(object):
    def __init__(self, pre_comp_file):
        self.pre_comp_file = pre_comp_file
        self.munkres = munkres.Munkres()
        if 'pkl' in pre_comp_file:
            with open(self.pre_comp_file, 'rb') as fp:
                self.vectors = pkl.load(fp)
        else:
            self.vectors = self.load_glove_vecs(self.pre_comp_file)
        print(len(self.vectors))

    def load_glove_vecs(self, fp):
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

    def prep_seq(self, seq):
        seq = seq.split(' ')
        seq = [w for w in seq if w in self.vectors]
        return seq
    
    def prep_seq_m(self, seq):
        # seq = seq.split(' ')
        seq = [w for w in seq if w in self.vectors]
        return seq
    
    def extract_nouns(self, seq):
        tokens = word_tokenize(seq)
        parts_of_speech = nltk.pos_tag(tokens)
        nouns = list(filter(lambda x: "NN" in x[1] , parts_of_speech))
        temp = []
        for noun in nouns:
            temp.append(noun[0].lower())
        return temp
    
    def extract_nouns_stanford(self, seq):
        nouns = list(filter(lambda x: "NN" in x[1] , seq))
        temp = []
        for noun in nouns:
            temp.append(noun[0].lower())
        return temp
    
    def extract_nouns_tags(self, seq):
        nouns = word_tokenize(seq)
        temp = []
        for noun in nouns:
            temp.append(noun.lower())
        return temp
    
    def extract_NN(self, sent):
        grammar = r"""
        NBAR:
            # Nouns and Adjectives, terminated with Nouns
            {<NN.*>*<NN.*>}

        NP:
            {<NBAR>}
            # Above, connected with in/of/etc...
            {<NBAR><IN><NBAR>}
        """
        chunker = nltk.RegexpParser(grammar)
        ne = []
        chunk = chunker.parse(nltk.pos_tag(nltk.word_tokenize(sent)))
        for tree in chunk.subtrees(filter=lambda t: t.label() == 'NP'):
            temp = ' '.join([child[0] for child in tree.leaves()])
            temp = temp.split()
            ne.append(temp[-1])
        return ne

    def score(self, seq_gt, seq_pred):
        seq_gt = self.prep_seq(seq_gt)
        seq_pred = self.prep_seq(seq_pred)
        m, n = len(seq_gt), len(seq_pred)  # length of two sequences

        if m == 0:
            return 1.
        if n == 0:
            return 0.

        similarities = torch.zeros((m, n))
        for i in range(m):
            for j in range(n):
                a = self.vectors[seq_gt[i]]
                b = self.vectors[seq_pred[j]]
                a = torch.from_numpy(a)
                b = torch.from_numpy(b)
                similarities[i, j] = torch.mean(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))).unsqueeze(-1)

        similarities = (similarities + 1) / 2
        similarities = similarities.numpy()
        ass = self.munkres.compute(munkres.make_cost_matrix(similarities))

        intersection_score = .0
        for a in ass:
            intersection_score += similarities[a]
        iou_score = intersection_score / (m + n - intersection_score)

        return iou_score
            
    def score_modified(self, seq_gt, seq_pred, postags=None):
        # import ipdb;ipdb.set_trace()
        
        # seq_gt = self.extract_NN(seq_gt)
        seq_gt = self.extract_nouns_tags(seq_gt)
        if postags == None:
            seq_pred = self.extract_nouns(seq_pred)
        else:
            seq_pred = self.extract_nouns_stanford(postags)

        # for i in range(len(seq_gt)):
        #     try:
        #         res = p.singular_noun(seq_gt[i])
        #         if res != False:
        #             seq_gt[i] = res
        #     except Exception as e:
        #         continue
        
        # for i in range(len(seq_pred)):
        #     try:
        #         res = p.singular_noun(seq_pred[i])
        #         if res != False:
        #             seq_pred[i] = res
        #     except Exception as e:
        #         continue
                
        seq_gt = self.prep_seq_m(seq_gt)
        seq_pred = self.prep_seq_m(seq_pred)
        seq_gt = [*set(seq_gt)]
        seq_pred = [*set(seq_pred)]
        m, n = len(seq_gt), len(seq_pred)  # length of two sequences

        # iou_score, gtmi_percent, prmi_percent, iou_score_real, gtmi_real, prmi_real, seq_gt, seq_pred
        if m == 0 and n == 0:
            return 1, 0, 0, 1, 0, 0, seq_gt, seq_pred
        if m == 0:
            return 0, 0, 1, 0, 0, 1, seq_gt, seq_pred
        if n == 0:
            return 0, 1, 0, 0, 1, 0, seq_gt, seq_pred

        similarities = torch.zeros((m, n))
        for i in range(m):
            for j in range(n):
                a = self.vectors[seq_gt[i]]
                b = self.vectors[seq_pred[j]]
                a = torch.from_numpy(a)
                b = torch.from_numpy(b)
                similarities[i, j] = torch.abs((F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)) + 1) / 2)

        similarities = similarities.numpy()
        ass = self.munkres.compute(munkres.make_cost_matrix(similarities))

        intersection_score = .0
        intersection = 0
        for a in ass:
            intersection_score += similarities[a]
            if similarities[a] > 0.5:
                intersection += 1
            if similarities[a] < 0:
                print('negative')
        iou_score = intersection_score / (m + n - intersection_score)
        gtmi = m - intersection_score
        prmi = n - intersection_score
        gtmi_percent = gtmi / m
        prmi_percent = prmi / n
        iou_score_real = intersection / (m + n - intersection)
        gtmi_real = (m - intersection) / m
        prmi_real = (n - intersection) / n

        return iou_score, gtmi_percent, prmi_percent, iou_score_real, gtmi_real, prmi_real, seq_gt, seq_pred