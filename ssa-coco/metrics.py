
import os
import pickle as pkl
from multiprocessing import Pool

import munkres
import nltk
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction
from utils import ngrams

__all__ = ["distinct_n_sentence_level", "distinct_n_corpus_level"]


def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)

def get_reference(test_data):
        reference = list()
        for text in test_data:
            text = nltk.word_tokenize(text)
            reference.append(text)
        return reference

def self_bleu(test_data, gram, reference=None, is_first=True):
        ngram = gram
        bleu = list()
        reference = get_reference(test_data)
        weight = tuple((1. / ngram for _ in range(ngram)))
        iter = 0
        for hypothesis in test_data:
            hypothesis = nltk.word_tokenize(hypothesis)
            bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                smoothing_function=SmoothingFunction().method1))
            iter += 1
        return sum(bleu) / len(bleu)

def calc_bleu(reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

def get_bleu_parallel(gram, reference=None):
        ngram = gram
        weight = tuple((1. / ngram for _ in range(ngram)))
        # pool = Pool(os.cpu_count())
        results = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            result = calc_bleu(other, hypothesis, weight)
            results.append(result)

        score = 0.0
        cnt = 0
        for i in results:
            score += i
            cnt += 1
        return score / cnt

def get_selfbleu_fast(test_data, gram, sample_size, reference=None, is_first=True):
        reference = get_reference(test_data)
        reference = reference[0:sample_size]
        return get_bleu_parallel(gram, reference=reference)
