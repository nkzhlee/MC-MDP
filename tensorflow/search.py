# -*- coding:utf8 -*-

import os

import time
import logging
import json
import types
import copy_reg
import Queue
from mctree import search_tree
import numpy as np
import tensorflow as tf
from collections import Counter
from utils import compute_bleu_rouge
from utils import normalize
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder


def list2string(lis):
    strin = ''
    for l in lis:
        strin += l
    return strin

def precision_recall_f1(prediction, ground_truth):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction
    if not isinstance(ground_truth, list):
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1

def recall(prediction, ground_truth):
    """
    This function calculates and returns the recall
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of recall
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[1]


def f1_score(prediction, ground_truth):
    """
    This function calculates and returns the f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of f1
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[2]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        metric_fn: metric function pointer which calculates scores according to corresponding logic.
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

class SearchTree(object):
    """
    Implements the main reading comprehension model.
    
    
    python -u run.py --prepare --emb_files ../data/vectors.txt --train_files ../data/demo/trainset/test10 --dev_files  ../data/demo/devset/test5
     
    python -u run.py --prepare --emb_files ../data/vectors.txt --train_files ../data/corpus/search.train.json --dev_files  ../data/demo/devset/search.dev.json 
    
    Z
    python -u run.py --train --algo MCST --draw_path ./log/haha --epochs 2 --search_time 5 --max_a_len 3  --beta 10 --batch_size 1 --max_p_len 10000 --hidden_size 150  --train_files ../data/demo/trainset/test10 --dev_files  ../data/demo/devset/test5 --test_files ../data/demo/test/search.test.json
    
    python -u run.py --train --algo MCST --draw_path ./log/test --epochs 2 --search_time 5 --max_a_len 3  --beta 1 --batch_size 1 --max_p_len 10000 --hidden_size 150  --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/test/search.test.json
    
    nohup python -u run.py --train --algo MCST --draw_path ./log/2 --epochs 30 --search_time 5000 --max_a_len 25  --beta 7 --batch_size 1 --max_p_len 10000 --hidden_size 150  --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/test/search.test.json ../data/demo/test/search.test.json >beta_7_30_5000_25.txt 2>&1 &
    
    python -u run.py --train --algo MCST --draw_path ./log/haha --epochs 2 --search_time 5 --max_a_len 3  --beta 10 --batch_size 1 --max_p_len 10000 --hidden_size 150  --train_files ../data/demo/trainset/test10 --dev_files  ../data/demo/devset/test5 --test_files ../data/demo/test/search.test.json
    
    nohup python -u run.py --train --algo MCST --draw_path ./log/1 --gpu 0 --epochs 30 --search_time 10 --max_a_len 5 --beta 100 --batch_size 1 --max_p_len 10000 --hidden_size 150  --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/test/search.test.json ../data/demo/test/search.test.json >beta_100_30_50_10.txt 2>&1 &
    
    python -u run.py --train --algo MCST --emb_files ../data/vectors.txt --draw_path ./log/test --epochs 1 --search_time 100 --max_a_len 3  --beta 1 --batch_size 1 --max_p_len 10000 --hidden_size 150  --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/test/search.test.json

    python -u run.py --train --algo MCST  --draw_path ./log/test --epochs 1 --search_time 30 --max_a_len 3 --beta 10 --batch_size 1 --max_p_len 100 --hidden_size 150  --train_files ../data/demo/trainset/test10 --dev_files  ../data/demo/devset/test5 --test_files ../data/demo/test/search.test.json
    
    nohup python -u run.py --train --algo MCST  --draw_path ./log/test100 --epochs 30 --search_time 3000 --max_a_len 5 --beta 100 --batch_size 1 --max_p_len 1000 --hidden_size 150  --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/test/search.test.json >beta_100_30_5_3000.txt 2>&1 &
    
    python -u run.py --train --algo MCST  --draw_path ./log/test --epochs 3 --search_time 30 --max_a_len 5 --beta 10 --batch_size 1 --max_p_len 1000 --hidden_size 150  --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/test/search.test.json 
    
    
    ﻿nohup python -u run.py --train --algo MCST --learning_rate 0.001 --RougeL 10 --Bleu1 2 --Bleu2 2 --Bleu3 2 --Bleu4 10 --draw_path ./log/test --epochs 1 --search_time 3 --max_a_len 5 --beta 50 --batch_size 1 --max_p_len 10000 --hidden_size 150  --train_files ../data/demo/trainset/test10 --dev_files  ../data/demo/devset/test5 --test_files ../data/demo/test/search.test.json >test_without_P.txt 2>&1 &
    nohup python -u run.py --train --algo MCST --learning_rate 0.0001 --RougeL 10 --Bleu1 2 --Bleu2 2 --Bleu3 2 --Bleu4 10 --draw_path ./log/test --epochs 30 --beta 50 --batch_size 1 --max_p_len 10000 --hidden_size 150  --train_files ../data/demo/trainset/search.train.json --dev_files  ../data/demo/devset/search.dev.json --test_files ../data/demo/test/search.test.json >test_without_P_new.txt 2>&1 &
    """

    def __init__(self, tfg, data, max_a_len, max_search_time, beta, m_value, dropout_keep_prob):
        self.tfg = tfg
        self.data = data
        self.beta = beta
        self.m_value = m_value
        self.max_a_len = max_a_len
        self.max_search_time = max_search_time
        self.dropout_keep_prob = dropout_keep_prob
        self.l_passage = 0
    #start search
    def list2string(self, lis):
        string = ''
        for l in lis:
            string += l
        return string


    def train_analysis(self, step):

        pred_answers, ref_answers = [], []
        fake_answers = []
        ref_answers.append({'question_id': self.data['question_id'],
                                'question_type': self.data['question_type'],
                                    'answers': self.data['ref_answers']})
        listSelectedSet = []
        all_set = []
        # print '+++++++++++++++++++++++++++++++++++++++++++'
        # print ('question_id', list2string(self.tfg.vocab.recover_from_ids(self.data['question_token_ids'])))
        for p_idx, is_selected in enumerate(self.data['passage_is_selected_list'],0):
            all_set += self.data['passage_token_ids_list'][p_idx]
            if is_selected == True:
                # print 'is True'
                # print('title ', self.data['passage_title_token_ids_list'][p_idx])
                #print('title', list2string(self.tfg.vocab.recover_from_ids(self.data['passage_title_token_ids_list'][p_idx])))
                listSelectedSet += self.data['passage_token_ids_list'][p_idx]

        pred_answer_str = ''

        str123_list = self.tfg.vocab.recover_from_ids(listSelectedSet)
        all_set_list = self.tfg.vocab.recover_from_ids(all_set)
        for s in str123_list:
            pred_answer_str += s
        selected_recall_score = 1
        selected_f1_score = 0
        all_racall_score = 1
        all_f1_score = 0
        if len(self.data['segmented_answers']) > 0 and len(str123_list) >0:
            selected_recall_score = metric_max_over_ground_truths(recall, str123_list, self.data['segmented_answers'])
            selected_f1_score = metric_max_over_ground_truths(f1_score, str123_list, self.data['segmented_answers'])
        if len(self.data['segmented_answers']) > 0 and len(str123_list) > 0:
            all_racall_score = metric_max_over_ground_truths(recall, all_set_list,self.data['segmented_answers'])
            all_f1_score = metric_max_over_ground_truths(f1_score, all_set_list, self.data['segmented_answers'])

        # print pred_answer_str
        # print ('ref_answer',self.data['ref_answers'] )
        # print('fake_answers', self.data['fake_answers'])
        # print('pre_answer', [''.join(pred_answer_str)])
        pred_answer = {'question_id': self.data['question_id'],
                       'question_type': self.data['question_type'],
                       'answers': [''.join(pred_answer_str)]}
        pred_answers.append(pred_answer)
        fake_answer = {'question_id': self.data['question_id'],
                                'question_type': self.data['question_type'],
                                    'answers': self.data['fake_answers']}
        fake_answers.append(fake_answer)
        # pre VS ref
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        value_with_mcts = bleu_rouge
        # print 'ref VS pre: '
        # print value_with_mcts

        # pre VS fac
        if len(ref_answers) > 0 and len(fake_answers) > 0 :
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, fake_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        value_with_mcts = bleu_rouge
        # print 'pre VS fac: '
        # print value_with_mcts

        # pre VS fac
        if len(ref_answers) > 0 and len(self.data['fake_answers']) > 0 :
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(fake_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        value_with_mcts = bleu_rouge
        # print 'fac VS ref: '
        # print value_with_mcts
        match_score = [selected_recall_score, selected_f1_score,all_racall_score, all_f1_score]
        #print ('match_score', match_score)
        return  pred_answer,fake_answer, match_score

    def train_encode_and_select_passage(self, step):

        total_loss = 0.0

        pred_answers, ref_answers = [], []
        fake_answers = []
        ref_answers.append({'question_id': self.data['question_id'],
                                'question_type': self.data['question_type'],
                                    'answers': self.data['ref_answers']})
        listSelectedSet = []
        all_set = []
        if step%20 == 0:
            print '+++++++++++++++++++++++++++++++++++++++++++'
            print ('question_id', self.data['question_id'])
        # print 'question'
        # print  list2string(self.tfg.vocab.recover_from_ids(self.data['question_token_ids']))

        p_result_list = []
        y_list = []

        input_q = []
        input_q_length = []
        input_p = []
        input_p_length = []
        input_t = []
        input_t_length = []
        input_selected = []

        for p_idx, if_selected in enumerate(self.data['passage_is_selected_list'],0):
            #print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
            all_set += self.data['passage_token_ids_list'][p_idx]
            # print 'title:'
            # print list2string(self.tfg.vocab.recover_from_ids(self.data['passage_title_token_ids_list'][p_idx]))
            # print 'passage'
            # print list2string(self.tfg.vocab.recover_from_ids(self.data['passage_token_ids_list'][p_idx]))
            is_selected = [0.0]
            if if_selected == True:
                is_selected = [1.0]
                y_list.append(1)
                # print 'is True'
                # print('title ', self.data['passage_title_token_ids_list'][p_idx])
                #print('title', list2string(self.tfg.vocab.recover_from_ids(self.data['passage_title_token_ids_list'][p_idx])))
                listSelectedSet += self.data['passage_token_ids_list'][p_idx]
            else:
                y_list.append(0)


            #print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'

            p_length = len(self.data['passage_token_ids_list'][p_idx])
            #print ('p_length',p_length)
            q_length = len(self.data['question_token_ids'])
            #print ('q_length',q_length)
            if len(self.data['passage_title_token_ids_list'][p_idx]) == 0 or len(self.data['passage_token_ids_list'][p_idx]) == 0:
                continue
            # print '-------------------------------------'
            # print ('passage_token_ids_list', np.shape([self.data['passage_token_ids_list'][p_idx]]))
            # print ('question_token_ids', np.shape([self.data['question_token_ids']]))
            # print ('passage_title_token_ids_list', np.shape([self.data['passage_title_token_ids_list'][p_idx]]))
            # print ('[p_length]', np.shape([p_length]))
            # print ('[q_length]', np.shape([q_length]))
            # print ('passage_title_length_list', np.shape([self.data['passage_title_length_list'][p_idx]]))
            # print ('[is_selected]', np.shape([is_selected]))
            # print '-------------------------------------'

            self.tfg.set_feed_dict_train([self.data['passage_token_ids_list'][p_idx]],
                                   [self.data['question_token_ids']],
                                   [self.data['passage_title_token_ids_list'][p_idx]],
                                   [p_length],
                                   [q_length],
                                   [self.data['passage_title_length_list'][p_idx]],
                                   [is_selected],
                                   self.dropout_keep_prob)


            #q_word_encode, p_word_encode, p_encode = self.tfg.run_session_shape()
            result = self.tfg.get_p_l()
            # print ('Q_encode', np.shape(result[0]))
            # print ('p_word_encode', np.shape(result[1]))
            # print ('p_encode', np.shape(result[2]))
            # print ('t_word_encode', np.shape(result[3]))
            # print ('T_encode', np.shape(result[4]))
            result = self.tfg.cal_loss()
            p_l = result[0]
            loss = result[1]
            if p_l[0][0] > 0.5:
                p_result_list.append(1)
            else:
                p_result_list.append(0)


            total_loss += loss
            if step % 20 == 0:
                print ('p', p_l[0][0])
                print ('loss', loss)
        #print '+++++++++++++++++++++++++++++++++++++++++++'
        all_num = 0
        acc_num = 0
        for i, pl in enumerate(p_result_list, 0):
            if pl == y_list[i]:
                acc_num += 1
            all_num += 1
        if step % 20 == 0:
            print ('p_result_list', p_result_list)
            print ('y_list', y_list)
        # print ('acc_num', acc_num)
        acc = 1.0 * acc_num/all_num

        ave_loss = total_loss / all_num


        return ave_loss, acc

    def dev_encode_and_select_passage(self, step):

        total_loss = 0.0

        pred_answers, ref_answers = [], []
        fake_answers = []
        ref_answers.append({'question_id': self.data['question_id'],
                            'question_type': self.data['question_type'],
                            'answers': self.data['ref_answers']})
        listSelectedSet = []
        all_set = []
        if step % 20 == 0:
            print '+++++++++++++++++++++++++++++++++++++++++++'
            print ('question_id', self.data['question_id'])
        # print 'question'
        # print  list2string(self.tfg.vocab.recover_from_ids(self.data['question_token_ids']))

        p_result_list = []
        y_list = []
        for p_idx, if_selected in enumerate(self.data['passage_is_selected_list'], 0):
            #print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
            all_set += self.data['passage_token_ids_list'][p_idx]
            # print 'title:'
            # print list2string(self.tfg.vocab.recover_from_ids(self.data['passage_title_token_ids_list'][p_idx]))
            # print 'passage'
            # print list2string(self.tfg.vocab.recover_from_ids(self.data['passage_token_ids_list'][p_idx]))
            if len(self.data['passage_title_token_ids_list'][p_idx]) == 0 or len(self.data['passage_token_ids_list'][p_idx]) == 0:
                continue
            is_selected = [0.0]
            if if_selected == True:
                is_selected = [1.0]
                y_list.append(1)
                # print 'is True'
                # print('title ', self.data['passage_title_token_ids_list'][p_idx])
                # print('title', list2string(self.tfg.vocab.recover_from_ids(self.data['passage_title_token_ids_list'][p_idx])))
                listSelectedSet += self.data['passage_token_ids_list'][p_idx]
            else:
                y_list.append(0)

            #print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'

            p_length = len(self.data['passage_token_ids_list'][p_idx])
            #print ('p_length', p_length)
            q_length = len(self.data['question_token_ids'])
            #print ('q_length', q_length)
            self.tfg.set_feed_dict_train([self.data['passage_token_ids_list'][p_idx]],
                                         [self.data['question_token_ids']],
                                         [self.data['passage_title_token_ids_list'][p_idx]],
                                         [p_length],
                                         [q_length],
                                         [self.data['passage_title_length_list'][p_idx]],
                                         [is_selected],
                                         self.dropout_keep_prob)

            # q_word_encode, p_word_encode, p_encode = self.tfg.run_session_shape()
            result = self.tfg.get_p_l()
            # print ('Q_encode', np.shape(result[0]))
            # print ('p_word_encode', np.shape(result[1]))
            # print ('p_encode', np.shape(result[2]))
            # print ('t_word_encode', np.shape(result[3]))
            # print ('T_encode', np.shape(result[4]))
            #print ('p', result)
            if result > 0.5:
                p_result_list.append(1)
            else:
                p_result_list.append(0)

            loss = self.tfg.test_loss()[0][0]
            total_loss += loss
            #print ('loss', loss)
        #print '+++++++++++++++++++++++++++++++++++++++++++'
        acc = 0.0
        all_num = 0
        acc_num = 0
        for i, pl in enumerate(p_result_list, 0):
            if pl == y_list[i]:
                acc_num += 1
            all_num += 1

        acc = 1.0 * acc_num / all_num
        ave_loss = total_loss / all_num
        return ave_loss, acc

    def _filter(self,token_ids, length):
        new_token_ids = []
        for i,id in enumerate(token_ids,0):
            #assert isinstance(type(id),int)
            if id == 1:
                length = length -1
            else:
                new_token_ids.append(id)
        return new_token_ids, length

    def _dynamic_padding(self, batch_data, pad_id = 0 ):
        """
        Dynamically pads the batch_data with pad_id
        """
        #print 'dynamic _padding...'
        #print 'pad_id' + str(pad_id)
        max_p_len = 1000
        max_q_len =1000
        pad_p_len = min(max_p_len, max(batch_data['passage_length']))+1
        #print 'pad_p_len' + str(pad_p_len)
        pad_q_len = min(max_q_len, max(batch_data['question_length']))
        #print 'pad_q_len' + str(pad_q_len)
        #for ids in batch_data['passage_token_ids'] :
            #print 'padding: '
            #print (ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        return batch_data, pad_p_len, pad_q_len