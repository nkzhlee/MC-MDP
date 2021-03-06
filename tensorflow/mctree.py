

# !/usr/bin/python
# -*- coding:utf-8 -*-

from treelib import Tree
import copy
from utils import normalize
from utils import compute_bleu_rouge
import time
import types
import copy_reg
import Queue
import numpy as np
"""
    num : number of visit time
    once_num : nothing 
    Q : value funtion calculate value of now-node 
    p : policy score of now-node 
    doc : doc episode list of now-state 
"""

class node(object):
    def __init__(self):
        self.num = 0.0
        self.Q = 0.0
        self.p = 0.0
        self.sen = []
        self.value = None

def list2string(lis):
    strin = ''
    for l in lis:
        strin += l
    return strin

class search_tree(object):
    def __init__(self, mcst, q_id, p_sen_list, max_depth, max_search_time, beta,m_value, l_passages, ref_answer, vocab):
        self.tree = Tree()
        self.q_id = q_id
        self.tree.create_node(identifier='question_' + str(q_id), data=node())
        root_node = self.tree.get_node('question_' + str(q_id))
        root_node.data.num = 1.0
        self.node_map = {}
        self.l_passages = l_passages
        self.m_value = m_value
        self.ref_answer = ref_answer
        self.max_search_time = max_search_time
        self.count = 0.0
        self.carpe_diem = mcst
        self.max_depth = max_depth
        self.beta = beta
        self.vocab = vocab
        self.p_sen_list = p_sen_list
        self.expand(self.tree.get_node(self.tree.root))


    def expand(self, leaf_node):
        #print '---------------------- start expand: -----------------------'
        time_tree_start = time.time()
        sens_list = leaf_node.data.sen
        # print 'word_list:'
        # print words_list
        sens_id_list = map(eval, sens_list)
        p_sen_id_list, p_pred_list = self.carpe_diem.get_policy(sens_id_list, self.l_passages)
        # print 'candidate_id: '
        # print np.shape(p_word_id)
        # print 'p_pred'
        # print np.shape(p_pred)
        for sen in p_sen_id_list:
            self.node_map[' '.join(sens_list + [str(sen)])] = len(self.node_map)
            new_node = node()
            new_node.sen = sens_list + [str(sen)]
            #print ('new_node.sen', new_node.sen)
            new_node.p = p_pred_list[p_sen_id_list.index(sen)]
            self.tree.create_node(identifier=self.node_map[' '.join(new_node.sen)], data=new_node,
                                  parent=leaf_node.identifier)
        #print ('&&&&&&&&&&&&&&& tree expand time = %3.2f s &&&&&&&&&&&&' % (time.time() - time_tree_start))

    def update(self, node_list, value):
        #print '----update'
        for node_id in node_list:
            tmp_node = self.tree.get_node(node_id)
            tmp_node.data.Q = (tmp_node.data.Q * tmp_node.data.num + value) / (tmp_node.data.num + 1)
            tmp_node.data.num += 1

    def search(self, start_node_id):
        #print '----tree search'
        tmp_node = self.tree.get_node(start_node_id)
        #print tmp_node.data.num
        has_visit_num = tmp_node.data.num - 1
        self.count = has_visit_num

        if int(self.max_search_time - has_visit_num) > 0:
            start_node_search_time = int(self.max_search_time - has_visit_num)
        else:
            start_node_search_time = 0

        for tm in range(start_node_search_time):
            if tm%10 == 0:
                batch_start_time = time.time()
                #print ('search time',tm)
            search_list = [start_node_id]
            tmp_node = self.tree.get_node(start_node_id)
            #print 'search time :'+ str(time)

            while not tmp_node.is_leaf():
                max_score = float("-inf")
                max_id = -1
                for child_id in tmp_node.fpointer:
                    child_node = self.tree.get_node(child_id)
                    score = self.beta * child_node.data.p * (
                    (tmp_node.data.num) ** 0.5 / (1 + child_node.data.num))

                    #print 'child_node.data.Q: '
                    #print child_node.data.Q
                    score += child_node.data.Q

                    #print 'score: '
                    #print score

                    #print '**************'

                    if score > max_score:
                        max_id = child_id
                        max_score = score
                search_list.append(max_id)
                tmp_node = self.tree.get_node(max_id)

            if not tmp_node.data.value == None:
                v = tmp_node.data.value
            else:
                if tmp_node.data.sen[-1] == str(self.l_passages - 1):
                    pred_answer = tmp_node.data.sen
                    # print 'search to end  pred_answer: '
                    # print pred_answer
                    # print 'listSelectedSet'
                    listSelectedSet_sens = []
                    listSelectedSet = map(eval, pred_answer)
                    # print listSelectedSet
                    for idx in listSelectedSet:
                        listSelectedSet_sens.append(self.p_sen_list[idx])
                    # print 'pred_answer '
                    pred_answer_str = ''
                    for sen in listSelectedSet_sens:
                        str123_list = self.carpe_diem.vocab.recover_from_ids(sen, 0)
                        for s in str123_list:
                            pred_answer_str += s
                    # print 'pred_answer_str: '
                    # print pred_answer_str
                    # print 'ref_answer_str: '
                    # print list2string(self.ref_answer[0]['answers'])
                    pred_answers = []

                    pred_answers.append({'question_id': [self.q_id],
                                         'question_type': [],
                                         'answers': [''.join(pred_answer_str)],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})
                    if len(self.ref_answer) > 0:
                        pred_dict, ref_dict = {}, {}
                        for pred, ref in zip(pred_answers, self.ref_answer):
                            question_id = ref['question_id']
                            if len(ref['answers']) > 0:
                                pred_dict[question_id] = normalize(pred['answers'])
                                ref_dict[question_id] = normalize(ref['answers'])
                                # print '========compare in tree======='
                                # print pred_dict[question_id]
                                # print '----------------------'
                                # print ref_dict[question_id]
                        bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
                    else:
                        bleu_rouge = None
                    # print 'last words ++++++++++++++ '
                    # print bleu_rouge
                    v = input_v = bleu_rouge['Rouge-L'] * self.m_value['Rouge-L'] \
                                  + bleu_rouge['Bleu-4'] * self.m_value['Bleu-4'] \
                                  + bleu_rouge['Bleu-1'] * self.m_value['Bleu-1'] \
                                  + bleu_rouge['Bleu-3'] * self.m_value['Bleu-3'] \
                                  + bleu_rouge['Bleu-2'] * self.m_value['Bleu-2']
                else:
                    v = self.carpe_diem.value_function(tmp_node.data.sen)[0][0]
                tmp_node.data.value = v

            # if tmp_node.data.sen[-1] == str(self.l_passages - 1):
            #     pred_answer = tmp_node.data.sen
            #     listSelectedSet_sens = []
            #     listSelectedSet = map(eval, pred_answer)
            #     # print listSelectedSet
            #     for idx in listSelectedSet:
            #         listSelectedSet_sens.append(self.p_sen_list[idx])
            #         # print 'pred_answer '
            #     pred_answer_str = ''
            #     for sen in listSelectedSet_sens:
            #         str123_list = self.carpe_diem.vocab.recover_from_ids(sen, 0)
            #         for s in str123_list:
            #             pred_answer_str += s
            #
            #     pred_answers = []
            #
            #     pred_answers.append({'question_id': [self.q_id],
            #                                  'question_type': [],
            #                                  'answers': [''.join(pred_answer_str)],
            #                                  'entity_answers': [[]],
            #                                  'yesno_answers': []})
            #     if len(self.ref_answer) > 0:
            #         pred_dict, ref_dict = {}, {}
            #         for pred, ref in zip(pred_answers, self.ref_answer):
            #             question_id = ref['question_id']
            #             if len(ref['answers']) > 0:
            #                     pred_dict[question_id] = normalize(pred['answers'])
            #                     ref_dict[question_id] = normalize(ref['answers'])
            #         bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
            #     else:
            #         bleu_rouge = None
            #     v = bleu_rouge['Rouge-L'] * self.m_value['Rouge-L'] \
            #                           + bleu_rouge['Bleu-4'] * self.m_value['Bleu-4'] \
            #                           + bleu_rouge['Bleu-1'] * self.m_value['Bleu-1'] \
            #                           + bleu_rouge['Bleu-3'] * self.m_value['Bleu-3'] \
            #                           + bleu_rouge['Bleu-2'] * self.m_value['Bleu-2']
            # else:
            #     v = self.carpe_diem.value_function(tmp_node.data.sen)[0][0]


            self.update(search_list, v)
            self.count += 1

            if tmp_node.is_leaf() and (self.tree.depth(tmp_node) < self.max_depth) and tmp_node.data.sen[-1] != str(self.l_passages-1):
                self.expand(tmp_node)

            # if tm %10 == 0:
            #     print ('==================== search 10  time = %3.2f s ====================' % (time.time() - batch_start_time))
            ###########
            '''
            if time % 100 == 0:
                tmp_policy = self.get_ppolicy(start_node_id)
                print tmp_policy.values()
                print sum(tmp_policy.values())
                print time
            '''
            #print tmp_node.data.word
            #print '------finish search '
        #print '===== finish all search ======'

    def search_eval(self, start_node_id):
        #print '----tree search'
        tmp_node = self.tree.get_node(start_node_id)
        #print tmp_node.data.num
        has_visit_num = tmp_node.data.num - 1
        self.count = has_visit_num

        if int(self.max_search_time - has_visit_num) > 0:
            start_node_search_time = int(self.max_search_time - has_visit_num)
        else:
            start_node_search_time = 0

        for time in range(start_node_search_time):
            #print ('search time',time)
            search_list = [start_node_id]
            tmp_node = self.tree.get_node(start_node_id)
            #print 'search time :'+ str(time)

            while not tmp_node.is_leaf():
                max_score = float("-inf")
                max_id = -1
                for child_id in tmp_node.fpointer:
                    child_node = self.tree.get_node(child_id)
                    score = self.beta * child_node.data.p * (
                    (tmp_node.data.num) ** 0.5 / (1 + child_node.data.num))

                    #print 'child_node.data.Q: '
                    #print child_node.data.Q
                    score += child_node.data.Q

                    #print 'score: '
                    #print score

                    #print '**************'

                    if score > max_score:
                        max_id = child_id
                        max_score = score
                search_list.append(max_id)
                tmp_node = self.tree.get_node(max_id)

            #if tmp_node.data.word[-1] == str(self.l_passages-1):
            v = self.carpe_diem.value_function(tmp_node.data.sen)[0][0]
                #print 'v: '
                #print v

            self.update(search_list, v)
            self.count += 1

            if tmp_node.is_leaf() and (self.tree.depth(tmp_node) < self.max_depth) and tmp_node.data.sen[-1] != str(self.l_passages-1):
                self.expand(tmp_node)

            ###########
            '''
            if time % 100 == 0:
                tmp_policy = self.get_ppolicy(start_node_id)
                print tmp_policy.values()
                print sum(tmp_policy.values())
                print time
            '''
            #print tmp_node.data.word
            #print '------finish search '
        #print '===== finish all search ======'

    def take_action(self, start_node_id):
        #print '----take action: '
        tmp_node = self.tree.get_node(start_node_id)
        max_time = -1
        prob = {}
        for child_id in tmp_node.fpointer:
            child_node = self.tree.get_node(child_id)
            prob[child_node.data.sen[-1]] = child_node.data.num / self.count

            if child_node.data.num > max_time:
                #print child_node.data.num
                #print max_time
                #print 'child_node.data.num > max_time'
                max_time = child_node.data.num
                select_word = child_node.data.sen[-1]
                select_word_node_id = child_node.identifier
            #else:
                #print 'not > max time '

        #print select_word
        #print select_word_node_id
        #print '-----take action end'
        return prob, select_word, select_word_node_id

    def get_ppolicy(self, start_node_id):
        tmp_node = self.tree.get_node(start_node_id)
        max_time = -1
        prob = {}
        for child_id in tmp_node.fpointer:
            child_node = self.tree.get_node(child_id)
            if self.count == 0:
                prob[child_node.data.sen[-1]] = 0.0
            else:
                prob[child_node.data.sen[-1]] = child_node.data.num / self.count
        return prob


