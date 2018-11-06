# -*- coding:utf8 -*-
"""
This module implements the reading comprehension models based on:
Reinforcement Learning and Monte-Carlo Tree Search
Note that we use Pointer Network for the decoding stage of both models.

"""

import os
import time
import logging
from utils import compute_bleu_rouge
from utils import normalize

from search import SearchTree

from tfgraph import TFGraph


class MCSTmodel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, vocab, args):

        # logging
        self.args = args
        self.logger = logging.getLogger("brc")

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len

        self.Bleu4 = args.Bleu4
        self.Bleu3 = args.Bleu3
        self.Bleu2 = args.Bleu2
        self.Bleu1 = args.Bleu1
        self.RougeL = args.RougeL

        self.m_value = {'Bleu-4': self.Bleu4,'Bleu-3': self.Bleu3,'Bleu-2': self.Bleu2,'Bleu-1': self.Bleu1,'Rouge-L':self.RougeL}
        #test paras
        self.search_time = args.search_time
        self.beta = args.beta
        #time
        self.init_times = 0.0
        self.search_times = 0.0
        self.act_times = 0.0
        self.grad_times = 0.0


        # the vocab
        self.vocab = vocab
        self.tfg = TFGraph('train', vocab, args)

    def _analysis(self, step , train_batches, dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_loss = 0
        num_loss = 0
        total_recall = [0.0,0.0,0.0,0.0]
        num_recall =0
        batch_start_time = 0
        batch_start_time = time.time()
        pred_answers, ref_answers = [], []
        fake_answers = []
        for fbitx, batch in enumerate(train_batches, 1):
            step += 1
            if fbitx % 1000 == 0:
                print '------ Batch Question: ' + str(fbitx)

            trees = []
            batch_tree_set = []

            batch_size = len(batch['question_ids'])
            #print ('batch_size)', batch_size)
            for bitx in range(batch_size):
                tree = {'question_id': batch['question_ids'][bitx],
                        'question_token_ids': batch['question_token_ids'][bitx],
                        'q_length': batch['question_length'][bitx],

                        'passage_token_ids_list': batch['passage_token_ids_list'][bitx],

                        'passage_title_token_ids_list': batch['passage_title_token_ids_list'][bitx],
                        'passage_title_length_list': batch['passage_title_length_list'][bitx],

                        'passage_sentence_token_ids_list': batch['passage_sentence_token_ids_list'][bitx],
                        'passage_sen_length': batch['passage_sen_length_list'][bitx],


                        #'p_length': batch['passage_length'][bitx],
                        'passage_is_selected_list': batch['passage_is_selected_list'][bitx],

                        'question_type': batch['question_types'][bitx],

                        'ref_answers': batch['ref_answers'][bitx],
                        'fake_answers': batch['fake_answers'][bitx],
                        'segmented_answers': batch['segmented_answers'][bitx]

                        }
                ref_answers.append({'question_id': tree['question_id'],
                                    'question_type': tree['question_type'],
                                    'answers': tree['ref_answers']})
                trees.append(tree)
                #print batch
                batch_tree = SearchTree(self.tfg, tree, self.max_a_len, self.search_time, self.beta, self.m_value, dropout_keep_prob)
                batch_tree_set.append(batch_tree)


            # for every data in batch do training process
            for idx, batch_tree in enumerate(batch_tree_set,1):
                pred_answer, fake_answer, recall = batch_tree.train_analysis(step)
                pred_answers.append(pred_answer)
                fake_answers.append(fake_answer)
                total_recall[0] += recall[0]
                total_recall[1] += recall[1]
                total_recall[2] += recall[2]
                total_recall[3] += recall[3]
                num_recall += 1
        print('ave select recall', total_recall[0] / num_recall)
        print('ave select f1', total_recall[1] / num_recall)
        print('ave all recall', total_recall[2] / num_recall)
        print('ave all f1', total_recall[3] / num_recall)
        ii = 0
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                ii += 1
                question_id = ref['question_id']
                #print('type', question_id)
                if len(ref['answers']) > 0:

                    ref_dict[question_id] = normalize(ref['answers'])
                    pred_dict[question_id] = normalize(pred['answers'])


            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        value_with_mcts = bleu_rouge
        print ('pre_scor',value_with_mcts)


        #return 1.0 * total_loss / num_loss, step
        return 0, step

    def _train_epoch(self, step , train_batches, dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_loss = 0
        num_loss = 0
        total_acc = 0
        num =0
        batch_start_time = 0
        batch_start_time = time.time()
        pred_answers, ref_answers = [], []
        fake_answers = []
        for fbitx, batch in enumerate(train_batches, 1):
            step += 1
            if fbitx % 1000 == 0:
                print '------ Batch Question: ' + str(fbitx)

            trees = []
            batch_tree_set = []
            step += 1
            batch_size = len(batch['question_ids'])
            #print ('batch_size)', batch_size)
            for bitx in range(batch_size):
                tree = {'question_id': batch['question_ids'][bitx],
                        'question_token_ids': batch['question_token_ids'][bitx],
                        'q_length': batch['question_length'][bitx],

                        'passage_token_ids_list': batch['passage_token_ids_list'][bitx],

                        'passage_title_token_ids_list': batch['passage_title_token_ids_list'][bitx],
                        'passage_title_length_list': batch['passage_title_length_list'][bitx],

                        'passage_sentence_token_ids_list': batch['passage_sentence_token_ids_list'][bitx],
                        'passage_sen_length': batch['passage_sen_length_list'][bitx],


                        #'p_length': batch['passage_length'][bitx],
                        'passage_is_selected_list': batch['passage_is_selected_list'][bitx],

                        'question_type': batch['question_types'][bitx],

                        'ref_answers': batch['ref_answers'][bitx],
                        'fake_answers': batch['fake_answers'][bitx],
                        'segmented_answers': batch['segmented_answers'][bitx]

                        }
                ref_answers.append({'question_id': tree['question_id'],
                                    'question_type': tree['question_type'],
                                    'answers': tree['ref_answers']})
                trees.append(tree)
                #print batch
                batch_tree = SearchTree(self.tfg, tree, self.max_a_len, self.search_time, self.beta, self.m_value, dropout_keep_prob)
                batch_tree_set.append(batch_tree)


            # for every data in batch do training process
            for idx, batch_tree in enumerate(batch_tree_set,1):
                loss, acc = batch_tree.train_encode_and_select_passage(step)
                total_loss += loss
                total_acc += acc
                num += 1
        ave_acc = total_acc / num
        ave_loss = total_loss / num
        print('ave acc', total_acc / num)
        print('ave loss', total_loss / num)

        #return 1.0 * total_loss / num_loss, step
        return ave_loss, ave_acc, step

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=1.0, evaluate=True):
        """
        Train the model with data
        Args:
            data: the class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        # print 'pad_id is '
        # print pad_id
        max_bleu_4 = 0
        train_step = 0
        dev_step = 0
        #pmct = PSCHTree(self.args, self.vocab)
        start_all_time = time.time()

        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            epoch_start_time = time.time()
            train_batches = data.gen_batches('train', batch_size, pad_id, shuffle=True)
            # mctree = MCtree(train_batches)
            # mctree.search()

            #result = self._train_epoch_new(pmct, train_batches, batch_size, dropout_keep_prob)
            train_ave_loss, train_ave_acc, train_step = self._train_epoch(train_step, train_batches, dropout_keep_prob)
            self.tfg.draw_train(train_ave_loss, train_ave_acc, train_step)
            epoch_end_time = time.time()
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_ave_loss))
            self.logger.info('Average train acc for epoch {} is {}'.format(epoch, train_ave_acc))
            #self.save(save_dir, save_prefix + '_' + str(epoch))
            self.logger.info(
                'Train time for epoch {} is {} min'.format(epoch, str((epoch_end_time - epoch_start_time) / 60)))
            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_loss, total_loss, num_loss = 0, 0, 0
                    eval_batches = data.gen_batches('dev', batch_size, pad_id, shuffle=False)
                    dev_ave_loss, dev_ave_acc, dev_step= self.evaluate(dev_step, eval_batches,dropout_keep_prob)
                    self.tfg.draw_test(dev_ave_loss, dev_ave_acc, dev_step)
                    self.logger.info('Average dev loss for epoch {} is {}'.format(epoch, dev_ave_loss))
                    self.logger.info('Average dev acc for epoch {} is {}'.format(epoch, dev_ave_acc))
            #
            #         if bleu_rouge['Bleu-4'] > max_bleu_4:
            #             pmct.save(save_dir, save_prefix)
            #             max_bleu_4 = bleu_rouge['Bleu-4']
            #     else:
            #         self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            # else:
            #     pmct.save(save_dir, save_prefix + '_' + str(epoch))
        self.logger.info(
            'All Train time is {} min'.format(str((time.time() - start_all_time) / 60)))



    def evaluate(self, step, eval_batches, dropout_keep_prob,result_dir=None, result_prefix=None, save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        total_loss = 0
        num_loss = 0
        total_acc = 0
        num = 0
        batch_start_time = 0
        batch_start_time = time.time()
        pred_answers, ref_answers = [], []
        fake_answers = []
        for fbitx, batch in enumerate(eval_batches, 1):
            step += 1
            if fbitx % 1000 == 0:
                print '------ Batch Question: ' + str(fbitx)

            trees = []
            batch_tree_set = []

            batch_size = len(batch['question_ids'])
            # print ('batch_size)', batch_size)
            for bitx in range(batch_size):
                tree = {'question_id': batch['question_ids'][bitx],
                        'question_token_ids': batch['question_token_ids'][bitx],
                        'q_length': batch['question_length'][bitx],

                        'passage_token_ids_list': batch['passage_token_ids_list'][bitx],

                        'passage_title_token_ids_list': batch['passage_title_token_ids_list'][bitx],
                        'passage_title_length_list': batch['passage_title_length_list'][bitx],

                        'passage_sentence_token_ids_list': batch['passage_sentence_token_ids_list'][bitx],
                        'passage_sen_length': batch['passage_sen_length_list'][bitx],

                        # 'p_length': batch['passage_length'][bitx],
                        'passage_is_selected_list': batch['passage_is_selected_list'][bitx],

                        'question_type': batch['question_types'][bitx],

                        'ref_answers': batch['ref_answers'][bitx],
                        'fake_answers': batch['fake_answers'][bitx],
                        'segmented_answers': batch['segmented_answers'][bitx]
                        }
                ref_answers.append({'question_id': tree['question_id'],
                                    'question_type': tree['question_type'],
                                    'answers': tree['ref_answers']})
                trees.append(tree)
                # print batch
                batch_tree = SearchTree(self.tfg, tree, self.max_a_len, self.search_time, self.beta, self.m_value,
                                        dropout_keep_prob)
                batch_tree_set.append(batch_tree)

            # for every data in batch do training process
            for idx, batch_tree in enumerate(batch_tree_set, 1):
                loss, acc = batch_tree.dev_encode_and_select_passage(step)
                total_loss += loss
                total_acc += acc
                num += 1
        print('ave acc', total_acc / num)
        print('ave loss', total_loss / num)
        ave_acc = total_acc / num
        ave_loss = total_loss / num

        # return 1.0 * total_loss / num_loss, step
        return ave_loss, ave_acc, step


