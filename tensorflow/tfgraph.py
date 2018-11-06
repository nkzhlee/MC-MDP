# -*- coding:utf8 -*-
import os
import time
import logging
import json
from mctree import search_tree
import numpy as np
import tensorflow as tf
from utils import compute_bleu_rouge
from utils import normalize
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder


class TFGraph(object):
    """
    Implements the main reading comprehension model.

    python -u run.py --train --algo MCST --epochs 10 --batch_size 1 --max_p_len 10000 --hidden_size 150  --train_files ../data/demo/trainset/test10 --dev_files  ../data/demo/devset/test20 --test_files ../data/demo/test/search.test.json
    """

    def __init__(self, name, vocab, args):
        self.tf_name = name
        self.logger = logging.getLogger("brc")
        self.vocab = vocab
        self.draw_path = args.draw_path

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1

        self._build_graph()

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        start_t = time.time()
        self._setup_placeholders()
        self._initstate()
        self._embed()
        self._encode()
        self._match()
        self._fuse()
        self._action()
        self._create_train_op()
        self._draw_rfboard()
        # param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        # self.logger.info('There are {} parameters in the model'.format(param_num))
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))

    def _setup_placeholders(self):
        """
        Placeholders
         """
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.t = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.t_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.is_selected = tf.placeholder(tf.float32, [None, None])
        self.result = tf.placeholder(tf.float32, None)
        self.acc = tf.placeh
        older(tf.float32, None)
        self.dropout_keep_prob = tf.placeholder(tf.float32)


    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True
            )
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)
            self.t_emb = tf.nn.embedding_lookup(self.word_embeddings, self.t)

    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes, _ = rnn('bi-lstm', self.p_emb, self.p_length, self.hidden_size)
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, self.sep_Q = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size)
        with tf.variable_scope('title_encoding'):
            self.sep_t_encodes, _ = rnn('bi-lstm', self.t_emb, self.t_length, self.hidden_size)
        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)
            self.sep_t_encodes = tf.nn.dropout(self.sep_t_encodes, self.dropout_keep_prob)
            self.sep_Q = tf.nn.dropout(self.sep_Q, self.dropout_keep_prob)

    def _match(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """

        match_layer = AttentionFlowMatchLayer(self.hidden_size)
        self.match_p_encodes, _ = match_layer.match(self.sep_p_encodes, self.sep_q_encodes,
                                                    self.p_length, self.q_length)
        match_layer_t = AttentionFlowMatchLayer(self.hidden_size)
        self.match_t_encodes, _ = match_layer_t.match(self.sep_t_encodes, self.sep_q_encodes,
                                                    self.t_length, self.q_length)
        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)
            self.match_t_encodes = tf.nn.dropout(self.match_t_encodes, self.dropout_keep_prob)

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        with tf.variable_scope('p_fusion'):
            self.fuse_p_encodes, self.fuse_P = rnn('bi-lstm', self.match_p_encodes, self.p_length,
                                         self.hidden_size, layer_num=1)
            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)
                self.fuse_P = tf.nn.dropout(self.fuse_P, self.dropout_keep_prob)

        with tf.variable_scope('t_fusion'):
            self.fuse_t_encodes, self.fuse_T = rnn('bi-lstm', self.match_t_encodes, self.t_length,
                                                           self.hidden_size, layer_num=1)
            if self.use_dropout:
                self.fuse_t_encodes = tf.nn.dropout(self.fuse_t_encodes, self.dropout_keep_prob)
                self.fuse_T = tf.nn.dropout(self.fuse_T, self.dropout_keep_prob)


    def _initstate(self):



        self.W_l = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.hidden_size * 4], -1. / self.hidden_size,
                                               1. / self.hidden_size))
        self.b_l = tf.Variable(tf.random_uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32, seed=None, name=None))


        #self.words = tf.reshape(self.p_encodes, [-1, self.hidden_size * 2])

        # self.words_list = tf.gather(self.words, self.p_words_id) # all words in a question doc


    def _action(self):

        self.P_T = tf.concat([self.fuse_T, self.fuse_P], 1)
        #self.P_T = self.fuse_T
        self.p_l = tf.sigmoid(tf.add(tf.matmul(tf.matmul(self.sep_Q, self.W_l), tf.transpose(self.P_T)), self.b_l))
        self.p_loss = tf.add(tf.matmul(tf.matmul(self.sep_Q, self.W_l), tf.transpose(self.P_T)), self.b_l)

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        
        """
        # self.log_p = tf.log(tf.clip_by_value(self.p_l, 1e-30, 1.0))
        # self.part_a = tf.matmul(self.is_selected, self.log_p)
        # self.part_b = tf.matmul((1 - self.is_selected), (1.0 - self.log_p))
        # self.loss = tf.add(self.part_a, self.part_b)
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.is_selected, logits=self.p_loss)

        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)

        self.train_op = self.optimizer.minimize(self.loss)

    def _draw_rfboard(self):
        self.loss_summary = tf.summary.scalar('loss', tf.reduce_mean(self.result))
        self.acc_summary = tf.summary.scalar('rb', tf.reduce_mean(self.acc))
        with tf.name_scope('summary'):
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.draw_path+'/train')
            self.test_writer = tf.summary.FileWriter(self.draw_path+'/test')

    def set_feed_dict_train(self,p,q,t,p_length, q_length, t_length,is_selected, dropout_keep_prob):
        self.feed_dict = {
                          self.p: p,
                          self.q: q,
                          self.t: t,
                          self.q_length: q_length,
                          self.p_length: p_length,
                          self.t_length: t_length,
                          self.is_selected: is_selected,
                          self.dropout_keep_prob: dropout_keep_prob}
    def set_feed_dict_test(self,p,q,t,p_length, q_length, t_length):
        self.feed_dict = {
                          self.p: p,
                          self.q: q,
                          self.t: t,
                          self.q_length: q_length,
                          self.p_length: p_length,
                          self.t_length: t_length,
                         }

    def cal_loss(self):
        # a, b, c, d, e = self.sess.run([self.sep_Q, self.fuse_p_encodes, self.fuse_P, self.fuse_t_encodes, self.fuse_T], feed_dict=self.feed_dict)
        # return a, b, c, d, e
        p, loss, _ = self.sess.run([self.p_l, self.loss, self.train_op], feed_dict=self.feed_dict)
        return p, loss

    def test_loss(self):
        # a, b, c, d, e = self.sess.run([self.sep_Q, self.fuse_p_encodes, self.fuse_P, self.fuse_t_encodes, self.fuse_T], feed_dict=self.feed_dict)
        # return a, b, c, d, e
        loss = self.sess.run([self.loss], feed_dict=self.feed_dict)
        return loss

    def draw_train(self, result, acc, step):
        feed_dict = dict({self.result: result, self.acc: acc})
        summary = self.sess.run(self.merged, feed_dict = feed_dict)
        self.train_writer.add_summary(summary, step)

    def draw_test(self, result, acc, step):
        feed_dict = dict({self.result: result, self.acc: acc})
        summary = self.sess.run(self.merged, feed_dict = feed_dict)
        self.test_writer.add_summary(summary, step)

    def get_p_l(self):
        # a, b, c, d, e = self.sess.run([self.sep_Q, self.fuse_p_encodes, self.fuse_P, self.fuse_t_encodes, self.fuse_T], feed_dict=self.feed_dict)
        # return a, b, c, d, e
        p = self.sess.run([self.p_l], feed_dict=self.feed_dict)
        return p

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))

