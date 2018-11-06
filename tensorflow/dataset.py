# -*- coding:utf8 -*-
"""
This module implements data process strategies.
"""

import os
import json
import logging
import numpy as np
from textblob import TextBlob
from collections import Counter
import sys

class MRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, max_p_num, max_p_len, max_q_len, max_s_len,
                 train_files=[], dev_files=[], test_files=[], vocab = None):
        self.logger = logging.getLogger("MRCDataset")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.max_s_len = max_s_len
        self.vocab = vocab

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file, train=True)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file, train = True)
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def list2string(self, lis):
        string = ''
        for l in lis:
            string += l
        return string

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        with open(data_path) as fin:
            data_set = []
            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                for doc in sample['passages']:
                    doc['passage_tokens'] = []
                    for sens in doc['segmented_passage_text']:
                        doc['passage_tokens'] += sens
                    # print(doc['passage_tokens'])
                data_set.append(sample)
        return data_set

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['segmented_query']:
                    yield token
                for passage in sample['passages']:
                    for token in passage['passage_tokens']:
                        yield token


    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['query_token_ids'] = vocab.convert_to_ids(sample['segmented_query'])
                sample['segmented_answers_ids'] = []
                for ans in sample['segmented_answers']:
                    sample['segmented_answers_ids'] += vocab.convert_to_ids(ans)
                # print(sample['query_token_ids'])
                for passage in sample['passages']:
                    passage['sentence_token_ids_list'] = []
                    for sen in passage['segmented_passage_text']:
                        sentence_token_ids = vocab.convert_to_ids(sen)
                        passage['sentence_token_ids_list'].append(sentence_token_ids)
                    # print(passage['sentence_token_ids_list'])

    def _convert_new_data(self, data):
        """Convert old data to new formation data structure.

        :param data: old data formation.
        :return: new data formation.
        """
        new_data = []
        for sample in data:
            new_sample = {}
            for key, value in sample.items():
                if key != 'passages':
                    new_sample[key] = value
            for passage in sample['passages']:
                for key, value in passage.items():
                    new_sample[key] = value
            new_data.append(new_sample)
        return new_data

    def _one_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'query_ids': [],                # [0,0,0,1,1,1,...]   1 * n
                      'query_token_ids': [],          # [[id list],[id list],[id list],...] m * n
                      'query_length': [],             # [length,length,length,length,....] 1 * n
                      'padded_p_len': [],             # []
                      'padded_s_len': [],
                      'padded_q_len': [],
                      'passage_token_ids': [],        # [[[sen],[sen],...],[[sen],[sen],...],...]   x * y * n
                      'passage_sen_list_length': [],  # [[sen_len,sen_len,...],[sen_len],[sen_len],...],...] k * n
                      'segmented_answers': [],
                      'answers': []}

        for idx, sample in enumerate(batch_data['raw_data']):
            # query
            # print(sample)
            batch_data['query_ids'].append(sample['query_id'])
            batch_data['query_token_ids'].append(sample['query_token_ids'])
            batch_data['query_length'].append(min(len(sample['query_token_ids']), self.max_q_len))
            # passage
            batch_data['passage_token_ids'].append(sample['sentence_token_ids_list'])
            batch_data['passage_sen_list_length'].append(
                [min(self.max_s_len, len(sen)) for sen in sample['sentence_token_ids_list']])
            # answer
            batch_data['segmented_answers'].append(sample['segmented_answers_ids'])
            batch_data['answers'].append(sample['answers'])

        batch_data, padded_p_len, padded_q_len, padded_s_len = self._dynamic_padding(batch_data, pad_id)
        batch_data['padded_p_len'].append(padded_p_len)
        batch_data['padded_s_len'].append(padded_s_len)
        batch_data['padded_q_len'].append(padded_q_len)

        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        # print 'dynamic _padding...'
        # padding query
        pad_q_len = min(self.max_q_len, max(batch_data['query_length']))
        batch_data['query_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['query_token_ids']]
        batch_data['segmented_answers'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                         for ids in batch_data['segmented_answers']]
        # padding passage
        pad_p_len = -1
        pad_s_len = -1
        for passage_sen_len in batch_data['passage_sen_list_length']:
            # sentence padding
            pad_p_len = max(pad_p_len, len(passage_sen_len))
            for sen_len in passage_sen_len:
                pad_s_len = max(pad_s_len, sen_len)
        pad_p_len = min(self.max_p_len, pad_p_len)
        pad_s_len = min(self.max_s_len, pad_s_len)
        print('pad_s_len', pad_s_len)
        print('pad_p_len', pad_p_len)
        new_passage_token_ids = []
        for passage in batch_data['passage_token_ids']:
            passage = [(ids + [pad_id] * (pad_s_len - len(ids)))[: pad_s_len]
                                               for ids in passage]
            new_passage_token_ids.append(passage)
        batch_data['passage_token_ids'] = new_passage_token_ids
        padding_sen = [pad_id] * pad_s_len

        batch_data['passage_token_ids'] = [(ids + [padding_sen] * (pad_p_len - len(ids)))[: pad_p_len]
                                            for ids in batch_data['passage_token_ids']]

        batch_data['passage_sen_list_length'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_sen_list_length']]

        return batch_data, pad_p_len, pad_q_len, pad_s_len

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        new_data = self._convert_new_data(data)
        data_size = len(new_data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(new_data, batch_indices, pad_id)

    def _dynamic_padding_new(self, passage_token_ids, passage_sen_length, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        #print 'dynamic _padding...'
        #print 'pad_id' + str(pad_id)
        for leng in passage_sen_length:
            if(leng > self.max_p_len ):
                leng = self.max_p_len
        pad_p_len = min(self.max_p_len, max(passage_sen_length))
            #print (ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
        passage_token_ids_padded = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in passage_token_ids]
        return passage_token_ids_padded, pad_p_len, passage_sen_length