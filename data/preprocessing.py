# -*- coding:utf8 -*-
"""
1. Segmentation
    python preprocessing.py --seg --seg_dir ./seg/ --target_file ./dataset_v1.1/2_train.json
    nohup python -u preprocessing.py --seg --seg_dir ./seg/ --target_file ./dataset_v1.1/100_train.json >./log/seg_100_log.txt 2>&1 &

2. glove train prepare
    python data_processing.py --seg_trainfile --seg_dir ./demo/trainset --target_files ./demo/trainset
    
Data 2018.1.10
Author Glee 
"""

import sys
import datetime
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import logging
from textblob import TextBlob
import json


def _parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser('Pre_procession of raw data set')
    parser.add_argument('--seg', default=None,
                        help='if use segment model',
                        action='store_true')
    parser.add_argument('--div', choices=['DEYN', 'FO'], default=None,
                                help="if divide data set into sub set, " +
                                     "choose DEYN: divide data use Answer type: Description / Entity / Yes_No or"+
                                     "choose FO: divide data use Question type: Fact / Opinion")

    model_settings = parser.add_argument_group('parameter settings')
    model_settings.add_argument('--size', type=int, default=100,
                                help='size of the line')
    model_settings.add_argument('--iftrain', type=bool, default=True,
                                help='If the file is train dataset')
    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--target_file', nargs='+',
                               default=['./draw/trainset/search.train.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--seg_dir', default=['./seg/'],
                               help='the dir to write segment results ')
    path_settings.add_argument('--result_dir', default='../data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


def segment(target_files, seg_dir, iftrain):
    logger = logging.getLogger("Data")
    if target_files:
        logger.info('Target file: {}.'.format(str(target_files[0])))
        f_name = target_files[0].split('/')
        f_name = f_name[len(f_name) - 1]
        out_file = open(seg_dir + f_name, 'w')
        out_file.truncate()
        logger.info('Write to : {} file.'.format(str(out_file)))
        with open(target_files[0]) as fin:
            n = 1
            for line in fin:
                logger.info('Processing number {} line.'.format(str(n)))
                n += 1
                sample = json.loads(line)
                # query
                # print(sample['query'])
                zen = TextBlob(sample['query'])
                # logger.info('Sentence is {}'.format(str(zen.sentences)))
                # logger.info('Words is {}'.format(str(zen.words)))
                sample['segmented_query'] = zen.words
                # passages
                for d_idx, doc in enumerate(sample['passages']):
                    # logger.info('Passage {}:'.format(d_idx))
                    doc['segmented_passage_text'] = []
                    # print(doc)
                    zen = TextBlob(doc['passage_text'])
                    # logger.info('Sentence is {}'.format(str(zen.sentences)))
                    for sens in zen.sentences:
                        # logger.info('Words is {}'.format(str(sens.words)))
                        doc['segmented_passage_text'].append(sens.words)
                if iftrain:
                    sample['segmented_answers'] = []
                    # print(sample['answers'])
                    for a_idx, ans in enumerate(sample['answers']):
                        zen = TextBlob(ans)
                        # logger.info('Sentence is {}'.format(str(zen.sentences)))
                        # logger.info('Words is {}'.format(str(zen.words)))
                        sample['segmented_answers'].append(str(zen.words))
                out_file.write(str(json.dumps(sample, ensure_ascii=False)) + '\n')
                out_file.flush()
            out_file.close()
        logger.info('Segment Success')



def run():
    args = _parse_args()
    start_time = str(datetime.datetime.now())
    logger = logging.getLogger("Data")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))
    if args.seg:
        segment(args.target_file, args.seg_dir, args.iftrain)
    end_time = str(datetime.datetime.now())
    logger.info('Start at ' + start_time)
    logger.info('End at ' + end_time)

if __name__ == '__main__':
    run()
