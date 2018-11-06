# -*- coding:utf8 -*-
"""
Data structure: json
line: {
passages: [{
url: 'str'
segmented_passage_text: 'list'
passage_text: 'str'
is_selected: '0/1'
},{},{}...
]
query_type: 'description/'
segmented_query: 'list'
query_id: '199699'
answers: 'str'
segmented_answers: 'list'
query: 'str'
}
"""


import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import logging
import json
#import thulac
import datetime


start_time = ""
end_time = ""

train_search_file = "search.train.json"
train_zhidao_file = "zhidao.train.json"
dev_search_file = "search.dev.json"
dev_zhidao_file = "zhidao.dev.json"
test_search_file = "search.test.json"
test_zhidao_file = "zhidao.test.json"

train_file = "trainset/"
dev_file = "devset/"
test_file = "testset/"


'''
python -u read.py ./dataset_v1.1/dev_part100.json ./result/
'''

def display():
    logger = logging.getLogger("data_process")
    target_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    logger.info('Target data file is ' + target_file_path)
    logger.info('Outout data file is ' + output_file_path)

    for dir_path in [output_file_path]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    with open(target_file_path) as target_file:
        logger.info('Target file: {} questions.'.format(str(target_file)))

        #punc_unicode = punc.decode("utf-8")
        #print punc_unicode

        num = 0
        for line in target_file:
            print('---------------------------------')
            num += 1
            sample = json.loads(line)
            #print (sampl
            for key,value in sample.items():
                print(key)
                print(value)
        print('num of line ++++++++++++ ', num)

def run():
    start_time = str(datetime.datetime.now())
    logger = logging.getLogger("data_process")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(str(sys.argv)))

    display()
    end_time = str(datetime.datetime.now())
    logger.info('Start at ' + start_time)
    logger.info('End at '+ end_time)


if __name__ == '__main__':
    run()