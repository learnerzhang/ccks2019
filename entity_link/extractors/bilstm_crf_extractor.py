#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-24 14:28
# @Author  : zhangzhen
# @Site    : 
# @File    : bilstm_crf_extractor.py.py
# @Software: PyCharm
import os
import sys
import time
import utils
import getopt
import codecs
import pprint
import pickle
import logging
import warnings
import numpy as np
import tensorflow as tf

from data.ner_data_utils import read_corpus
from entity_link.extractors.bilstm_crf_model import BiLstmCRF
from entity_link.extractors.utils import read_dictionary, random_embedding, random_split_data
from typing import Any, Dict, Optional, Text, List

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


class BiLSTMCrfExtractor(object):

    def __init__(self, component_config: Optional[Dict[Text, Text]] = None,
                 **kwargs: Any) -> None:
        logging.info("bi-lstm cfg: {}".format(component_config))
        self.model_path = component_config['model_path'] + os.sep + str(int(time.time()))

        self.eval_perl = component_config['eval_perl']
        self.tag2label = component_config['tag2label']
        self.vocab2id = read_dictionary(component_config['word2id'])

        self.embedding_dim = component_config.get('embedding_dim', 100)
        self.batch_size = component_config.get('batch_size', 64)
        self.epoch = component_config.get('epoch', 10)
        self.hidden_dim = component_config.get('hidden_dim', 60)

        _model_path = self.model_path + os.sep + 'checkpoints'
        _result_path = self.model_path + os.sep + 'results'
        _summary_path = self.model_path + os.sep + 'summaries'

        self.kwargs = {
            'model_path': kwargs.get('model_path') if kwargs.get('model_path', None) else _model_path,
            'result_path': kwargs.get('result_path') if kwargs.get('result_path', None) else _result_path,
            'summary_path': kwargs.get('result_path') if kwargs.get('result_path', None) else _summary_path,
        }

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

        self.config = config
        self.model = self.build_model()
        self.model.build_graph()

    def build_model(self):
        embeddings = random_embedding(len(self.vocab2id), self.embedding_dim)
        model = BiLstmCRF(self.config, embeddings, self.tag2label, self.vocab2id, self.eval_perl,
                          batch_size=self.batch_size,
                          hidden_dim=self.hidden_dim,
                          epoch=self.epoch,
                          **self.kwargs)
        return model

    def train(self, train, dev) -> None:
        logging.info("train data set: {}, dev data set: {}".format(len(train), len(dev)))
        self.model.train(train=train, dev=dev)

    def process(self, text, entities: List) -> None:

        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            saver.restore(sess, self.kwargs['model_path'])
            tokens = [ch for ch in text]
            demo_data = [(tokens, ['O'] * len(tokens))]
            tags = self.model.run_one_case(sess, demo_data)
            entities.extend(self._convert_simple_tagging_to_entity_result(tags, tokens))

    def _convert_simple_tagging_to_entity_result(self, tags: List, tokens: List, ignores=True):

        entities = []
        flag = True
        offset = -1
        tmp_ent = []
        for i, (token, tag) in enumerate(zip(tokens, tags)):
            if tag != 'O' and tag != 0:
                if flag:
                    if not ignores and str(tag).startswith('I'):
                        continue
                    offset = i
                    flag = False
                tmp_ent.append(token)
            else:
                if tmp_ent:
                    entities.append({
                        'value': "".join(tmp_ent),
                        'start': offset,
                        'end': offset + len(tmp_ent),
                        'entity': 'Span'
                    })
                    flag = True
                    tmp_ent = []
                    offset = -1

        if tmp_ent:
            entities.append({
                'value': "".join(tmp_ent),
                'start': offset,
                'end': offset + len(tmp_ent),
                'entity': 'Span'
            })
        return entities

    def persist(self,
                file_name: Text,
                model_dir: Text) -> Optional[Dict[Text, Any]]:

        if self.kwargs:
            file_name = file_name + ".json"
            entity_lstm_file = os.path.join(model_dir, file_name)
            utils.write_json_to_file(entity_lstm_file, self.kwargs, separators=(',', ': '))
            return {"file": file_name}
        else:
            return {"file": None}

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             ) -> 'EntitySynonymMapper':

        file_name = meta.get("file")
        if not file_name:
            kwargs = {}
            return cls(meta, **kwargs)
        else:
            entity_lstm_file = os.path.join(model_dir, file_name)
            if os.path.isfile(entity_lstm_file):
                kwargs = utils.read_json_file(entity_lstm_file)
                model_path = kwargs.get('model_path')
                kwargs['model_path'] = tf.train.latest_checkpoint(model_path)
            else:
                kwargs = {}
                warnings.warn("Failed to load lstm file from '{}'"
                              "".format(entity_lstm_file))
        return cls(meta, **kwargs)


def test(model_path=""):
    text = """2009年，张首晟被清华大学特聘为教授。张首晟是2013年中国科学院外籍院士。=是/杨振宁/的弟子。2017年7月21日，美国加州大学洛杉矶分校University of California, Los Angeles(UCLA)华裔科学家王康隆、斯坦福大学华裔科学家张首晟、上海科技大学教授寇煦丰等团队合作在《科学》杂志上发表了一项重大发现：在整个物理学界历经80年的，。、；‘’【】、{}！@#￥`%……&（）——=+探索之后，他们终于,./<>?;':"[]{}|`!@#$%^&()_=发现了手性马约拉纳费米子的存在。张首晟–将其命名//为“天使粒子”"""

    component_config = {
        "model_path": tf.train.latest_checkpoint(model_path + os.sep + "checkpoints"),
        "result_path": model_path + os.sep + "results",
        "summary_path": model_path + os.sep + "summaries",
        "eval_perl": "data/ccks/conlleval_rev.pl",
        "tag2label": {'O': 0, 'B-Span': 1, 'I-Span': 2},
        "word2id": "data/ccks/word2id.pk",
    }
    entities = []
    lstm = BiLSTMCrfExtractor(component_config)
    lstm.process(text, entities)
    pprint.pprint(entities)


def train(model_path="models/bilstm_ner/1561100499"):
    component_config = {
        "model_path": model_path + os.sep + "checkpoints",
        "result_path": model_path + os.sep + "results",
        "summary_path": model_path + os.sep + "summaries",
        "eval_perl": "data/ccks/conlleval_rev.pl",
        "tag2label": {'O': 0, 'B-Span': 1, 'I-Span': 2},
        "word2id": "data/ccks/word2id.pk",
    }

    model = BiLSTMCrfExtractor(component_config)
    data = read_corpus(corpus_path='data/ner')
    train_data, dev_data = random_split_data(data)
    model.train(train=train_data, dev=dev_data)


if __name__ == '__main__':
    import pprint

    args = sys.argv[1:]

    path = 'entity_link.extractors.bilstm_crf_extractor'
    tips = 'python {} {}  --mode <train/test>'.format('-m', path)
    if len(args) == 0:
        print(tips)
        sys.exit()

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hm:", ["mode="])
    except getopt.GetoptError:
        print(tips)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(tips)
            sys.exit()
        elif opt in ("-m", "--mode"):
            mode = arg

    if mode == 'train':
        print("start train")
        train()
    elif mode == 'test':
        print("start test")
        test()
    else:
        print(tips)
