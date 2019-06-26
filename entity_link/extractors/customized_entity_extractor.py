#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-25 14:24
# @Author  : zhangzhen
# @Site    : 
# @File    : customized_entity_extractor.py.py
# @Software: PyCharm
import json
import logging
import copy
import re
import os
import time
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


class CustomizedEntityExtractor(object):

    def __init__(self, ):
        self.root_path = '/Users/zhangzhen/data/ccks2019_el.v1'
        self.ahocorasick = self.load_ahocorasick(ahocorasick_path='data/ccks/ac.pkl')

    def process(self, text, entities):
        for item in self.ahocorasick.iter(text):  # synonym match
            end, val = item
            offset = end - len(val) + 1 if end > 0 else end
            # [(span, offset),...]
            entities.append((val, offset))
        return entities

    def load_ahocorasick(self, ahocorasick_path):
        """ 加载数据库中自定义实体中的字典数据
        Returns:  ahocorasick ac自动机对象
        """
        import ahocorasick
        import pickle

        start = time.time()
        if os.path.exists(ahocorasick_path):
            logging.info("[*] Load from pkl {}".format(ahocorasick_path))
            with open(ahocorasick_path, 'rb') as f:
                ac = pickle.load(f)
        else:
            import codecs
            logging.info("[*] Load ahocorasick from {}".format(self.root_path))
            ac = ahocorasick.Automaton()
            with codecs.open(self.root_path + os.sep + 'train.json', encoding='utf-8') as f:
                for l in tqdm(f):
                    try:
                        _ = json.loads(l, encoding='utf-8')
                        for x in _['mention_data']:
                            if len(x['mention']) <= 2 and x['kb_id'] != 'NIL':
                                ac.add_word(x['mention'], x['mention'])
                    except ValueError:
                        print("乱码字符串: ", l)
                ac.make_automaton()
            with open(ahocorasick_path, 'wb') as f:
                pickle.dump(ac, f)

        logging.info("[*] Load ahocorasick success. cost {}s".format(time.time() - start))
        return ac


if __name__ == '__main__':
    import pprint

    text = """2009年，张首晟被清华大学特聘为教授。张首晟是2013年中国科学院外籍院士。=是/杨振宁/的弟子。
        2017年7月21日，美国加州大学洛杉矶分校University of California, Los Angeles(UCLA)华裔科学家王康隆、斯坦福大学华裔科学家张首晟、上海科技大学教授寇煦丰等团队合作在《科学》杂志上发表了一项重大发现：在整个物理学界历经80年的，。、；‘’【】、{}！@#￥`%……&（）——=+探索之后，他们终于,./<>?;':"[]{}|`!@#$%^&()_=发现了手性马约拉纳费米子的存在。
        张首晟–将其命名//为“天使粒子”"""
    entities = []
    cee = CustomizedEntityExtractor()
    cee.process(text, entities)
    pprint.pprint(entities)
