#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-25 16:09
# @Author  : zhangzhen
# @Site    : 
# @File    : entity_extractor.py
# @Software: PyCharm
from entity_link.extractors.customized_entity_extractor import CustomizedEntityExtractor
from entity_link.extractors.duckling_http_extractor import DucklingEntityExtractor
from entity_link.extractors.utils import deduplicate_entities
import pprint

text = """2009年，张首晟被清华大学特聘为教授。张首晟是2013年中国科学院外籍院士。=是/杨振宁/的弟子。
        2017年7月21日，美国加州大学洛杉矶分校University of California, Los Angeles(UCLA)华裔科学家王康隆、斯坦福大学华裔科学家张首晟、上海科技大学教授寇煦丰等团队合作在《科学》杂志上发表了一项重大发现：在整个物理学界历经80年的，。、；‘’【】、{}！@#￥`%……&（）——=+探索之后，他们终于,./<>?;':"[]{}|`!@#$%^&()_=发现了手性马约拉纳费米子的存在。
        张首晟–将其命名//为“天使粒子”"""
entities = []

duckling = DucklingEntityExtractor()
duckling.process(text, entities)

cee = CustomizedEntityExtractor()
cee.process(text, entities)

print("去重前:", len(entities))
pprint.pprint(entities)
entities = deduplicate_entities(entities)
print("去重后:", len(entities))
pprint.pprint(entities)
