import json
import time

import logging
import os
import requests
import simplejson
from typing import Any, List, Optional, Text, Dict

from entity_link.extractors import HttpSessionContext

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


class DucklingEntityExtractor():
    http_session = HttpSessionContext()
    request = None

    def __init__(self):
        self.duckling_url = 'http://d65.mlamp.co:8001/parse'
        self.language = 'zh'
        self.dims = []

    def process(self, text, entities: List):
        # type: (Doc, Language, List[Dict[Text, Any]]) -> Dict[Text, Any]
        self.request = self.http_session.open()
        extracted = self.extract_entities(text)
        entities.extend(extracted)
        self.request.close()

    def extract_entities(self, text):
        """

        Args:
            src_text: 原始的查询语句
            tokenized_text: 加分词,空格
            language: 语言
            dims: 查询的维度
            duckling_url: 请求地址url

        Returns: []

        """
        pdata = {"text": text, "dims": self.dims, "locale": self.language}
        logging.debug('call duckling backend , request params: ' + str(pdata))

        final_output = []
        try:
            output = json.loads(self.request.post(self.duckling_url, pdata, timeout=3).text)
        except:
            logging.exception(
                "connection time out.\t url:{duckling_url}, json_data:{pdata}".format_map(vars()))
            return final_output
        logging.debug('call duckling backend , response values:' + str(output))

        return [(ent['body'], ent['start']) for ent in output]


if __name__ == '__main__':
    import pprint

    text = """2009年，张首晟被清华大学特聘为教授。张首晟是2013年中国科学院外籍院士。=是/杨振宁/的弟子。
            2017年7月21日，美国加州大学洛杉矶分校University of California, Los Angeles(UCLA)华裔科学家王康隆、斯坦福大学华裔科学家张首晟、上海科技大学教授寇煦丰等团队合作在《科学》杂志上发表了一项重大发现：在整个物理学界历经80年的，。、；‘’【】、{}！@#￥`%……&（）——=+探索之后，他们终于,./<>?;':"[]{}|`!@#$%^&()_=发现了手性马约拉纳费米子的存在。
            张首晟–将其命名//为“天使粒子”"""

    entities = []
    duckling = DucklingEntityExtractor()
    duckling.process(text, entities)

    pprint.pprint(entities)
