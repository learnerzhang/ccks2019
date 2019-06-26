#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-24 11:35
# @Author  : zhangzhen
# @Site    : 
# @File    : __init__.py.py
# @Software: PyCharm
"""
    实体抽取
"""
import requests


class HttpSessionContext:
    """ 根据requests session 创建 http请求的上下文环境, 用于with 或者手动创建连接, 关闭连接

    """

    def __init__(self):
        self.session = requests.Session()

    def __enter__(self):
        self._set_adapter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
        return exc_type is not None

    def _set_adapter(self):
        a = requests.adapters.HTTPAdapter(max_retries=3)
        b = requests.adapters.HTTPAdapter(max_retries=3)
        self.session.mount('http://', a)
        self.session.mount('https://', b)

    def open(self):
        self._set_adapter()
        return self.session

    def close(self):
        self.session.close()