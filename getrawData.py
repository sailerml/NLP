#coding=utf-8
##将原始数据转化为normal样式

import json
import xml.dom.minidom
import jieba




def getRaw(datapath):
    dom = xml.dom.minidom.parse(datapath)
    root = dom.documentElement
    rawdata = []
    weibos = root.getElementsByTagName('weibo')
    for weibo in weibos:
        weiboData = {"label":"","text":""}
        weiboData["label"] = weibo.getAttribute("emotion-type1")
        sentences = weibo.getElementsByTagName('sentence')
        for sentence in sentences:
            sent = sentence.firstChild.data
            weiboData['text'] += sent
        rawdata.append(weiboData)


    return rawdata

