import json


# 写json
def readJson(path):
    f = open(path, encoding='utf-8')
    a = json.load(f)
    f.close()
    return a


# 读取json
def toJson(dic, path):
    f = open(path, 'w', encoding='utf-8')
    jsonData = json.dumps(dic, indent=4, ensure_ascii=False)
    f.write(jsonData)
    f.close()


# 计算各种指标
def getMetrices(tp, fp, tn, fn):
    num = (tp + tn + fp + fn)
    acc = (tp + tn) / num
    pre = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * pre * recall / (pre + recall)
    return acc, pre, recall, f1


wwm_bert_path = r'D:\公用数据\tfhub\chinese_roberta_wwm_ext_L-12_H-768_A-12'
bert_name = 'bert-base-chinese'
cache_dir = './cache'
