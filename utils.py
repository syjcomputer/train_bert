import json
import torch.nn as nn



def load_data(filename):
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        all = json.load(f)
        for l in all:
            D.append((l['text'], int(l['label'])))
    return D

def load_data1(filename):
    text = []
    label = []
    id = []
    with open(filename, 'r', encoding='utf-8') as f:
        all = json.load(f)
        for l in all:
            text.append(l['text'])
            label.append(int(l['label']))
            id.append(l['id'])
    return text, label, id


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.xavier_normal_(m.weight)
        #nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class ExistError(Exception):
    """自定义异常类"""
    def __init__(self, message):
        super().__init__(message)
        self.message = message
