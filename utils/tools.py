# -*- coding: utf-8 -*-
from pickle import load,dump
import numpy as np
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = load(f)
        return data

def save_pickle(data,file_path):
    with open(file_path, 'wb') as f:
        dump(data,f)

class LayerActivations:#对某一层（layer_num)层插入钩子，记录其输出。使用方法： 需要先定义其
    features = None
    def __init__(self, model,Sequential=False,layer_num=None):
        if Sequential:
            self.hook = model[layer_num].register_forward_hook(self.hook_fn)
        else:
            self.hook = model.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output
    def remove(self):
        self.hook.remove()

#打印中间层的梯度信息
'''
method 1:
显示的保存梯度信息，增加内存量
y.retain_grad()
out.backward()
print(y.grad)
'''
#grads={}#需要提前声明字典
# def save_grad(name):
#     def hook(grad):
#         grads[name] = grad
#     return hook
'''
使用：
grads={}
y.register_hook(ave_grad('y'))
z.backward()
y_grad = grads['y']
'''
def save_grad(g):
    global feature_grad
    feature_grad=g
'''
使用：
y.register_hook(extract)
z.backward()
y_grad = features_grad
'''