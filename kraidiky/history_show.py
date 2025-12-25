#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 22:16:26 2025
@author: vlad
Базовая визуализация для моего формата history
"""

import sys
import math
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
from pathlib import Path

#from kraidiky.tools import plot_gradient

import history as h

input_path:str = r"logs/20251224_2110-59ce01ee-5d84-4a08-b9ea-72acb0a6d2bd/last_history.pt"
if len(sys.argv) > 1:
    input_path = sys.argv[1]
parent_path = Path(input_path).parent
history = torch.load(input_path)


def color_by_id(color:int):
    ''' Вынимает цвета по айдишнику из стандартной матплотлибовской таблицы
        Args:
            color (int): Первый цвет в формате #RRGGBB
        Returns:
            str: цвет в формате #RRGGBB
    '''
    if isinstance(color, int):
        colors = mpl.rcParams["axes.prop_cycle"]
        return [c['color'] for c in colors][color % len(colors)]
    return color

########## ########## LOSS ########## ##########
loss =  h.get(history, h.keys.loss)
loss_train = h.get(loss, h.keys.train)
loss_train = [t for t,v in loss_train], [v for t,v in loss_train]
plt.plot(loss_train[0],loss_train[1], c=color_by_id(0), label=f'train: {loss_train[1][-1]:.3f} min:{min(loss_train[1]):.3f}')

window: int = 10
train = np.array(loss_train[1], dtype=float)
# Удлиняем массив крайними значениями
train = np.pad(train, (window, window), mode='edge')
indices = np.arange(len(loss_train[1])) + np.arange(-window, window + 1)[:, None] + window
train = np.mean(train[indices], axis=0)
plt.plot(loss_train[0],train, c=color_by_id(2))

loss_val = h.get(loss, h.keys.val)
loss_val = [t for t,v in loss_val], [v for t,v in loss_val]

plt.plot(loss_val[0],loss_val[1], c=color_by_id(1), label=f'val:{loss_val[1][-1]:.3f} min:{min(loss_val[1]):.3f}')

if h.keys.train_0 in loss:
    loss_train_0 = h.get(loss, h.keys.train_0)
    loss_train_0 = [t for t,v in loss_train_0], [v for t,v in loss_train_0]
    # Ищем последний общий индекс с val
    i1, i2 = len(loss_train_0[0])-1, len(loss_val[0])-1
    while (i1>=0) and (i2>=0) and (loss_train_0[0][i1] != loss_val[0][i2]):
        if loss_train_0[0][i1] > loss_val[0][i2]: i1 -= 1
        elif loss_train_0[0][i1] < loss_val[0][i2]: i2 -= 1
        else:
            break
    overfit = f'{loss_val[1][i2]/loss_train_0[1][i1]:.2f}' if (i1>=0) and (i2>=0) else '--'
    plt.plot(loss_train_0[0],loss_train_0[1], c=color_by_id(3), label=f'train[0]:{loss_train_0[1][-1]:.3f} overfit:{overfit}')

if h.keys.train_1 in loss:
    loss_train_1 = h.get(loss, h.keys.train_1)
    loss_train_1 = [t for t,v in loss_train_1], [v for t,v in loss_train_1]
    # Ищем последний общий индекс с val
    i1, i2 = len(loss_train_1[0])-1, len(loss_val[0])-1
    while (i1>=0) and (i2>=0) and (loss_train_1[0][i1] != loss_val[0][i2]):
        if loss_train_1[0][i1] > loss_val[0][i2]: i1 -= 1
        elif loss_train_1[0][i1] < loss_val[0][i2]: i2 -= 1
        else:
            break
    overfit = f'{loss_val[1][i2]/loss_train_1[1][i1]:.2f}' if (i1>=0) and (i2>=0) else '--'
    plt.plot(loss_train_1[0],loss_train_1[1], c=color_by_id(4), label=f'train[-1]:{loss_train_1[1][-1]:.3f} overfit:{overfit}')

plt.title("loss")
plt.legend()
plt.grid(which='both')
plt.yscale('log')
plt.savefig(parent_path/"loss.png")
plt.close('all')
print('loss_val:', loss_val)

########## ########## PERPLEXITY ########## ##########
plt.plot(loss_train[0],[math.exp(l) for l in train] , c=color_by_id(0), label=f'train: {math.exp(loss_train[1][-1]):.3f} min:{math.exp(min(loss_train[1])):.3f}')
plt.plot(loss_val[0],[math.exp(l) for l in loss_val[1]] , c=color_by_id(4), label=f'val: {math.exp(loss_val[1][-1]):.3f} min:{math.exp(min(loss_val[1])):.3f}')

plt.title("perplexity")
plt.legend()
plt.grid(which='both')
plt.yscale('log')
plt.savefig(parent_path/"perplexity.png")
plt.close('all')

########## ########## CONFIGURATION ########## ##########
configs = h.get(history, h.keys.config)
for config_key,values in list(configs.items()):
    if config_key == h.keys.model: ##### Это ключ со структурой модели, хотя логично было бы вынести это в отдельный ключ
        #print(f'model:\n{[(n,list(s)) for n,s in values]}')
        pass
    else:
        for key,series in values.items():
            if not all([item[1] == series[0][1] for item in series]):
                for scale in ['log','linear']:
                    x = np.array([i for i,v in series])
                    y = np.array([v for i,v in series])
                    plt.plot(x, y, label=f"{y[-1]:.2e} min:{y.min():.2e} max:{y.max():.2e}")
                    plt.scatter(x,y, c=color_by_id(2), s=9)
                    plt.legend()
                    plt.grid(which='both')
                    plt.title(key)
                    plt.yscale(scale)
                    plt.savefig(parent_path/f"{config_key}-{key}-{scale}.png")
                    plt.close('all')
            else:
                print(f'{config_key}/{key}={series[0][1]}')
