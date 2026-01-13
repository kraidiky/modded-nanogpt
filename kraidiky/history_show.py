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

input_path:str = r"logs/20260111_2102-baseline/last_history.pt"
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
def SyncTimelines(X: np.ndarray, Y: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """ Линейная аппроксимация Y для X2 с экстраполяцией крайними значениями.
    X отсортирован по возрастанию, может содержать повторы. """
    X,Y,X2 = np.asarray(X), np.asarray(Y), np.asarray(X2)
    if len(X2) == 0:
        return np.array([])
    # Находим позиции для вставки (индексы правых границ)
    indices = np.searchsorted(X, X2, side='right') - 1
    # Ограничиваем индексы для интерполяции
    indices = np.clip(indices, 0, len(X) - 2)
    # Получаем левые и правые значения X и Y
    x_left = X[indices]
    x_right = X[indices + 1]
    y_left = Y[indices]
    y_right = Y[indices + 1]
    # Коэффициенты для линейной интерполяции
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(x_right != x_left,
                    (X2 - x_left) / (x_right - x_left),
                    0)
    # Линейная интерполяция
    Y2 = y_left + t * (y_right - y_left)
    # Экстраполяция: значения за границами X
    Y2[X2 <= X[0]] = Y[0]      # Меньше или равно минимуму
    Y2[X2 >= X[-1]] = Y[-1]    # Больше или равно максимуму
    return Y2

configs = h.get(history, h.keys.config)
for config_key,values in list(configs.items()):
    if config_key == h.keys.model: ##### Это ключ со структурой модели, хотя логично было бы вынести это в отдельный ключ
        #print(f'model:\n{[(n,list(s)) for n,s in values]}')
        pass
    else:
        for key,series in values.items():
            if not isinstance(series[0][1], str):
                (parent_path/"log").mkdir(parents=True,exist_ok=True)
                (parent_path/"per_loss").mkdir(parents=True,exist_ok=True)
                if not all([item[1] == series[0][1] for item in series]):
                    x = np.array([i for i,v in series])
                    y = np.array([v for i,v in series])
                    plt.plot(x, y, label=f"{y[-1]:.2e} min:{y.min():.2e} max:{y.max():.2e}")
                    #plt.scatter(x,y, c=color_by_id(2), s=9)
                    plt.legend()
                    plt.grid(which='both')
                    plt.title(key)
                    plt.savefig(parent_path/f"{config_key}-{key}.png")
                    plt.yscale("log")
                    plt.savefig(parent_path/"log"/f"{config_key}-{key}.png")
                    plt.close('all')
                    # Поставить в соответствие два ряда во втором из которых есть пропуски
                    plt.plot(SyncTimelines(loss_val[0],loss_val[1],x),y, label=f"{y[-1]:.2e} min:{y.min():.2e} max:{y.max():.2e}")
                    plt.legend()
                    plt.grid(which='both')
                    plt.title(f"{key} / loss")
                    plt.xlim((plt.xlim()[1],plt.xlim()[0]))
                    plt.xscale("log")
                    plt.savefig(parent_path/"per_loss"/f"{config_key}-{key}.png")
                    plt.close('all')
                    #if len(x) < 30: print(key, 'per loss', SyncTimelines(loss_val[0], loss_val[1],x),y)
                else:
                    print(f'{config_key}/{key}={series[0][1]}')
            else:
                print(f'{config_key}/{key}={series}')
