#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 22:16:26 2025
@author: vlad
Подбор гиперпараметров по максимизации loss_val налету. Я знаю, что протечка несущественна, но при final run чтобы избежать претензий можно в этот калибровочный val отрезать последний батч трейна, тем более, что данных у меня как у дурака махорки...'
"""
from types import SimpleNamespace
from typing import Callable, Any

import kraidiky.history as h

class Affector:
    def __init__(self, name:str, tolerance:float=1.0001):
        """ tolerance - Множитель для полученного в эксперименте loss-а,
        нужен чтобы эксперименты не меняющие лосс сколько неибудь существенно не вызывали шума.
        Может быть меньше 1, например 0,9995 и тогда этот аффектор будет предпочитаться, если это не существенно повлияло на loss."""
        if name is not None:
            self.name = name
        self.tolerance = tolerance
    def affected_names(self) -> set[str]:
        ''' Переопределять в наследниках, которые влияют на группу имён, а не на одно '''
        return set([self.name])
    def __call__(self, cfg:dict[str, float | int], history:dict) -> dict[str, float | int] | None:
        return None
class ExperimentData:
    def __init__(self, affectors:list[list[Affector]]):
        """ affectors - Массив массивов аффекторов принятие которых определяется тем дают ли они выйгрыш в лоссе.
        На втором уровне взаимоисключающие - Если хотя бы один из них принят, остальные не проверяются,
        а выбраный пробуем применять ещё пока даёт выигрыш. Например [lr_up,lr_down]
        На первом уровне - такие группы, которые все будут попробованы по очереди, возможно и не по разу,
        например: [[lr_up, lr_down], [wd_up,wd_down]], Пустой массив означает, что на этом этапе никаких экспериментов проводиться не будет, просто пошедулили и пошли вперёд.
        Если хочется попробовать менять lr, wd, потом снова lr - обычно это происходит в начале, просто повторите гиперапараметры. """
        assert all([isinstance(next(iter(a)), Affector) for a in affectors]), "Аффекторы эксперимента должны быть двухуровневыми, взимодополняющие и взаимоисключающие"
        self.affectors:list[list[Affector]] = affectors

class HyperparametersTuning: 
    def __init__(self, load_state:Callable[[tuple[dict,dict]|str],None], save_state:Callable[[str],tuple[dict,dict]], logger:Callable[[str],None] = None):
        '''save_state принимает id - ['initial'|'best'] '''
        self.load_state:Callable[[tuple[dict,dict]|str],None] = load_state # Лоадер меняет в том числе и текущий шаг
        self.save_state:Callable[[str],tuple[dict,dict]] = save_state
        self.registered_parameters:dict[str, SimpleNamespace] = {}
        self.program:list[ExperimentData] = None
        self.index:int = [0,0,0] # Индексы текщуего аффектора, все три уровня, Первый прогон - без измнения в конфиге
        self._reset_experiment_state() 
        self.logger = logger

    def _reset_experiment_state(self):
        self.index[1],self.index[2], = (0,0)
        self.initial_state:tuple[dict,dict] = None # [model_state, history_state] - Начальное смостояние на входе в эксперимент
        self.initial_config:dict = None # Конфиг начального состояния, получается после применения безусловных аффекторов
        self.best_state:tuple[dict,dict] = None # [model_state, history_state] - Бэкап состояния в конце лучшего из экспериментов
        self.best_config:dict = None # Конфиг лучшей попытки - служит началом следующих экспериментов
        self.best_loss_val:float = 100500 # Лосс лучшей попытки, чтобы избавиться от привычки лазить в сохзранённое состояние, иало ли в каком формате оно хранится
        self.current_cfg:dict = None # Текущий конфиг на котором сейчас идёт эксперимент
        self.repeat:bool = False

    def register_affected_parameter(self, name:str, getter:Callable[[],float | int], setter:Callable[[float | int],None]):
        self.registered_parameters[name] = SimpleNamespace(get=getter, set=setter)

    def set_program(self, program:list[list[list[Affector]] | ExperimentData]):
        """ Программа тюнинга трёхуровневая.
        На самом нижнем уровне список аффекторов взаимоисключающий, оптимизатор сначала пробует первый, если он эфективен делает им ещё шаги, и игнорирует все остальные в этом уровне, если нет - переходит к следующему.
        Второй снизу уровень - те аффекторы, которые должны быть попробованы в ходе этого шага, каждый раз откатывая степ. Зокончил с первой группой, вне зависисмости от результата - пробует следующую.
        Третий снизу уровень - Последовательность аффекторков на шаги алгоритма, попробовал первую группу на одном шаге, вторую на следующем и так далее.
        Если на втором уровне нужны повторения или schedulers то вместо массива передаём ExperimentData """
        self.program = [e if isinstance(e, ExperimentData) else ExperimentData(e) for e in program]
        self.index[0] = 0
        self._reset_experiment_state()

    # Какие могут быть варианты прихода в эту функцию?
    # 1) Начинаем эксперимент, и должны на выходе пойти в дефолт.
    # 2) Пришли послед дефолтного прохода, должны или начать эксперименты или выйти если в данных нет ни одного подходящего аффектора
    # 3) После аффектора нам стало лучше и нам надо продолжать этот аффектор
    # 4) После аффектора нам лучше не стало.
    #   - Если мы поваторяли, то надо переходить к следующей группе аффекторов (или на выход)
    #   - Если мы не повторяли, то переходим к следующему аффектору в группе
    def _current_affector(self) -> Affector:
        current_experiment = self.program[self.index[0]]
        if current_experiment.affectors is None: return None
        if len(current_experiment.affectors) <= self.index[1]: return None
        if len(current_experiment.affectors[self.index[1]]) <= self.index[2]: return None
        return current_experiment.affectors[self.index[1]][self.index[2]]

    def cycle(self, history:dict):
        last_loss = h.loss_val(history)[-1][1] # Даже если нулёвый вызов там всё равно уже есть хотя бы нулевая запись
        ''' Вызывается после того как loss.val уже посчитан, но перед train-ом
        history нужен чтобы доставать из него текущее состояние. Тут и loss.val и step если нужно для расписаний. '''
        if self.initial_state is None: # Первый шаг по инерции без экспериментов
            self._start_phase(history)
        elif (self.best_state is None): # Это после первого прохождения, теперь надо запустить первый аффектор, который найдём, если найдём...
            self.best_state = self.save_state('best')
            self.best_config = self.initial_config
            self.best_loss_val = last_loss
            self.logger and self.logger(f'Initial Run result:{last_loss:.4f}; step:{h.loss_val(history)[-1][0]}')
            self._start_experiment(history)
        elif (last_loss * self.program[self.index[0]].affectors[self.index[1]][self.index[2]].tolerance < self.best_loss_val):
            current_affector = self._current_affector()
            if (current_affector is not None) and (hasattr(current_affector,'postprocessing')):
                current_affector.postprocessing(self.best_loss_val, h.get(h.get(history, h.keys.loss),h.keys.val)[-1][1])
            self.logger and self.logger('Better Run result:{0:.4f}({1:+.4f}) tolerance:{2:.1e}; from best:{3}; from initial:{4}; index:{5}; step:{6}'.format(
                                            last_loss, last_loss - self.best_loss_val,
                                            self.program[self.index[0]].affectors[self.index[1]][self.index[2]].tolerance-1,
                                            self._diff(self.current_cfg, self.best_config, self.initial_config), self._diff(self.current_cfg, self.initial_config),
                                            self.index, h.loss_val(history)[-1][0]
                                        ))
            # Это лучший результат, значит надо повторять эту ветку изменения на том же аффекторе
            self.best_state = self.save_state('best')
            self.best_config = self.current_cfg
            self.best_loss_val = last_loss
            self.load_state(self.initial_state)
            self.current_cfg = self._applay_affector(self.program[self.index[0]].affectors[self.index[1]][self.index[2]], self.current_cfg, history)
            if self.current_cfg is not None:
                self._applay_config(self.current_cfg)
                self.repeat = True
            else: # Упёрлись в предельное значение, Значит эта исключающая серия словила предел, переходим к следующей
                self.index[1] += 1
                self.index[2] = 0
                self._start_experiment(history)
        else: # not the best attempt
            current_affector = self._current_affector()
            if (current_affector is not None) and (hasattr(current_affector,'postprocessing')):
                current_affector.postprocessing(self.best_loss_val, h.get(h.get(history, h.keys.loss),h.keys.val)[-1][1])
            self.logger and self.logger('Worse Run result:{0:.4f}({1:+.4f}) tolerance:{2:.1e}; from best:{3}; from initial:{4}; index:{5}; step:{6}'.format(
                                            last_loss, last_loss - self.best_loss_val,
                                            self.program[self.index[0]].affectors[self.index[1]][self.index[2]].tolerance-1,
                                            self._diff(self.current_cfg, self.best_config, self.initial_config), self._diff(self.current_cfg, self.initial_config),
                                            self.index, h.loss_val(history)[-1][0]
                                        ))
            if self.repeat: # Это было повторение, значит уходим на следующий уровень
                self.index[1] += 1
                self.index[2] = 0
            else: # Это всего лишь неудачная попытка пробуем следующий аффектор нашего уровня
                self.index[2] += 1
            self._start_experiment(history)
            
    def _resolve_current_affector(self, cfg:dict[str, float|int], history:dict):
        # Берёт ближайший не пустой аффектор, По мере необходимости перебирает для этого индексы
        if self.index[0] >= len(self.program):
            raise Exception('Такого не должно было происходить')
        phase:ExperimentData = self.program[self.index[0]]
        if self.index[1] >= len(phase.affectors):
            return None
        experiment = phase.affectors[self.index[1]]
        if self.index[2] >= len(experiment):
            self.index[1],self.index[2] = (self.index[1]+1,0)
            return self._resolve_current_affector(cfg, history)
        affector = experiment[self.index[2]]
        next_cfg = self._applay_affector(affector, cfg, history)
        if next_cfg is not None:
            return next_cfg
        else:
            self.index[2] = self.index[2] + 1
            return self._resolve_current_affector(cfg, history)

    def _applay_affector(self, a:Affector, cfg:dict[str,Any], history:dict) -> dict[str,Any]:
        cfg = cfg.copy()
        for name in a.affected_names():
            if name not in self.initial_config:
                self.initial_config[name] = self.registered_parameters[name].get()
        for name,value in self.initial_config.items():
            if name not in cfg:
                cfg[name] = self.initial_config[name]
        return a(cfg, history)
    def _applay_config(self, cfg:dict[str,int|float]):
        for n in cfg:
            if self.initial_config[n] != cfg[n]:
                self.registered_parameters[n].set(cfg[n])
    def _diff(self, new:dict, *old:list[dict]):
        result = {}
        for k in new:
            for one_off_olds in old:
                if k in one_off_olds:
                    if one_off_olds[k] != new[k]:
                        result[k] = (one_off_olds[k], new[k])
                    break
        return result

        return {k:(old[k],new[k]) if k in old else ('--',new[k]) for k in new if (k not in old) or (old[k] != new[k])}
    def _start_phase(self, history):
        self.initial_config = {}
        # Цикл начинается с дефолтного прохождения, применяются только шедулеры
        self.current_cfg = {}
        self._applay_config(self.current_cfg)
        self.initial_state = self.save_state('initial') # После всех шедулеров, чтобы состояние оптимизатора было уже проработанное
        self.initial_config = self.current_cfg.copy()
        self.repeat = False
        self.logger and self.logger(f'Phase; step:{h.loss_val(history)[-1][0]}')
    def _start_experiment(self,history):
        self.current_cfg = self._resolve_current_affector(self.best_config, history)
        if self.current_cfg is not None: # Пошли в эксперимент
            self.load_state(self.initial_state)
            self._applay_config(self.current_cfg)
            self.repeat = False
        else: # Не нашли ни одного подходящего аффектора. Выходим в следующий эксперимент
            self.load_state(self.best_state)
            self._applay_config(self.best_config)
            self.best_state = None
            self.best_config = None
            self.best_loss_val = 100500
            self.index = [(self.index[0] + 1) % len(self.program),0,0]
            self._start_phase(history)

        
def sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0
##### ##### Аффекторы и всякие композиты ##### #####
class IncrementialAffector(Affector):
    '''Аффектор, который за одно действие меняет гиперпараметр на фиксированный шаг пока не выйдет за границу. '''
    def __init__(self, name:str, effect:float | int, limit: float | int, epsilon:float=0.01, tolerance:float=1.0001):
        '''
        epsilon - отличие меньше чем epsilon*effect от лимита считается попаданием в лимит, это в основном, чтобы избежать проблем с округлением
        '''
        super(IncrementialAffector, self).__init__(name, tolerance=tolerance)
        self.effect = effect
        self.limit = limit
        self.epsilon=epsilon
    def __call__(self, cfg:dict[str, float | int], history:dict) -> dict[str, float | int] | None:
        value = cfg[self.name]
        if (self.limit is not None) and ((value >= self.limit) if self.effect > 0 else (value <= self.limit)):
            return None
        value = value + self.effect
        if self.limit is not None:
            value = min(value, self.limit) if self.effect > 0 else max(value, self.limit)
            if (self.effect > 0) and ((self.limit-value) < (self.effect*self.epsilon)):
                value = self.limit
            if (self.effect < 0) and ((value - self.limit) < (-self.effect*self.epsilon)):
                value = self.limit
        cfg = cfg.copy()
        cfg[self.name] = value
        return cfg
class ExponentialAffector(Affector):
    '''Аффектор, который за одно действие меняет гиперпараметр на множитель пока не выйдет за границу. '''
    def __init__(self, name:str, effect:float | int, limit: float | int, epsilon:float=0.01, tolerance:float=1.0001):
        '''
        epsilon - отличие меньше чем epsilon*effect от лимита считается попаданием в лимит, это в основном, чтобы избежать проблем с округлением
        '''
        super(ExponentialAffector, self).__init__(name,tolerance=tolerance)
        self.effect = effect
        self.limit = limit
        self.epsilon=epsilon
    def __call__(self, cfg:dict[str, float | int], history:dict) -> dict[str, float | int] | None:
        value = cfg[self.name]
        if (self.limit is not None) and ((value >= self.limit) if self.effect > 1 else (value <= self.limit)):
            return None
        value = value*self.effect
        if self.limit is not None:
            value = min(value, self.limit) if self.effect > 1 else max(value, self.limit)
            if (self.effect > 1) and ((self.limit-value) < (self.effect*self.epsilon)):
                value = self.limit
            if (self.effect < 1) and ((value - self.limit) < (-self.effect*self.epsilon)):
                value = self.limit
        cfg = cfg.copy()
        cfg[self.name] = value
        return cfg
class Between(Affector):
    ''' Класс рассчитан только на аффекторы, влияющие на один параметр. '''
    class delegate(Affector):
        def __init__(self, src:Affector):
            super().__init__(src.name, src.tolerance)
            self.affector = src
            self.reset()
        def reset(self):
            self.value = None
            self.loss = None
        def __call__(self, cfg:dict[str, float | int], history:dict) -> dict[str, float | int] | None:
            center = cfg[self.affector.name]
            cfg = self.affector(cfg, history)
            self.value = (cfg[self.affector.name] - center) if cfg is not None else None
            self.loss = None
            return cfg
        def postprocessing(self, best_loss:float, our_loss:float):
            self.loss = our_loss - best_loss

    def __init__(self, forth:Affector, back:Affector, tolerance = 1.0001):
        super().__init__(forth.name, tolerance)
        assert forth.name == back.name
        self.forth = Between.delegate(forth)
        self.back = Between.delegate(back)
    def __call__(self, cfg:dict[str,float|int], history:dict) -> dict[str,float|int]:
        ''' Вызываеься только после того как туда-сюда с первого раза зафейлились зафейлились. '''
        x1 = self.forth.value
        x3 = self.back.value
        y1 = self.forth.loss
        y3 = self.back.loss
        a = (y3*x1-y1*x3)/(x3-x1)/x1/x3 
        b = (y1-a*x1*x1)/x1 #(y1​-a*x1*x1)/x1
        cfg = cfg.copy()
        cfg[self.name] = cfg[self.name]-b/(2*a)
        return cfg
    def affectors(self) -> list[Affector]:
        return [self.forth, self.back, self]
        
lr_up = ExponentialAffector('lr', pow(10,1/5), 1e-1)
lr_down = ExponentialAffector('lr', 1/pow(10,1/5), 1e-7)
lr = [lr_down,lr_up]
adam_lr_up = ExponentialAffector('adam_lr', pow(10,1/5), 1e-1)
adam_lr_down = ExponentialAffector('adam_lr', 1/pow(10,1/5), 1e-7)
adam_lr = [adam_lr_down,adam_lr_up]
muon_lr_up = ExponentialAffector('muon_lr', pow(10,1/5), 1e-1)
muon_lr_down = ExponentialAffector('muon_lr', 1/pow(10,1/5), 1e-7)
muon_lr = [muon_lr_down,muon_lr_up]
wd_up = ExponentialAffector('wd', pow(10,1/5), 1e-1)
wd_down = ExponentialAffector('wd', 1/pow(10,1/5), 1e-7)
wd = [wd_up, wd_down]
adam_wd_up = ExponentialAffector('adam_wd', pow(10,1/5), 1e-1)
adam_wd_down = ExponentialAffector('adam_wd', 1/pow(10,1/5), 1e-7)
adam_wd = [adam_wd_up, adam_wd_down]
muon_wd_up = ExponentialAffector('muon_wd', pow(10,1/5), 1e-1)
muon_wd_down = ExponentialAffector('muon_wd', 1/pow(10,1/5), 1e-7)
muon_wd = [muon_wd_up, muon_wd_down]
betas_up = ExponentialAffector('betas', pow(10,1/5), 1024, tolerance=1.000001)
betas_down = ExponentialAffector('betas', 1/pow(10,1/5), 1, tolerance=1.000001)
betas = [betas_up, betas_down]
betas_between = Between(betas_up, betas_down, tolerance=1)
muon_momentum_up = ExponentialAffector('muon_momentum', pow(10,1/5), 1000, tolerance=1.000001)
muon_momentum_down = ExponentialAffector('muon_momentum', 1/pow(10,1/5), 1, tolerance=1.000001)
muon_momentum = [muon_momentum_up, muon_momentum_down]
muon_momentum_between = Between(muon_momentum_up, muon_momentum_down, tolerance=1)

all_params = [adam_lr,muon_lr,betas,muon_momentum,adam_wd,muon_wd,]
all_grouped_params = [betas,muon_momentum,wd,lr]
momentums = [betas,muon_momentum]
momentums_between = [betas_between.affectors(), muon_momentum_between.affectors()]
momentums_between = [betas_between.affectors(), muon_momentum_between.affectors()]

programs = {'initial':[all_params+all_params+all_params]+[all_params]*100,
            'all_params':[]+[all_params]*100, # [[]]+ Предполагаем, что на момент запуска цикла инишиал параметры уже подобраны
            'between':[momentums_between+momentums_between+momentums_between] + [momentums_between]*100,
            'momentums':[momentums+momentums+momentums]+[momentums]*100,
            'default':[[lr],
                       [wd],
                       [lr],
                       [betas,muon_momentum]]}

