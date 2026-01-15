from dataclasses import dataclass, fields, is_dataclass
from typing import Union, List, Tuple, Any, Callable
from types import SimpleNamespace
import torch


factory = SimpleNamespace(list = lambda:[],
                          dict = lambda:{},
                          namespace = lambda:SimpleNamespace())
keys = SimpleNamespace(loss='loss', train='train', val='val', lr='lr',
                       train_0='train_0', train_1='train_1',
                       config='config', model='model',
                       structure='structure', zeros='zeros', zeros_per_layer='zeros_per_layer',
                       )

def get(target, name:str, default:Callable[[],Any] = None):
    if isinstance(target, SimpleNamespace):
        if not hasattr(target, name):
            if default is not None:
                setattr(target, name, default())
            else:
                raise Exception(f'There is not {name} at {set(dir(target))-set(dir(SimpleNamespace))}')
        return getattr(target, name)
    elif isinstance(target, dict):
        if name not in target:
            if default is not None:
                target[name] = default()
            else:
                raise Exception(f'There is not {name} at {target.keys()}')
        return target[name]
    else:
        raise Exception(f'Unexpected type: {type(target)}')

def empty():
    return {keys.loss:{keys.train:factory.list(), keys.val:factory.list()},
            keys.lr:factory.list(),
            keys.config:factory.dict(),
            keys.structure:factory.dict(),
            }

def loss_train(target:dict) -> tuple[list,list]:
    return get(get(target, keys.loss, factory.dict), keys.train, factory.list)
def loss_val(target:dict) -> tuple[list,list]:
    return get(get(target, keys.loss, factory.dict), keys.val, factory.list)
def lr(target:dict) -> tuple[list,list]:
    return get(target, keys.lr, factory.list)

def capture_model_state(target, model:torch.nn.Module, t:int):
    structure = get(target, keys.structure, factory.dict)
    if keys.model not in structure:
        structure['model'] = [(n,p.shape) for n,p in model.named_parameters()]
    zeros_per_layer = get(structure, keys.zeros_per_layer, factory.list)
    model_zeros_per_layer = [(p==0).sum().item() for p in model.parameters()]
    if (len(zeros_per_layer) == 0) or (any([p!=n for p,n in zip(zeros_per_layer[-1][1], model_zeros_per_layer)])):
        zeros_per_layer.append((t,model_zeros_per_layer))
    model_zeros = sum(model_zeros_per_layer)
    zeros = get(structure, keys.zeros, factory.list)
    if (len(zeros) == 0) or (zeros[-1][1] != model_zeros):
        zeros.append((t, model_zeros))
        
def capture_config(target, t:int, **configs):
    config = get(target, keys.config, factory.dict)
    for config_key,values in configs.items():
        pairs_list = get_key_value_pairs(values)
        if pairs_list is None:
            sequence = get(config, config_key, factory.list)
            if (len(sequence) == 0) or (sequence[-1][1] != values):
                sequence.append((t,values))
        else:
            child_config = get(config, config_key, factory.dict)
            for key,value in pairs_list:
                sequence = get(child_config, key, factory.list)
                if (len(sequence) == 0) or (sequence[-1][1] != value):
                    sequence.append((t,value))
    pass

def get_key_value_pairs(obj: Union[object, dict]) -> List[Tuple[str, Any]]:
    """
    Преобразует объект в список пар (ключ, значение).
    Поддерживает: dataclass, SimpleNamespace, dict, обычные объекты.
    """
    
    # Для словаря
    if isinstance(obj, dict):
        return [(key, value) for key, value in obj.items()]
    
    # Для dataclass
    elif is_dataclass(obj):
        return [(field.name, getattr(obj, field.name)) for field in fields(obj)]
    
    # Для SimpleNamespace
    elif isinstance(obj, SimpleNamespace):
        return [(key, value) for key, value in vars(obj).items()]
    
    # Для обычного объекта - используем __dict__
    elif hasattr(obj, '__dict__'):
        return [(key, value) for key, value in vars(obj).items() 
                if not (key.startswith('_') or callable(value))]
    
    return None

if __name__ == "__main__":
    h = empty()
    loss_train(h).append((0,95.))
    loss_val(h).append((0,99.))
    lr(h).append((0,3e-4))
    model = torch.nn.Sequential(torch.nn.Linear(2, 10), torch.nn.Linear(10,3))
    with torch.no_grad():
        model[0].weight[torch.rand_like(model[0].weight)<0.3]=0
    capture_model_state(h, model, 0)
    
    @dataclass
    class Hyperparameters:
        train_files: str = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
        val_tokens: int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
        train_accumulate:int = 8
    cfg = Hyperparameters()
    capture_config(h,0,cfg=cfg)
    cfg.train_accumulate = 16
    capture_config(h,1,cfg=cfg)
    print(h)

