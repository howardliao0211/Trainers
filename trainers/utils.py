from typing import Any
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
import pathlib
import numpy as np

def graph_loss(result: dict[str, list[float]], x_axis: str='Epochs', y_axis: str='Statistic') -> None:
    plt.figure()
    
    for name, data in result.items():
        plt.plot(data, label=name)
    
    plt.title(f'{y_axis} VS {x_axis}')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid()
    plt.legend()
    
    plt.show()

def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> dict:
    print(f'Load Checkpoint: {checkpoint_path}')

    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    check = {}

    for k, v in checkpoint.items():
        if k == 'model_state_dict':
            model.load_state_dict(checkpoint['model_state_dict'])

        elif k == 'optimizer_state_dict':
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        else:
            print(f'{k}: {v}')
            check[k] = v
        
    return check

def graph_from_checkpoint(checkpoint_dir: str, x_axis: str='Epochs', y_axis: str='Statistic') -> None:
    checkpoint_path = pathlib.Path(checkpoint_dir)

    statistics = defaultdict(list)
    for path in checkpoint_path.rglob('*/*.pt'):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        for k, v in checkpoint.items():
            if 'state_dict' in k:
                continue
            statistics[k].append(v)
    
    assert 'epoch' in statistics, 'Epoch number should be included in checkpoints. '

    plt.figure()

    x = statistics['epoch']
    for k, v in statistics.items():
        if k == 'epoch':
            continue
        
        plt.plot(x, v, label=k)
    
    plt.title(f'{y_axis} VS {x_axis}')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid()
    plt.legend()
    
    plt.show()


if __name__ == '__main__':
    graph_from_checkpoint(r'Checkpoints')
