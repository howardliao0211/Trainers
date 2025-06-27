from typing import Any
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
import pathlib
import numpy as np

def graph_loss(result: dict[str, list[float]], title:str='', x_axis: str='Epochs', y_axis: str='Statistic') -> None:
    plt.figure()
    
    for label_name, data in result.items():
        plt.plot(data, label=label_name)
    
    plt.title(f'{title} {y_axis} VS {x_axis}')

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid()
    plt.legend()
    
    plt.show()

def graph_loss_animation_start(stat_names: list[str], title:str='', x_axis: str='Epochs', y_axis: str='Statistic') -> None:
    
    plt.ion()  # interactive mode on
    fig, ax = plt.subplots()
    ax.set_title(f'{title} {y_axis} VS {x_axis}')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    lines = {}
    x_data = []
    y_data = {stat: [] for stat in stat_names}

    for stat in stat_names:
        line, = ax.plot([], [], label=stat)
        lines[stat] = line

    ax.legend(loc='upper right')
    ax.grid()
    ax.set_xlim(0, 10)     # will grow dynamically
    ax.set_ylim(0, 1.0)    # adjust as needed

    return fig, ax, lines, x_data, y_data

def graph_loss_animation_update(epoch, new_result, ax, lines, x_data, y_data):
    x_data.append(epoch)
    for stat in new_result:
        y_data[stat].append(new_result[stat])
        lines[stat].set_data(x_data, y_data[stat])

    ax.relim()        # recompute limits
    ax.autoscale()    # autoscale for new data
    plt.draw()
    plt.pause(0.01)   # allow GUI event loop to update

def graph_loss_animation_end():
    plt.ioff()
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
    import random
    import time

    stat_names = [
        'Train Loss',
        'Test Loss',
        'I Dont Know'
    ]

    fix, ax, lines, x_data, y_data = graph_loss_animation_start(stat_names)

    for epoch in range(20):
        result = {}
        result['Train Loss'] = random.random()
        result['Test Loss'] = random.random()
        result['I Dont Know'] = random.random()
        
        print(f'epoch: {epoch}')
        graph_loss_animation_update(epoch, result, ax, lines, x_data, y_data)
        time.sleep(0.5)
        
    
    graph_loss_animation_end()
    
