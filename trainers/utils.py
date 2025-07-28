from typing import Any
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
from typing import Protocol, Optional
import torch
import pathlib
import numpy as np

class Plottable(Protocol):
    """
    Protocol for plottable object
    """

    def plot(self, title: str, epoch: int, result: dict[str, list[float]], is_finish: bool, *args, **kwargs) -> None:
        """
        Plot a graph of statistic over epoch.

        Args:
            result (dict[str, list[float]]): a dictionary contend statistic name and its values.
        """
        pass

class StaticPlotter:

    def __init__(self, x_axis='Epochs', y_axis='Statistic'):
        self.x_axis = x_axis
        self.y_axis = y_axis

        self.is_started = False
    
    def plot(self, title: str, epoch: int, result: dict[str, list[float]], is_finish: bool, *args, **kwargs) -> None:
        
        if self.is_started is False:
            self.is_started = True
            self.x_data, self.y_data = self._plot_start()
        
        self._plot_update(
            epoch=epoch,
            result=result,
            x_data=self.x_data,
            y_data=self.y_data
        )

        if is_finish:
            self.is_started = False
            self._plot_end(
                title=title,
                x_data=self.x_data,
                y_data=self.y_data
            )
    
    def _plot_start(self):
        x_data = list()
        y_data = defaultdict(list)
        return x_data, y_data

    def _plot_update(self, epoch: int, result: dict[str, float], x_data: list, y_data: dict[str, list]) -> None:
        x_data.append(epoch)
        for name, value in result.items():
            y_data[name].append(value)

    def _plot_end(self, title: str, x_data: list, y_data: dict) -> None:
        plt.figure()
        
        for label_name, data in y_data.items():
            plt.plot(x_data, data, label=label_name)
        
        plt.title(f'{title} {self.y_axis} VS {self.x_axis}')

        plt.xlabel(self.x_axis)
        plt.ylabel(self.y_axis)
        plt.grid()
        plt.legend()
        
        plt.show()

class AnimatePlotter:

    def __init__(self, x_axis='Epochs', y_axis='Statistic'):
        self.x_axis = x_axis
        self.y_axis = y_axis

        self.is_started = False
    
    def plot(self, title: str, epoch: int, result: dict[str, list[float]], is_finish: bool, *args, **kwargs) -> None:
        if self.is_started is False:
            self.is_started = True
            self.fig, self.ax, self.lines, self.x_data, self.y_data = self._plot_loss_animation_start(
                stat_names=list(result.keys()),
                title=title,
                x_axis=self.x_axis,
                y_axis=self.y_axis
            )
        
        self._plot_loss_animation_update(
            epoch=epoch,
            new_result=result,
            ax=self.ax,
            lines=self.lines,
            x_data=self.x_data,
            y_data=self.y_data
        )

        if is_finish:
            self._plot_loss_animation_end()
            self.is_started = False

    def _plot_loss_animation_start(self, stat_names: list[str], title:str, x_axis: str, y_axis: str) -> None:
        
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

    def _plot_loss_animation_update(self, epoch, new_result, ax, lines, x_data, y_data):
        x_data.append(epoch)
        for stat in new_result:
            y_data[stat].append(new_result[stat])
            lines[stat].set_data(x_data, y_data[stat])

        ax.relim()        # recompute limits
        ax.autoscale()    # autoscale for new data
        plt.draw()
        plt.pause(0.01)   # allow GUI event loop to update

    def _plot_loss_animation_end(self):
        plt.ioff()
        plt.show()

def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], device: torch.device) -> dict:
    print(f'Load Checkpoint: {checkpoint_path}')

    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    check = {}

    for k, v in checkpoint.items():
        if k == 'model_state_dict':
            model.load_state_dict(checkpoint[k])

        elif k == 'optimizer_state_dict':
            optimizer.load_state_dict(checkpoint[k])
        
        elif k == 'scheduler_state_dict' and scheduler is not None:
            scheduler.load_state_dict(checkpoint[k])

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

    plotter = AnimatePlotter()

    for epoch in range(20):
        result = {}
        result['Train Loss'] = random.random()
        result['Test Loss'] = random.random()
        result['I Dont Know'] = random.random()
        
        print(f'epoch: {epoch}')
        plotter.plot(
            title='test',
            epoch=epoch,
            result=result,
            is_finish=epoch==19
        )
    
