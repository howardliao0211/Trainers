from typing import Any
import torch
import matplotlib.pyplot as plt
import pathlib

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

if __name__ == '__main__':
    import torch
    
    loss1 = torch.rand((64,))
    loss2 = torch.rand((128,))
    loss3 = torch.rand((1,))

    result = {
        'loss1' : loss1,
        'loss2' : loss2,
        'loss3' : loss3,
    }

    graph_loss(result)
