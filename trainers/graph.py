import matplotlib.pyplot as plt
from dataclasses import dataclass

def graph_loss(result: dict[str, list[float]], x_axis: str='epochs', y_axis: str='Loss') -> None:
    plt.figure()
    
    for name, data in result.items():
        plt.plot(data, label=name)
    
    plt.title(f'{y_axis} VS {x_axis}')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid()
    plt.legend()
    
    plt.show()

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
