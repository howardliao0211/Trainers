from torchvision import datasets, transforms
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from trainers.core import Trainer
from trainers.utils import StaticPlotter

if __name__ == '__main__':
    class LinearModel(nn.Module):
        def __init__(self, class_num: int):
            super(LinearModel, self).__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(128),
                nn.ReLU(),
                nn.LazyLinear(64),
                nn.ReLU(),
                nn.LazyLinear(class_num),
            )
    
        def forward(self, x):
            return self.net(x)

    train_loader = DataLoader(
        datasets.FakeData(size=1000, transform=transforms.ToTensor()),
        batch_size=32,
        shuffle=True,
    )

    test_loader = DataLoader(
        datasets.FakeData(size=200, transform=transforms.ToTensor()),
        batch_size=32,
        shuffle=False,
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = LinearModel(class_num=10).to(device)
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(
        name='Example',
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        plotter=StaticPlotter(),
        device=device
    )

    trainer.fit(epochs=5, graph=True)
