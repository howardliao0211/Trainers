import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers.core import Trainer
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

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


def main():
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(   
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    model = LinearModel(class_num=10)
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader
    )
    trainer.fit(epochs=5)


if __name__ == '__main__':
    main()


