from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Callable
import torch
import matplotlib.pyplot as plt

@dataclass
class BaseTrainer(ABC):
    """
    Base class for all trainers. This class defines the interface for training and evaluation methods.
    """
    @abstractmethod
    def fit(self, epochs: int, graph: bool = False) -> None:
        """
        Train the model on the given dataset.
        """
        pass


@dataclass
class Trainer(BaseTrainer):
    """
    Concrete implementation of the BaseTrainer class. This class provides the actual training and evaluation logic.
    """

    model: nn.Module
    optimizer: Optimizer
    loss_fn: Callable
    train_loader: DataLoader
    test_loader: DataLoader

    def fit(self, epochs: int) -> None:
        """
        Train the model and optionally plot loss in real-time.
        
        Args:
            epochs (int): Number of training epochs.
        """
        print("Training the model...")
        for epoch in range(epochs):
            print(f'============ Epoch {epoch + 1} ============')
            self.train_loop()
            self.test_loop()
    
    def train_loop(self):
        self.model.train()
        
        for batch, (inputs, labels) in enumerate(self.train_loader):
            # Forward pass
            predict = self.model(inputs)
            loss = self.loss_fn(predict, labels)
            
            # Backward pass & optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                index = (batch + 1) * self.train_loader.batch_size
                print(f'    loss: {loss.item(): 5f} ----- {index: 6d} / {len(self.train_loader.dataset)}')
    
    def test_loop(self) -> None:
        self.model.eval()

        test_loss = 0
        with torch.no_grad():
            for batch, (inputs, labels) in enumerate(self.test_loader):
                predict = self.model(inputs)
                test_loss += self.loss_fn(predict, labels).item()
        print(f'Test Loss: {test_loss / len(self.test_loader.dataset)}')


if __name__ == '__main__':
    from torchvision import datasets, transforms
    from torch.optim import SGD

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
