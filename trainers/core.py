from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Callable
from .graph import graph_loss
from collections import defaultdict
import torch

@dataclass
class BaseTrainer(ABC):
    """
    Base class for all trainers. This class defines the interface for training and evaluation methods.
    """

    model: nn.Module
    optimizer: Optimizer
    loss_fn: Callable
    train_loader: DataLoader
    test_loader: DataLoader

    def fit(self, epochs: int, graph: bool=False) -> None:
        """
        Train the model and optionally plot loss in real-time.
        
        Args:
            epochs (int): Number of training epochs.
        """
        losses = defaultdict(list)

        print("Training the model...")
        for epoch in range(epochs):
            print(f'============ Epoch {epoch + 1} ============')
            
            if self.train_loop is not None:
                train_losses = self.train_loop()
                losses['Train Loss'].extend(train_losses)
            
            if self.test_loop is not None:
                test_losses = self.test_loop()
                losses['Test Loss'].extend(test_losses)
        
        if graph:
            graph_loss(losses)

    @abstractmethod
    def train_loop(self) -> list[float]:
        """
        Perform one training loop over the dataset.
        """
        pass

    @abstractmethod
    def test_loop(self) -> list[float]:
        """
        Perform one evaluation loop over the dataset.
        """
        pass


@dataclass
class Trainer(BaseTrainer):
    """
    Concrete implementation of the BaseTrainer class. This class provides the actual training and evaluation logic.
    """

    record_loss_batch: int = 10

    def train_loop(self) -> list[float]:
        self.model.train()
        
        losses = []
        for batch, (inputs, labels) in enumerate(self.train_loader):
            # Forward pass
            predict = self.model(inputs)
            loss = self.loss_fn(predict, labels)
            
            # Backward pass & optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % self.record_loss_batch == 0:
                losses.append(loss.item())
                index = (batch + 1) * self.train_loader.batch_size
                print(f'    loss: {loss.item(): 5f} ----- {index: 6d} / {len(self.train_loader.dataset)}')
        
        return losses
    
    def test_loop(self) -> list[float]:
        self.model.eval()

        losses = []
        test_loss = 0
        with torch.no_grad():
            for batch, (inputs, labels) in enumerate(self.test_loader):
                predict = self.model(inputs)
                loss = self.loss_fn(predict, labels).item()
                test_loss += loss

                if batch % self.record_loss_batch == 0:
                    losses.append(loss)

        print(f'Test Loss: {test_loss / len(self.test_loader.dataset)}')
        return losses


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

    trainer.fit(epochs=5, graph=True)
