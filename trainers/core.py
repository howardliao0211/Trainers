from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from typing import Callable, Optional
from .utils import Plottable
from collections import defaultdict
import torch
import pathlib
import datetime

@dataclass
class BaseTrainer(ABC):
    """
    Base class for all trainers. This class defines the interface for training and evaluation methods.
    """

    name: str
    model: nn.Module
    optimizer: Optimizer
    scheduler: Optional[LRScheduler]
    loss_fn: Callable
    train_loader: DataLoader
    test_loader: DataLoader
    device: torch.device
    plotter: Plottable

    def fit(self, epochs: int, trained_epochs: int=0, graph: bool=False, save_check_point: bool=False) -> None:
        """
        Train the model and optionally plot loss in real-time.
        
        Args:
            epochs (int): Number of training epochs.
        """

        print("Training the model...")
        for epoch in range(epochs):
            epoch_idx = epoch + trained_epochs + 1

            print(f'============ Epoch {epoch_idx}/{epochs + trained_epochs} ============')
            
            train_state = self.train_loop()
            test_state = self.test_loop()
            current_statistic = {}
            current_statistic.update(train_state)
            current_statistic.update(test_state)

            if save_check_point:
                # Create Checkpoint Directory
                date = datetime.datetime.today().strftime("%Y%m%d")
                time = datetime.datetime.now().strftime("%H%M%S")
                checkpoint_dir = pathlib.Path(f'Checkpoints') / self.name / date
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Create Checkpoint Path
                checkpoint_name = f'{self.name}_epoch{epoch_idx}_{date}_{time}.pt'
                checkpoint_path = str(checkpoint_dir/checkpoint_name)
                checkpoint_dict = self.get_checkpoint_dict(self.model, self.optimizer, self.scheduler, epoch_idx, current_statistic)
                torch.save(checkpoint_dict, checkpoint_path)

            if graph:
                self.plotter.plot(
                    title=self.name,
                    result=current_statistic,
                    epoch=epoch,
                    is_finish=epoch == epochs-1
                )

    @abstractmethod
    def train_loop(self) -> dict:
        """
        Perform one training loop over the dataset.
        """
        pass

    @abstractmethod
    def test_loop(self) -> dict:
        """
        Perform one evaluation loop over the dataset.
        """
        pass

    def get_checkpoint_dict(self, model: nn.Module, optimizer: Optimizer, scheduler: Optional[LRScheduler], epoch: int, statistic: dict) -> dict:
        """
        Get checkpoint.
        """
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        checkpoint_dict.update(statistic)

        if scheduler is not None:
            checkpoint_dict['scheduler_state_dict'] = scheduler.state_dict()

        return checkpoint_dict


@dataclass
class Trainer(BaseTrainer):
    """
    Concrete implementation of the BaseTrainer class. This class provides the actual training and evaluation logic.
    """

    record_loss_batch: int = 10

    def train_loop(self) -> dict:
        self.model.train()
        
        train_loss = 0.0

        for batch, (inputs, labels) in enumerate(self.train_loader):
            
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            predict = self.model(inputs)
            loss = self.loss_fn(predict, labels)
            
            # Backward pass & optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % self.record_loss_batch == 0:
                train_loss += loss.item()
                print(f'    loss: {loss.item(): 5f} ----- {batch+1: 6d} / {len(self.train_loader)}')
        
        train_loss /= len(self.train_loader)
        return {'Train Loss': train_loss}
    
    def test_loop(self) -> dict:
        self.model.eval()

        test_loss = 0.0
        with torch.no_grad():
            for batch, (inputs, labels) in enumerate(self.test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                predict = self.model(inputs)
                loss = self.loss_fn(predict, labels).item()
                test_loss += loss

        test_loss /= len(self.test_loader)
        print(f'Test Loss: {test_loss}')
        return {'Test Loss': test_loss}

