from typing import Any
import optuna
from optuna.trial import Trial, TrialState
import os
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import JUND_Dataset
from models import MLP, LSTM, CNN

def get_dataloader(
    data_dir: str,
    batch_size: int,
) -> DataLoader:
    """Returns a DataLoader for JUND transcription factor data.

    Args:
        data_dir (str): Path to directory containing data files.
        batch_size (int): Number of samples to load per batch.

    Returns:
        DataLoader: PyTorch DataLoader for JUND transcription factor data.
    """
    dataset = JUND_Dataset(data_dir=data_dir)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader

def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    optimizer: optim.Optimizer,
): 
    """Performs one epoch of model training.

    Args:
        model (nn.Module): A model that takes a batch of chromosome segments (X) and their accessibility (a). 
        data_loader (DataLoader): A DataLoader that returns batches of chromosome segments (X), transcription factor binding indicator (y), segment weight (w), accessibility (a).
        device (str): The device on which the PyTorch Tensors will be allocated. 
        optimizer (optim.Optimizer): PyTorch Optimizer.
    """
    model.train()

    for batch_idx, batch_data in enumerate(data_loader):
        X, y, w, a = batch_data
        X, y, w, a = X.to(device), y.to(device), w.to(device), a.to(device)

        optimizer.zero_grad()  

        output = model(X, a)
        loss = F.binary_cross_entropy_with_logits(
            input=output,
            target=y,
            weight=w,
        )

        loss.backward()
        optimizer.step()

def validate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
) -> float:
    """Performs one epoch of model validation.

    Args:
        model (nn.Module): A model that takes a batch of chromosome segments (X) and their accessibility (a). 
        data_loader (DataLoader): A DataLoader that returns batches of chromosome segments (X), transcription factor binding indicator (y), segment weight (w), accessibility (a).
        device (str): The device on which the PyTorch Tensors will be allocated.

    Returns:
        (float): Weighted classification accuracy.
    """
    model.eval()

    weight_total = 0.
    weight_correct = 0.
    with torch.no_grad():
        for batch_data in data_loader:
            X, y, w, a = batch_data
            X, y, w, a = X.to(device), y.to(device), w.to(device), a.to(device)

            output = model(X, a)

            y_predicted = F.sigmoid(input=output).round()

            weight_total += torch.sum(w)
            weight_correct += torch.sum(w * (torch.eq(input=y, other=y_predicted)))

    accuracy = weight_correct / weight_total
    return accuracy.item()

class Objective_Base():
    def __init__(
        self, 
        data_dir_train: str, 
        data_dir_validation: str,
        batch_size_train: int,
        batch_size_validation: int,    
    ):
        self.dataloader_train = get_dataloader(
            data_dir=data_dir_train,
            batch_size=batch_size_train,
        )
        self.dataloader_validation = get_dataloader(
            data_dir=data_dir_validation,
            batch_size=batch_size_validation,
        )
        self.best_model = None
        self._model = None

    def perform_training(
        self, 
        model: nn.Module, 
        device: str, 
        optimizer: optim.Optimizer, 
        num_epochs: int, 
        trial: Trial
    ) -> float:
        """Performs epochs of model training and validation.

        Args:
            model (nn.Module): PyTorch model.
            device (str): Device on which to allocate Tensors.
            optimizer (optim.Optimizer): Optimizer to update model parameters.
            num_epochs (int): Number of epochs.
            trial (Trial): Optuna Trial for hyperparameter tuning.

        Raises:
            optuna.exceptions.TrialPruned: Trial pruned by Study.

        Returns:
            float: Last epoch validation accuracy.
        """
        for epoch in range(num_epochs):
            train_epoch(
                model=model,
                device=device,
                optimizer=optimizer,
                data_loader=self.dataloader_train
            )

            accuracy = validate_epoch(
                model=model,
                data_loader=self.dataloader_validation,
                device=device,
            )

            trial.report(value=accuracy, step=epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        self._model = model

        return accuracy

    def callback(
        self, 
        study: optuna.study.Study, 
        trial: optuna.trial.FrozenTrial
    ):
        """Updates best model state.

        Args:
            study (optuna.study.Study): Hyperparameter tuning Study.
            trial (optuna.trial.FrozenTrial): One Trial of tuning.
        """
        if study.best_trial.number == trial.number:
            self.best_model = self._model

class Objective_MLP(Objective_Base):
    def __init__(
        self, 
        data_dir_train: str, 
        data_dir_validation: str,
    ) -> None:
        super().__init__(
            data_dir_train=data_dir_train,
            data_dir_validation=data_dir_validation,
            batch_size_train=256,
            batch_size_validation=1024,
        )

    def __call__(self, trial: Trial) -> float:    
        num_epochs = trial.suggest_int(name="epochs", low=10, high=30, step=10)
        hidden_layer_size = trial.suggest_categorical(name="MLP_hidden_layer_size", choices=[16, 32, 64, 128,])
        lr = trial.suggest_float(name="lr", low=1e-4, high=1e-1, log=True)

        model = MLP(hidden_layer_size=hidden_layer_size)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.to(device)

        optimizer = optim.Adam(params=model.parameters(), lr=lr)

        return self.perform_training(
            model=model, 
            device=device, 
            optimizer=optimizer, 
            num_epochs=num_epochs, 
            trial=trial
        )


class Objective_LSTM(Objective_Base):
    def __init__(
        self,
        data_dir_train: str,
        data_dir_validation: str,
    ) -> None:
        super().__init__(
            data_dir_train=data_dir_train,
            data_dir_validation=data_dir_validation,
            batch_size_train=256,
            batch_size_validation=1024,
        )

    def __call__(self, trial: Trial) -> float:
        """select how many epochs you want to use and to choose the hidden dimension."""
        num_epochs = trial.suggest_int(name="epochs", low=10, high=30, step=10)
        lstm_hidden_layer_size = trial.suggest_categorical(name="LSTM_hidden_layer_size", choices=[16, 32, 64, 128,])
        mlp_hidden_layer_size = trial.suggest_categorical(name="MLP_hidden_layer_size", choices=[16, 32, 64])
        lr = trial.suggest_float(name="lr", low=1e-4, high=1e-1, log=True)

        model = LSTM(
            lstm_hidden_layer_size=lstm_hidden_layer_size,
            mlp_hidden_layer_size=mlp_hidden_layer_size,
        )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.to(device)

        optimizer = optim.Adam(params=model.parameters(), lr=lr)

        return self.perform_training(
            model=model, 
            device=device, 
            optimizer=optimizer, 
            num_epochs=num_epochs, 
            trial=trial
        )


class Objective_CNN(Objective_Base):
    def __init__(self, data_dir_train: str, data_dir_validation: str):
        super().__init__(
            data_dir_train=data_dir_train, 
            data_dir_validation=data_dir_validation, 
            batch_size_train=256, 
            batch_size_validation=1024,
        )
    
    def __call__(self, trial: Trial):
        num_epochs = trial.suggest_int(name="epochs", low=10, high=30, step=10)
        lr = trial.suggest_float(name="lr", low=1e-4, high=1e-1, log=True)

        conv_layer_1_num_channels = trial.suggest_categorical(name="conv_layer_1_num_channels", choices=[4, 8, 16,])
        conv_layer_1_kernel_size = trial.suggest_categorical(name="conv_layer_1_kernel_size", choices=[3, 5, 9,])

        conv_layer_2_num_channels = trial.suggest_categorical(name="conv_layer_2_num_channels", choices=[4, 8, 16,])
        conv_layer_2_kernel_size = trial.suggest_categorical(name="conv_layer_2_kernel_size", choices=[3, 5, 9,])

        mlp_hidden_layer_size = trial.suggest_categorical(name="MLP_hidden_layer_size", choices=[16, 32, 64,])

        model = CNN(
            conv_layer_1_num_channels=conv_layer_1_num_channels,
            conv_layer_1_kernel_size=conv_layer_1_kernel_size,
            max_pool_layer_1_kernel_size=3,
            conv_layer_2_num_channels=conv_layer_2_num_channels,
            conv_layer_2_kernel_size=conv_layer_2_kernel_size,
            max_pool_layer_2_kernel_size=3,
            mlp_hidden_layer_size=mlp_hidden_layer_size,
        )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.to(device)

        optimizer = optim.Adam(params=model.parameters(), lr=lr)

        return self.perform_training(
            model=model, 
            device=device, 
            optimizer=optimizer, 
            num_epochs=num_epochs, 
            trial=trial
        )
