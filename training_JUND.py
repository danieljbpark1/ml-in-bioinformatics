import optuna
from optuna.trial import Trial, TrialState
import os
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .datasets import JUND_Dataset
from .models import MLP

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

class Objective():
    def __init__(
        self, 
        data_dir_train: str, 
        data_dir_validation: str,
    ) -> None:
        self.dataloader_train = get_dataloader(
            data_dir=data_dir_train,
            batch_size=256,
        )
        self.dataloader_validation = get_dataloader(
            data_dir=data_dir_validation,
            batch_size=1024,
        )
        self.best_model = None
        self._model = None

    def __call__(self, trial: Trial):    
        num_epochs = trial.suggest_int(name="epochs", low=10, high=30, step=10)
        hidden_layer_size = trial.suggest_categorical(name="MLP_hidden_layer_size", choices=[16, 32, 64, 128,])
        lr = trial.suggest_float(name="lr", low=1e-4, high=1e-1, log=True)

        model = MLP(hidden_layer_size=hidden_layer_size)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.to(device)

        optimizer = optim.Adam(params=model.parameters(), lr=lr)

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
    
    def callback(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if study.best_trial.number == trial.number:
            self.best_model = self._model


if __name__ == "__main__":
    DATA_DIR_TRAIN = os.getenv("DATA_DIR_TRAIN")
    DATA_DIR_VALIDATION = os.getenv("DATA_DIR_VALIDATION")
    DATA_DIR_TEST = os.getenv("DATA_DIR_TEST")
    
    objective = Objective(
        data_dir_train=DATA_DIR_TRAIN,
        data_dir_validation=DATA_DIR_VALIDATION,
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(func=objective, n_trials=20, timeout=1800, callbacks=[objective.callback])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    dataloader_test = get_dataloader(data_dir=DATA_DIR_TEST, batch_size=1024)
    accuracy_test = validate_epoch(
        model=objective.best_model,
        data_loader=dataloader_test,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    print("Accuracy of best model after hyperparameter tuning on test dataset: ", accuracy_test)
