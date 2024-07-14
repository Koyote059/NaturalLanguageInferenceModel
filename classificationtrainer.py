import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List
import seaborn as sns

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from torch import nn
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from transformers import PreTrainedModel


@dataclass
class TrainingSettings:
    """
        A class to encapsulate all settings related to model training.

        Attributes:
            split (Tuple[float, float, float]): Dataset split ratios for train, validation, and test sets.
                Default is (0.8, 0.1, 0.1).
            epochs (int): Number of training epochs. Default is 32.
            batch_size (int): Number of samples per batch. Default is 32.
            lr (float): Learning rate for the optimizer. Default is 0.0001.
            loss_fn (nn.Module): Loss function to use for training.
                Default is nn.CrossEntropyLoss().
            optimizer (str): Name of the optimizer to use. Options are "Adamw", "sgd", or "rmsprop".
                Default is "Adamw".
            save_each_epochs (Optional[int]): If set, saves the model every specified number of epochs.
                If None, only saves at the end of training. Default is None.
            model_name (Optional[str]): Name to use when saving the model.
                If None, uses the model's class name. Default is None.
            collate_fn (Optional[object]): Function to merge a list of samples to form a mini-batch.
                If None, uses default_collate. Default is None.
        """
    split: Tuple[float, float, float] = field(default=(0.8, 0.1, 0.1))
    epochs: int = field(default=32)
    batch_size: int = field(default=32)
    lr: float = field(default=1e-5)
    loss_fn: nn.Module = field(default_factory=nn.CrossEntropyLoss)
    optimizer: str = field(default="Adamw")
    save_each_epochs: Optional[int] = field(default=None)
    collate_fn: Optional[object] = field(default=None)
    class_names: List[str] = field(default=None)


@dataclass
class LogSettings:
    model_name: Optional[str] = None
    train_losses: bool = field(default=True)
    val_losses: bool = field(default=True)
    test_accuracy: bool = field(default=True)
    val_accuracies: bool = field(default=True)
    confusion_matrix: bool = field(default=True)
    f1_score: bool = field(default=True)
    precision: bool = field(default=True)
    recall: bool = field(default=True)
    optimizer: bool = field(default=True)
    loss_fn: bool = field(default=True)
    lr: bool = field(default=True)
    epochs: bool = field(default=True)
    batch_size: bool = field(default=True)
    train_dataset_size: bool = field(default=True)
    val_dataset_size: bool = field(default=True)
    test_dataset_size: bool = field(default=True)
    times: bool = field(default=True)
    criterion: bool = field(default=True)


class ClassificationTrainer(ABC):
    """
    Generic class to train models.
    """

    def __init__(self, model: PreTrainedModel, dataset: Dataset, directory: str,
                 training_settings: TrainingSettings = TrainingSettings(), log_settings: LogSettings = LogSettings()):
        """
        :param model: the model to train.
        :param dataset: the dataset from which to train.
        :param directory: the directory to save checkpoints and logs.
        :param training_settings: TrainingSettings object containing training parameters.
        :param log_settings: LogSettings object containing parameters for training logs.
        """
        self.model = model
        self.dataset = dataset
        self.settings = training_settings
        self.log_settings = log_settings
        self.model_name = log_settings.model_name if log_settings.model_name is not None else model.__class__.__name__
        self.class_names = training_settings.class_names
        if training_settings.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=training_settings.lr)
        elif training_settings.optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=training_settings.lr)
        elif training_settings.optimizer.lower() == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=training_settings.lr)
        else:
            raise ValueError("Optimizer must be either adam or sgd or rmsprop")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.directory = directory
        self.models_directory = os.path.join(directory, "models")
        self.checkpoints_directory = os.path.join(directory, "checkpoints")
        os.makedirs(self.models_directory, exist_ok=True)
        os.makedirs(self.checkpoints_directory, exist_ok=True)
        self.train_dataset, self.valid_dataset, self.test_dataset = random_split(self.dataset, self.settings.split)

    def _evaluate_model(self, dataloader: DataLoader) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        """
        A function for evaluating the model.
        :param dataloader: the dataloader from which to evaluate the model.
        :return: a tuple with the following structure:
            - The loss score
            - The predicted values
            - The true values
        """
        loss_score = 0.0
        predictions = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                loss, y_pred, y_true = self.train_batch(batch, self.model, self.device)
                loss_score += loss.item()
                predictions.extend(y_pred)
                labels.extend(y_true)

        loss_score = loss_score / len(dataloader)
        self.model.train()
        return loss_score, predictions, labels

    def train(self, verbose: int = 1):
        if verbose and self.device == torch.device("cpu"):
            print("WARNING: You have chosen to train on CPU only!")
        self.model.to(self.device)
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.settings.batch_size, shuffle=True,
                                      collate_fn=self.settings.collate_fn)
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.settings.batch_size, shuffle=False,
                                      collate_fn=self.settings.collate_fn)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.settings.batch_size, shuffle=False,
                                     collate_fn=self.settings.collate_fn)
        if verbose:
            print("Training settings: ")
            print(f"\tDevice being used: {self.device}")
            print(f"\tBatch size: {self.settings.batch_size}")
            print(f"\tLearning rate: {self.settings.lr}")
            print(f"\tNum epochs: {self.settings.epochs}")
            print(f"\tTraining dataset size: {len(self.train_dataset)}")
            print(f"\tValidation dataset size: {len(self.valid_dataset)}")
            print(f"\tTest dataset size: {len(self.test_dataset)}")
            print(f"\tBatches per training epoch: {len(train_dataloader)}")
            print(f"\tOptimizer: {self.optimizer.__class__.__name__}")
            print(f"\tLoss function: {self.settings.loss_fn.__class__.__name__}")
            if self.settings.save_each_epochs:
                print(f"\tSaving model every {self.settings.save_each_epochs} epochs")
            print(f"\nSaving model in: {self.directory}")
            print(f"Model name: {self.model_name}")
            print("\nStarting training...")

        train_losses = []
        valid_losses = []
        valid_accuracies = []
        times = []
        self.model.train()
        for epoch in range(1, self.settings.epochs + 1):
            if verbose:
                print(f"Epoch: {epoch}")
            start_time = time.time()
            train_loss = 0.0
            for i, batch in enumerate(train_dataloader):
                if verbose > 2:
                    print(f"\tBatch: {i + 1}/{len(train_dataloader)}")
                self.optimizer.zero_grad()
                loss, _, _ = self.train_batch(batch, self.model, self.device)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            if verbose > 1:
                print("Training finished. Evaluating epoch...")
            train_loss = train_loss / len(train_dataloader)
            valid_loss, y_pred, y_true = self._evaluate_model(valid_dataloader)
            valid_accuracy = accuracy_score(y_true, y_pred)
            tot_time = time.time() - start_time
            times.append(tot_time)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)
            if verbose:
                print(
                    f"\tEpoch: {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, "
                    f"Valid Accuracy: {valid_accuracy:.4f}, Time: {tot_time:.2f}s")

            if ((self.settings.save_each_epochs and epoch % self.settings.save_each_epochs == 0)
                    or epoch == self.settings.epochs):
                if verbose > 1:
                    print("Testing model...")
                loss, y_pred, y_true = self._evaluate_model(test_dataloader)
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0.0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0.0)
                f1 = f1_score(y_true, y_pred, average='weighted')
                conf_matrix = confusion_matrix(y_true, y_pred)
                conf_matrix = [row.tolist() for row in conf_matrix]
                if verbose > 1:
                    print("Test metrics:")
                    print(f"\tAccuracy: {accuracy:.4f}")
                    print(f"\tPrecision: {precision:.4f}")
                    print(f"\tRecall: {recall:.4f}")
                    print(f"\tF1: {f1:.4f}")
                    print("\nSaving model...")

                meta_data = {
                    "epochs": epoch if self.log_settings.epochs else None,
                    "batch_size": self.settings.batch_size if self.log_settings.batch_size is not None else None,
                    'training_dataset_size': len(self.train_dataset) if self.log_settings.train_dataset_size else None,
                    'valid_dataset_size': len(self.valid_dataset) if self.log_settings.val_dataset_size else None,
                    'test_dataset_size': len(self.test_dataset) if self.log_settings.test_dataset_size else None,
                    "train_losses": train_losses if self.log_settings.train_losses else None,
                    "valid_losses": valid_losses if self.log_settings.val_losses else None,
                    "valid_accuracies": valid_accuracies if self.log_settings.val_accuracies else None,
                    "times": times if self.log_settings.times else None,
                    'accuracy': accuracy if self.log_settings.test_accuracy else None,
                    'precision': precision if self.log_settings.precision else None,
                    'recall': recall if self.log_settings.recall else None,
                    'f1': f1 if self.log_settings.f1_score else None,
                    'conf_matrix': conf_matrix if self.log_settings.confusion_matrix else None,
                    'criterion': self.settings.loss_fn.__class__.__name__ if self.log_settings.criterion else None,
                    'optimizer': self.optimizer.__class__.__name__ if self.log_settings.optimizer else None,
                    'lr': self.settings.lr if self.log_settings.lr else None,
                }
                train_losses = []
                valid_losses = []
                valid_accuracies = []
                times = []
                self._save(meta_data, epoch)

    def test(self, verbose=1):
        self.model.to(self.device)
        test_dataloader = DataLoader(self.dataset, batch_size=self.settings.batch_size, shuffle=False,
                                     collate_fn=self.settings.collate_fn)
        if verbose:
            print("Testing settings: ")
            print("\tData")
            print("\tBatch size: ", self.settings.batch_size)
            print("\tDataset size: ", len(self.train_dataset))
        loss, y_pred, y_true = self._evaluate_model(test_dataloader)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0.0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0.0)
        f1 = f1_score(y_true, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_true, y_pred)
        if verbose:
            print("Test metrics:")
            print(f"\tAccuracy: {accuracy:.4f}")
            print(f"\tPrecision: {precision:.4f}")
            print(f"\tRecall: {recall:.4f}")
            print(f"\tF1: {f1:.4f}")

        if verbose > 1:
            conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
            plt.figure(figsize=(10, 8))
            if self.class_names:
                sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=self.class_names,
                            yticklabels=self.class_names)
            else:
                sns.heatmap(conf_matrix_normalized, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.show()
        conf_matrix = [row.tolist() for row in conf_matrix]

        return accuracy, precision, recall, f1, conf_matrix

    @abstractmethod
    def train_batch(self, batch, model, device) -> Tuple[_WeightedLoss, list[np.ndarray], list[np.ndarray]]:
        """
        Function to extend to make the class model-specific.
        It trains the model for a single batch.
        It doesn't update the training information.
        :param batch: the batch to train from.
        :return: a tuple with the following structure:
            - Model loss
            - Predicted values
            - True values
        """
        pass

    def _save(self, log: Dict, epochs: int):
        """
        It saves the model and the log from the training time in the "directory".
        Each file containing the model will have the name:
            - MODEL_NAME-EPOCHS.pt
        While the log files:
            - MODEL_NAME-EPOCHS.log
        :param log: the log to save.
        """
        save_name = f"{self.model_name}_{epochs}"
        model_file_path = os.path.join(self.models_directory, f"{save_name}.pt")
        model_meta_path = os.path.join(self.checkpoints_directory, f"{save_name}.cp")
        self.model.save_pretrained(model_file_path)
        with open(model_meta_path, 'w') as f:
            json.dump(log, f)
