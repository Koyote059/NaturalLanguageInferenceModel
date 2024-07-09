from typing import Tuple, Dict, List

import numpy as np
import torch
from torch.nn.modules.loss import _WeightedLoss
from transformers import DistilBertForMaskedLM

from classificationtrainer import ClassificationTrainer, LogSettings, TrainingSettings
from dataset import NLIaFeverDataset, MaskedAFeverDataset


class SequencePadder:

    def __init__(self, pad_token):
        self.pad_token = pad_token

    def __call__(self, batch):
        return self.collate_fn(batch)

    def collate_fn(self, batch):
        # Separate inputs and labels
        inputs = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        # Pad the input sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=self.pad_token)
        label_ids = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.pad_token)

        # Create attention masks
        attention_mask = (input_ids != self.pad_token).long()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids
        }


class DistilBertForSequenceClassificationTrainer(ClassificationTrainer):

    def __init__(self, model: DistilBertForMaskedLM, dataset: NLIaFeverDataset, directory: str,
                 training_settings: TrainingSettings = TrainingSettings(), log_settings: LogSettings = LogSettings()):
        super().__init__(model=model, dataset=dataset, directory=directory, training_settings=training_settings,
                         log_settings=log_settings)
        training_settings.collate_fn = SequencePadder(dataset.tokenizer.get_pad_id())

    def train_batch(self, batch: Dict, model, device) -> Tuple[_WeightedLoss, list[np.ndarray], list[np.ndarray]]:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        y_pred = torch.argmax(outputs.logits, dim=1).to('cpu').numpy()
        y_true = torch.argmax(labels, dim=1).to('cpu').numpy()
        return loss, y_pred, y_true


class MultiDSDistilBertForSequenceClassificationTrainer(DistilBertForSequenceClassificationTrainer):

    def __init__(self, model: DistilBertForMaskedLM, train_dataset: NLIaFeverDataset,
                 test_dataset: NLIaFeverDataset, valid_dataset: NLIaFeverDataset, directory: str,
                 training_settings: TrainingSettings = TrainingSettings(), log_settings: LogSettings = LogSettings()):
        super().__init__(model=model, dataset=train_dataset, directory=directory, training_settings=training_settings,
                         log_settings=log_settings)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.valid_dataset = valid_dataset

    def train_batch(self, batch: Dict, model, device) -> Tuple[_WeightedLoss, list[np.ndarray], list[np.ndarray]]:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        y_pred = torch.argmax(outputs.logits, dim=1).to('cpu').numpy()
        y_true = torch.argmax(labels, dim=1).to('cpu').numpy()
        return loss, y_pred, y_true


class DistilbertForMaskedLMTrainer(ClassificationTrainer):

    def __init__(self, model: DistilBertForMaskedLM, train_dataset: MaskedAFeverDataset,
                 test_dataset: MaskedAFeverDataset, valid_dataset: MaskedAFeverDataset, directory: str,
                 training_settings: TrainingSettings = TrainingSettings(), log_settings: LogSettings = LogSettings()):
        super().__init__(model=model, dataset=train_dataset, directory=directory, training_settings=training_settings,
                         log_settings=log_settings)
        training_settings.collate_fn = SequencePadder(train_dataset.tokenizer.get_pad_id())
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.valid_dataset = valid_dataset

    def train_batch(self, batch: Dict, model, device) -> Tuple[_WeightedLoss, List[List[int]], List[List[int]]]:
        # Sposta tutti gli elementi del batch sul device specificato
        batch = {k: v.to(device) for k, v in batch.items()}

        # Esegui il forward pass del modello
        outputs = model(**batch)
        loss = outputs.loss

        # Ottieni le etichette e le previsioni
        labels = batch['labels']
        logits = outputs.logits

        mask_token_id = self.train_dataset.mask_token_id

        # Inizializza liste per y_pred e y_true per ogni frase
        y_pred_batch = []
        y_true_batch = []

        # Itera su ogni frase nel batch
        for i in range(batch['input_ids'].shape[0]):
            # Trova le posizioni dei token mascherati per questa frase
            masked_positions = (batch['input_ids'][i] == mask_token_id).nonzero(as_tuple=True)[0]

            # Ottieni le previsioni solo per i token mascherati di questa frase
            masked_predictions = logits[i, masked_positions]
            y_pred = torch.argmax(masked_predictions, dim=-1).cpu().numpy()

            # Ottieni le vere etichette per i token mascherati di questa frase
            y_true = labels[i, masked_positions].cpu().numpy()

            y_pred_batch.append(y_pred.tolist())
            y_true_batch.append(y_true.tolist())

        return loss, y_pred_batch, y_true_batch

