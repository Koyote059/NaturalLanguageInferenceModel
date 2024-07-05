import json
import os
from typing import Tuple, Dict, List

import numpy as np
import torch
from torch.nn.modules.loss import _WeightedLoss
from transformers import DistilBertForMaskedLM, \
    DistilBertForSequenceClassification, DistilBertConfig, DistilBertModel

from classificationtrainer import ClassificationTrainer, LogSettings, TrainingSettings
from dataset import NLIaFeverDataset, MaskedAFeverDataset
from distilbertclassifier import DistilBertClassifier


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


class RobertaForSequenceClassificationTrainer(ClassificationTrainer):

    def __init__(self, model: DistilBertForMaskedLM, train_dataset: NLIaFeverDataset,
                 test_dataset: NLIaFeverDataset, valid_dataset: NLIaFeverDataset, directory: str,
                 training_settings: TrainingSettings = TrainingSettings(), log_settings: LogSettings = LogSettings()):
        super().__init__(model=model, dataset=train_dataset, directory=directory, training_settings=training_settings,
                         log_settings=log_settings)
        training_settings.collate_fn = SequencePadder(train_dataset.tokenizer.get_pad_id())
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


class RobertaForMaskedLMTrainer(ClassificationTrainer):

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

    # TODO better prints
    #   Valid loss is too low!


def train_masked_roberta():
    train_dataset = MaskedAFeverDataset('test')
    test_dataset = MaskedAFeverDataset('train')
    valid_dataset = MaskedAFeverDataset('validation')
    model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    # Freezing parameters and stopping when reached 3rd layer
    for name, params in model.named_parameters():
        if 'layer.3' in name:
            break
        params.requires_grad = False

    log_settings = LogSettings(
        confusion_matrix=False
    )
    train_settings = TrainingSettings(
        epochs=240,
        batch_size=16,
        save_each_epochs=15,
        lr=5 * 1e-5,
    )

    trainer = RobertaForMaskedLMTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        valid_dataset=valid_dataset,
        directory='./masked_distilberts',
        log_settings=log_settings,
        training_settings=train_settings,
    )
    trainer.train(verbose=1)


def train_roberta_for_classifications():
    train_dataset = NLIaFeverDataset('test')
    test_dataset = NLIaFeverDataset('train')
    valid_dataset = NLIaFeverDataset('validation')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

    # Freezing parameters and stopping when reached 3rd layer
    for name, params in model.named_parameters():
        if 'layer.3' in name:
            break
        params.requires_grad = False

    train_settings = TrainingSettings(
        epochs=100,
        batch_size=16,
        save_each_epochs=5,
        lr=1e-6
    )

    trainer = RobertaForSequenceClassificationTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        valid_dataset=valid_dataset,
        directory='./NLI_distilberts',
        training_settings=train_settings,
    )
    trainer.train(verbose=1)


def train_masked_distilbert_for_classification(pretrained_path: str):
    train_dataset = NLIaFeverDataset('test')
    test_dataset = NLIaFeverDataset('train')
    valid_dataset = NLIaFeverDataset('validation')
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased')

    config.num_labels = 3
    model = DistilBertClassifier.from_pretrained(pretrained_path, config=config)
    # Freezing parameters and stopping when reached the classifier
    for name, params in model.named_parameters():
        if 'classifier' in name:
            break
        params.requires_grad = False

    log_settings = LogSettings(
        model_name='DistilBertClassifier_3'
    )

    train_settings = TrainingSettings(
        epochs=160,
        batch_size=16,
        save_each_epochs=10,
        lr=1e-3,
    )

    trainer = RobertaForSequenceClassificationTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        valid_dataset=valid_dataset,
        directory='./NLI_masked_distilberts',
        training_settings=train_settings,
        log_settings=log_settings
    )
    trainer.train(verbose=1)


def choose_model(checkpoint_dir: str, tolerance: float = 0.02):
    logs = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.cp'):
            with open(os.path.join(checkpoint_dir, file), 'rb') as f:
                log = json.load(f)
                logs.append(log)

    max_accuracy = max(logs, key=lambda x: x['accuracy'])['accuracy']
    # Take all the logs which have accuracy >= max_accuracy-tolerance
    best_logs = [log for log in logs if log['accuracy'] + tolerance >= max_accuracy]
    best_logs_overfit = {log["epochs"]: sum(log["valid_losses"]) for log in best_logs}
    epoch = min(best_logs_overfit, key=best_logs_overfit.get)
    print("Best model is at epoch: ", epoch)
    # Best one for NLI_distilberts at epoch 60 (?) Check again
    # Best one for Masked_Distilbert: 75  Check again
    # Best for Masked_Distilbert: DistilBert_classifier_2_80
    return epoch

# TODO check if training is good
# TODO create code for running the model

def main():
    # choose_model(checkpoint_dir='./masked_distilberts/checkpoints', tolerance=2) output: 75
    # epoch = choose_model(checkpoint_dir='./NLI_distilberts/checkpoints', tolerance=1) output: 55
    # train_masked_distilbert_for_classification(r'./NLI_masked_distilberts/models/DistilBertClassifier_30.pt')
    choose_model(checkpoint_dir=r"C:\Users\feder\PycharmProjects\MNLP_Homework_1B\NLI_distilberts\checkpoints", tolerance=0.02)

if __name__ == '__main__':
    main()
