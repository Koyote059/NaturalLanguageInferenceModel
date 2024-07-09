import json
import logging
import os
import random

import nltk
import torch
from datasets import load_dataset
from nltk import pos_tag
from torch.utils.data import Dataset

from augmentation import AugmentationPipeline, RandomHypothesisSubstitution, HypernymySubstitution, \
    VerbSynonymSubstitution, VerbNegation, PremiseSummarization
from tokenizer import NLITokenizer, NLIDistilBertTokenizer

_dataset = load_dataset("tommasobonomo/sem_augmented_fever_nli")


class NLIaFeverDataset(Dataset):
    """
    A dataset based on 'tommasobonomo/sem_augmented_fever_nli' ( an augmentation of Fever ) specialized for training
    a model for Natural Language Inference.
    """

    def __init__(self, split: str, tokenizer: NLITokenizer = NLIDistilBertTokenizer()):
        """
        :param split: possible values: train, test, valid.
        """
        if split not in ['train', 'test', 'validation']:
            raise ValueError(f'Split {split} is not valid. Possible values: train, test, validation.')
        self.split = split
        self.dataset = _dataset[split]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # TODO add cls token? I think I don't need to
        record = self.dataset[idx]
        input_tensor, attention_mask = self.tokenizer.tokenize(record['premise'], record['hypothesis'])
        label = [label_id for label_id, label_name in self.get_labels().items() if
                 label_name.lower() == record['label'].lower()][0]
        label_tensor = torch.zeros(3)
        label_tensor[label] = 1
        return {
            'input_ids': input_tensor,
            'attention_mask': attention_mask,
            'labels': label_tensor,
        }

    @staticmethod
    def get_labels():
        return {
            0: 'ENTAILMENT',
            1: 'CONTRADICTION',
            2: 'NEUTRAL'
        }


class MaskedAFeverDataset(Dataset):
    """
    A dataset based on 'tommasobonomo/sem_augmented_fever_nli' (an augmentation of Fever) specialized for training
    a model for Natural Language Inference with Masking.
    """

    def __init__(self, split: str, tokenizer: NLITokenizer = NLIDistilBertTokenizer()):
        """
        :param split: possible values: train, test, validation.
        :param tokenizer: tokenizer to use for tokenization and masking
        """
        if split not in ['train', 'test', 'validation']:
            raise ValueError(f'Split {split} is not valid. Possible values: train, test, validation.')
        self.split = split
        self.dataset = _dataset[split]
        self.tokenizer = tokenizer
        logging.getLogger('nltk').setLevel(logging.ERROR)

        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('punkt', quiet=True)

    def __len__(self):
        return len(self.dataset)

    @property
    def mask_token_id(self):
        return self.tokenizer.mask_token_id

    @staticmethod
    def _get_word_to_tag(words):
        allowed_tags = ('NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBP', 'VBN', 'VBZ')
        tagged = pos_tag(words)
        allowed_words = [word[0] for word in tagged if word[1] in allowed_tags]
        return random.choice(allowed_words) if allowed_words else random.choice(words)

    def mask_random_word(self, input_tensor, hypothesis: str):
        word_ids = {self.tokenizer.decode(input_id): input_id for input_id in input_tensor}
        tokens, _ = self.tokenizer.tokenize(hypothesis)
        words = [self.tokenizer.decode(token) for token in tokens]
        word_to_mask = self._get_word_to_tag(words)
        masked_input_tensor = input_tensor.clone()
        for i in range(masked_input_tensor.shape[0]):
            if masked_input_tensor[i] == word_ids[word_to_mask]:
                masked_input_tensor[i] = self.mask_token_id
                break
        return masked_input_tensor

    def __getitem__(self, idx):
        record = self.dataset[idx]
        input_tensor, attention_mask = self.tokenizer.tokenize(record['premise'], record['hypothesis'])
        masked_input_tensor = self.mask_random_word(input_tensor, record['hypothesis'])
        return {
            "input_ids": masked_input_tensor,
            "attention_mask": attention_mask,
            "labels": input_tensor,
        }


class AugmentedNLIaFeverTestDataset(NLIaFeverDataset):

    def __init__(self, tokenizer: NLITokenizer = NLIDistilBertTokenizer()):
        super(AugmentedNLIaFeverTestDataset, self).__init__('test', tokenizer)
        self.dataset = load_dataset("iperbole/adversarial_fever_nli")['test']


ADVERSARIAL_DATASETS = {
    'test': 'augmented_test.jsonl',
    'train': 'augmented_train.jsonl',
    'validation': 'augmented_valid.jsonl',
}


class AdversarialNLIaFeverDataset(NLIaFeverDataset):

    def __init__(self, directory: str, split: str, tokenizer: NLITokenizer = NLIDistilBertTokenizer()):
        super(AdversarialNLIaFeverDataset, self).__init__(split, tokenizer)
        file_path = os.path.join(directory, ADVERSARIAL_DATASETS[self.split])
        self.dataset = []
        with open(file_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                self.dataset.append(record)


class AdversarialMaskedAFeverDataset(MaskedAFeverDataset):

    def __init__(self, directory: str, split: str, tokenizer: NLITokenizer = NLIDistilBertTokenizer()):
        super(MaskedAFeverDataset, self).__init__(split, tokenizer)
        file_path = os.path.join(directory, ADVERSARIAL_DATASETS[self.split])
        self.dataset = []
        with open(file_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                self.dataset.append(record)


def augment_datasets(directory: str, verbose=1):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for split in _dataset:

        pipeline = AugmentationPipeline(priority_methods={
            VerbNegation(): 1,
            HypernymySubstitution(): 0.5,
            VerbSynonymSubstitution(): 0.3,
            # AdverbInversion(): 1,
            # DateSubstitution(): 1.5,
        }, secondary_methods={
            RandomHypothesisSubstitution(_dataset[split]): 1,
            PremiseSummarization(): 0.5,
        })

        file_path = os.path.join(directory, ADVERSARIAL_DATASETS[split])
        if verbose:
            print("Creating file: {}".format(file_path))
        with open(file_path, 'w') as file:
            for i, record in enumerate(_dataset[split]):
                if verbose > 1:
                    print(f"Augmenting record {i + 1} / {len(_dataset[split])}")
                augmented_record = pipeline(record)
                if augmented_record is None:
                    continue
                file.write(json.dumps(augmented_record) + '\n')
