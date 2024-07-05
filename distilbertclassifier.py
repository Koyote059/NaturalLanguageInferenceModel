from torch import nn
from torch.nn import Sequential
from transformers import DistilBertForSequenceClassification


class DistilBertClassifier(DistilBertForSequenceClassification):

    def __init__(self, config):
        super(DistilBertClassifier, self).__init__(config)
        self.classifier = Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(config.seq_classif_dropout),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.num_labels),
            nn.LogSoftmax(dim=-1)
        )
