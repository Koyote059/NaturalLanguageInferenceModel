from abc import ABC

from transformers import RobertaTokenizer, DistilBertTokenizer
from transformers.tokenization_utils_base import TruncationStrategy


class NLITokenizer(ABC):
    def tokenize(self, premise: str, hypothesis: str = None):
        pass

    def decode(self, input_id):
        pass

    @property
    def mask_token_id(self):
        return None

    @property
    def mask_token(self):
        return None

    def get_pad_id(self):
        pass


class NLIDistilBertTokenizer(NLITokenizer):
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize(self, premise: str, hypothesis: str = None):
        tokens = self.tokenizer(premise, hypothesis, return_tensors='pt', padding=True,
                                truncation=TruncationStrategy.LONGEST_FIRST)
        return tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0)

    def decode(self, token):
        return self.tokenizer.decode(token)

    @property
    def mask_token_id(self):
        return self.tokenizer.mask_token_id

    @property
    def mask_token(self):
        return self.tokenizer.mask_token

    def get_pad_id(self):
        return self.tokenizer.pad_token_id
