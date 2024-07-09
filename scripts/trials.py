from transformers import DistilBertConfig, DistilBertForMaskedLM

from classificationtrainer import TrainingSettings, LogSettings
from dataset import AdversarialNLIaFeverDataset, MaskedAFeverDataset, \
    NLIaFeverDataset
from distilbertclassifier import DistilBertClassifier
from training import MultiDSDistilBertForSequenceClassificationTrainer, DistilbertForMaskedLMTrainer
from transformers import DistilBertConfig, DistilBertForMaskedLM

from classificationtrainer import TrainingSettings, LogSettings
from dataset import AdversarialNLIaFeverDataset, MaskedAFeverDataset, \
    NLIaFeverDataset
from distilbertclassifier import DistilBertClassifier
from training import MultiDSDistilBertForSequenceClassificationTrainer, DistilbertForMaskedLMTrainer

ADVERSARIAL_DATASET_DIRECTORY = './report_augmented_dataset'

# RandomHypothesis, Hypernimy, VerbSynonym, AdverbInversion, DateSubstitution,VerbNegation
cases = [(True, True, True, True, True, True),  # 0
         (True, False, True, True, True, True),  # 1
         (False, True, True, True, True, True),  # 2
         (True, True, False, True, True, True),  # 3
         (True, True, True, False, True, True),  # 4
         (True, True, True, True, False, True),  # 5
         (True, True, True, True, True, False),  # 6
         ]


# Actual best is 4_5 now ( let's try final at home! )
# Case: 5
# Best model on adversarial: .\NLI_distilberts\5\models\NLIDistilbert_trained_on_adversarial_dataset_10.pt
# Best masked on adversarial: .\NLI_distilberts_masked_classification\5\models\MaskedDistilbertForSequenceClassification_30.pt


def train_masked_distilbert(directory):
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
        confusion_matrix=False,
        model_name="MaskedDistilbert"
    )
    train_settings = TrainingSettings(
        epochs=45,
        batch_size=16,
        save_each_epochs=15,
        lr=1e-4,
    )

    trainer = DistilbertForMaskedLMTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        valid_dataset=valid_dataset,
        directory=directory,
        log_settings=log_settings,
        training_settings=train_settings,
    )
    trainer.train(verbose=0)
    return directory


def train_masked_distilbert_for_classification_on_adversarial(pretrained_path: str, directory):
    train_dataset = AdversarialNLIaFeverDataset(ADVERSARIAL_DATASET_DIRECTORY, 'test')
    test_dataset = AdversarialNLIaFeverDataset(ADVERSARIAL_DATASET_DIRECTORY, 'train')
    valid_dataset = AdversarialNLIaFeverDataset(ADVERSARIAL_DATASET_DIRECTORY, 'validation')
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased')

    config.num_labels = 3
    model = DistilBertClassifier.from_pretrained(pretrained_path, config=config)
    # Freezing parameters and stopping when reached the classifier
    for name, params in model.named_parameters():
        if 'classifier' in name:
            break
        params.requires_grad = False

    log_settings = LogSettings(
        model_name='MaskedDistilbertForSequenceClassification'
    )

    train_settings = TrainingSettings(
        epochs=30,
        batch_size=16,
        save_each_epochs=10,
        lr=1e-4,
    )
    trainer = MultiDSDistilBertForSequenceClassificationTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        valid_dataset=valid_dataset,
        directory=directory,
        training_settings=train_settings,
        log_settings=log_settings
    )
    trainer.train(verbose=0)
    return directory


def train_masked_distilbert_for_classification_on_original(pretrained_path: str, directory):
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
        model_name='MaskedDistilbertForSequenceClassification'
    )

    train_settings = TrainingSettings(
        epochs=30,
        batch_size=16,
        save_each_epochs=10,
        lr=1e-4,
    )
    trainer = MultiDSDistilBertForSequenceClassificationTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        valid_dataset=valid_dataset,
        directory=directory,
        training_settings=train_settings,
        log_settings=log_settings
    )
    trainer.train(verbose=0)
    return directory

if __name__ == '__main__':
    train_masked_distilbert('./MASKED_distilberts')
    best_masked = r'C:\Users\feder\PycharmProjects\MNLP_Homework_1B\MASKED_distilberts\models\MaskedDistilbert_30.pt'
    train_masked_distilbert_for_classification_on_original(best_masked,r"./NLI_masked_on_original")
    train_masked_distilbert_for_classification_on_adversarial(best_masked, r"./NLI_masked_on_adversarial")