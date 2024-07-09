import os
import sys

from transformers import DistilBertForSequenceClassification
from dataset import NLIaFeverDataset, AdversarialNLIaFeverDataset, AugmentedNLIaFeverTestDataset
from classificationtrainer import TrainingSettings, LogSettings
from training import MultiDSDistilBertForSequenceClassificationTrainer, DistilBertForSequenceClassificationTrainer

ADVERSARIAL_DATASET_DIRECTORY = './augmented_dataset'
MODELS = './NLI_distilberts'


def get_parameters():
    if len(sys.argv) == 3:
        command_type = sys.argv[1]
        data_type = sys.argv[2]
        if command_type in ['train', 'test'] and data_type in ['original', 'adversarial']:
            return command_type, data_type
    name = sys.argv[0]
    print(f"Command Error!. \nThe correct usage is: python3 {name} [train|test] --data [original|adversarial]")
    sys.exit(1)


def train_on_original_dataset():
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
        epochs=60,
        batch_size=16,
        lr=1e-6
    )

    log_settings = LogSettings(
        model_name="NLIDistilbert"
    )

    trainer = MultiDSDistilBertForSequenceClassificationTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        valid_dataset=valid_dataset,
        directory=MODELS,
        training_settings=train_settings,
        log_settings=log_settings
    )

    print("Training begun...")
    print("Saving results in: ./NLI_distilberts/")
    trainer.train(verbose=1)


def test_on_original_dataset():
    dataset = NLIaFeverDataset('test')
    file_path = os.path.join(MODELS, 'models', 'NLIDistilbert_trained_on_original_dataset_60.pt')
    print("Loading model: ", file_path)
    if not os.path.exists(file_path):
        print("Model not found! You must do the training first!")
        return
    model = DistilBertForSequenceClassification.from_pretrained(file_path, num_labels=3)
    labels = NLIaFeverDataset.get_labels()
    train_settings = TrainingSettings(
        batch_size=16,
        class_names=[labels[0],labels[1],labels[2]],
    )
    tester = DistilBertForSequenceClassificationTrainer(
        model=model,
        dataset=dataset,
        training_settings=train_settings,
        directory=""
    )
    print("Testing model...")
    tester.test(verbose=2)


def train_on_adversarial_dataset():
    train_dataset = AugmentedNLIaFeverTestDataset()
    test_dataset = AdversarialNLIaFeverDataset(split='train', directory=ADVERSARIAL_DATASET_DIRECTORY)
    valid_dataset = AdversarialNLIaFeverDataset(split='validation', directory=ADVERSARIAL_DATASET_DIRECTORY)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

    # Freezing parameters and stopping when reached 3rd layer
    for name, params in model.named_parameters():
        if 'layer.3' in name:
            break
        params.requires_grad = False

    train_settings = TrainingSettings(
        epochs=150,
        batch_size=16,
        lr=1e-5,
        save_each_epochs=15
    )

    log_settings = LogSettings(
        model_name="NLIDistilbert"
    )

    trainer = MultiDSDistilBertForSequenceClassificationTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        valid_dataset=valid_dataset,
        directory=MODELS,
        training_settings=train_settings,
        log_settings=log_settings
    )

    print("Training begun...")
    print("Saving results in: ./NLI_distilberts/")
    trainer.train(verbose=1)


def test_on_adversarial_dataset():
    dataset = AugmentedNLIaFeverTestDataset()
    file_path = os.path.join(MODELS, 'models', 'NLIDistilbert_' + "60.pt")
    if not os.path.exists(file_path):
        print("Model not found! You must do the training first!")
        return
    model = DistilBertForSequenceClassification.from_pretrained(file_path, num_labels=3)

    train_settings = TrainingSettings(
        batch_size=16,
    )
    tester = DistilBertForSequenceClassificationTrainer(
        model=model,
        dataset=dataset,
        training_settings=train_settings,
        directory=""
    )
    print("Testing model...")
    tester.test(verbose=1)


def main():
    command_type, data_type = get_parameters()
    if command_type == 'train' and data_type == 'original':
        train_on_original_dataset()
    elif command_type == 'train' and data_type == 'adversarial':
        train_on_adversarial_dataset()
    elif command_type == 'test' and data_type == 'original':
        test_on_original_dataset()
    elif command_type == 'test' and data_type == 'adversarial':
        test_on_adversarial_dataset()
    else:
        raise ValueError()


if __name__ == '__main__':
    sys.argv = [sys.argv[0], "test", "original"]
    main()
