import json
from collections import Counter

import torch
from matplotlib import pyplot as plt
from transformers import DistilBertForSequenceClassification

from classificationtrainer import TrainingSettings
from dataset import NLIaFeverDataset, AugmentedNLIaFeverTestDataset, AdversarialNLIaFeverDataset
from distilbertclassifier import DistilBertClassifier
from training import DistilBertForSequenceClassificationTrainer


def display_dataset_proportions(dataset, class_labels):
    classes = [torch.argmax(record["labels"]).item() for record in dataset]
    class_counts = Counter(classes)

    total_count = sum(class_counts.values())

    print(f"Total count: {total_count}")
    print("\nClass proportions:")
    print("------------------")

    for label, count in class_counts.items():
        percentage = (count / total_count) * 100
        print(f"{class_labels[label]}: {count} ({percentage:.2f}%)")


def display_all_datasets_proportions():
    labels = NLIaFeverDataset.get_labels()
    print("Original dataset: ")
    print("\tTrain split: ")
    train_dataset = NLIaFeverDataset("train")
    display_dataset_proportions(train_dataset, labels)
    print("\tTest split: ")
    test_dataset = NLIaFeverDataset("test")
    display_dataset_proportions(test_dataset, labels)
    print("\tValidation split: ")
    valid_dataset = NLIaFeverDataset("validation")
    display_dataset_proportions(valid_dataset, labels)

    adversarial_path = r'C:\Users\feder\PycharmProjects\MNLP_Homework_1B\augmented_dataset'
    print("\nAdversarial dataset: ")
    print("\tTrain split: ")
    train_dataset = AdversarialNLIaFeverDataset(adversarial_path, "train")
    display_dataset_proportions(train_dataset, labels)
    print("\tTest split: ")
    test_dataset = AdversarialNLIaFeverDataset(adversarial_path, "test")
    display_dataset_proportions(test_dataset, labels)
    print("\tValidation split: ")
    valid_dataset = AdversarialNLIaFeverDataset(adversarial_path, "validation")
    display_dataset_proportions(valid_dataset, labels)

    print("\nAdversarial test dataste: ")
    test_dataset = AugmentedNLIaFeverTestDataset()
    display_dataset_proportions(test_dataset, labels)


def visualize_losses_of_training(checkpoint_file: str):
    with open(checkpoint_file, "r") as f:
        checkpoint = json.load(f)

    train_losses = checkpoint['train_losses']
    valid_losses = checkpoint['valid_losses']

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.show()


def visualize_all_losses():
    normal_model_on_original = r'C:\Users\feder\PycharmProjects\MNLP_Homework_1B\NLI_distilberts\checkpoints\NLIDistilbert_60.cp'
    normal_model_on_adversarial = r'C:\Users\feder\PycharmProjects\MNLP_Homework_1B\NLI_distilberts\FINAL\checkpoints\DistilbertForSequenceClassification_10.cp'
    masked_base = r'C:\Users\feder\PycharmProjects\MNLP_Homework_1B\MASKED_distilberts\checkpoints\MaskedDistilbert_30.cp'
    masked_model_on_original = r'C:\Users\feder\PycharmProjects\MNLP_Homework_1B\NLI_masked_on_original\checkpoints\MaskedDistilbertForSequenceClassification_20.cp'
    masked_model_on_adversarial = r'C:\Users\feder\PycharmProjects\MNLP_Homework_1B\NLI_masked_on_adversarial\checkpoints\MaskedDistilbertForSequenceClassification_20.cp'

    print("Visualizing loss of Normal Model: ")
    visualize_losses_of_training(normal_model_on_original)
    input("Press Enter to continue...")
    print("\nVisualizing loss of Adversarial Model: ")
    visualize_losses_of_training(normal_model_on_adversarial)
    input("Press Enter to continue...")
    print("\nVisualization of Masked Base Model: ")
    visualize_losses_of_training(masked_base)
    input("Press Enter to continue...")
    print('\nVisualizing loss of Masked Model on original: ')
    visualize_losses_of_training(masked_model_on_original)
    input("Press Enter to continue...")
    print("\nVisualizing loss of Adversarial Masked Model: ")
    visualize_losses_of_training(masked_model_on_adversarial)
    input("Press Enter to continue...")


def test_all_models():
    normal_model_on_original_path = r'C:\Users\feder\PycharmProjects\MNLP_Homework_1B\NLI_distilberts\models\NLIDistilbert_60.pt'
    normal_model_on_adversarial_path = r'C:\Users\feder\PycharmProjects\MNLP_Homework_1B\NLI_distilberts\FINAL\models\DistilbertForSequenceClassification_10.pt'
    masked_model_on_original_path = r'C:\Users\feder\PycharmProjects\MNLP_Homework_1B\NLI_masked_on_original\models\MaskedDistilbertForSequenceClassification_20.pt'
    masked_model_on_adversarial_path = r'C:\Users\feder\PycharmProjects\MNLP_Homework_1B\NLI_masked_on_adversarial\models\MaskedDistilbertForSequenceClassification_20.pt'

    normal_model = DistilBertForSequenceClassification.from_pretrained(normal_model_on_original_path, num_labels=3)
    normal_model_on_adversarial = DistilBertForSequenceClassification.from_pretrained(normal_model_on_adversarial_path,
                                                                                      num_labels=3)
    masked_model = DistilBertClassifier.from_pretrained(masked_model_on_original_path, num_labels=3)
    masked_model_on_adversarial = DistilBertClassifier.from_pretrained(masked_model_on_adversarial_path, num_labels=3)

    labels = NLIaFeverDataset.get_labels()
    train_settings = TrainingSettings(
        batch_size=16,
        class_names=[labels[0], labels[1], labels[2]],
    )

    original_dataset = NLIaFeverDataset('test')
    adversarial_dataset = AugmentedNLIaFeverTestDataset()

    print("\nTesting normal model on original dataset: ")
    normal_tester = DistilBertForSequenceClassificationTrainer(
        model=normal_model,
        dataset=original_dataset,
        training_settings=train_settings,
        directory=""
    )
    normal_tester.test(verbose=2)
    input("Press Enter to continue...")

    print("\nTesting normal model on adversarial dataset: ")
    adversarial_tester = DistilBertForSequenceClassificationTrainer(
        model=normal_model,
        dataset=adversarial_dataset,
        training_settings=train_settings,
        directory=""
    )
    adversarial_tester.test(verbose=2)
    input("Press Enter to continue...")

    print("\nTesting adv normal model on original dataset: ")
    normal_tester = DistilBertForSequenceClassificationTrainer(
        model=normal_model_on_adversarial,
        dataset=original_dataset,
        training_settings=train_settings,
        directory=""
    )
    normal_tester.test(verbose=2)
    input("Press Enter to continue...")

    print("\nTesting adv normal model on adversarial dataset: ")
    adversarial_tester = DistilBertForSequenceClassificationTrainer(
        model=normal_model_on_adversarial,
        dataset=adversarial_dataset,
        training_settings=train_settings,
        directory=""
    )
    adversarial_tester.test(verbose=2)
    input("Press Enter to continue...")

    print("\nTesting masked model on original dataset: ")
    masked_tester = DistilBertForSequenceClassificationTrainer(
        model=masked_model,
        dataset=original_dataset,
        training_settings=train_settings,
        directory=""
    )
    masked_tester.test(verbose=2)
    input("Press Enter to continue...")

    print("\nTesting masked model on adversarial dataset: ")
    adversarial_tester = DistilBertForSequenceClassificationTrainer(
        model=masked_model,
        dataset=adversarial_dataset,
        training_settings=train_settings,
        directory=""
    )
    adversarial_tester.test(verbose=2)
    input("Press Enter to continue...")

    print("\nTesting adv masked model on original dataset: ")
    masked_tester = DistilBertForSequenceClassificationTrainer(
        model=masked_model_on_adversarial,
        dataset=original_dataset,
        training_settings=train_settings,
        directory=""
    )
    masked_tester.test(verbose=2)
    input("Press Enter to continue...")

    print("\nTesting adv masked model on adversarial dataset: ")
    adversarial_tester = DistilBertForSequenceClassificationTrainer(
        model=masked_model_on_adversarial,
        dataset=adversarial_dataset,
        training_settings=train_settings,
        directory=""
    )
    adversarial_tester.test(verbose=2)
    input("Press Enter to continue...")


if __name__ == "__main__":
    display_all_datasets_proportions()
