from dataset import augment_datasets

ADVERSARIAL_DATASET_DIRECTORY = './augmented_dataset'


def main():
    print("Data augmentation begun...")
    augment_datasets(ADVERSARIAL_DATASET_DIRECTORY)
    print("Adversarial datasets saved in: ", ADVERSARIAL_DATASET_DIRECTORY)


if __name__ == '__main__':
    main()
