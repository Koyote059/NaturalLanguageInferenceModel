import sys

from dataset import NLIaFeverDataset
from model import RoBERTi
from classificationtrainer import train


def get_parameters():
    if len(sys.argv) == 3:
        command_type = sys.argv[1]
        data_type = sys.argv[2]
        if command_type in ['train', 'test'] and data_type in ['original', 'adversarial']:
            return command_type, data_type
    name = sys.argv[0]
    print(f"Command Error!. \nThe correct usage is: python3 {name} [train|test] --data [original|adversarial]")
    sys.exit(1)


def main():
    command_type, data_type = get_parameters()
    if command_type == 'train':
        trained_model = train(
            train_dataset=NLIaFeverDataset("train"),
            val_dataset=NLIaFeverDataset("validation"),
            model=RoBERTi()
        )
    else:
        pass


if __name__ == '__main__':
    main()
