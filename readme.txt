Make sure you have Python 3.10+ installed.
Install all the packages with:
- pip install -r requirements.txt 132
To create the augmented version of the dataset:
- python 2133034-augment.py

To train or test the model:
- python 2133034-main [train|test] [original|adversarial]

Notice:
- By doing ... test adversarial, it will use the adversarial test dataset
- By doing ... train adversarial, it will override the weights of the model trained on the
    original dataset
- After training, two folders will be created: one containing the model "models" and one containing
    log information about the training process ( checkpoints )


In "report_models" you will find all the models used in the report.
In "report_augmented_dataset" you will find the augmented dataset used in the report ( which
    - since it's generated randomly - it could change every time )