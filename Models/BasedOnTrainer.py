from sklearn.model_selection import train_test_split
import accelerate
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import optuna
import torch.nn as nn
import wandb
import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, DistilBertModel
from transformers import Trainer, TrainingArguments
import torch.nn.init as init

wandb.login(key="892412d702fd24f7c9ff06cbac4625513f88b27d")


class ChatGPTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def preprocess_testset(testset_path):
    test_set = pd.read_excel(testset_path)
    human_texts_test = test_set['Human Review'].dropna().tolist()
    bot_texts_test = test_set['Bot-made counterpart'].dropna().tolist()

    # Assign labels: 0 for human-generated text, 1 for bot-generated text
    human_labels_test = [0] * len(human_texts_test)
    bot_labels_test = [1] * len(bot_texts_test)

    # Combine human and bot texts, and their corresponding labels
    texts_test = human_texts_test + bot_texts_test
    #     print(len(texts_test))
    labels_test = human_labels_test + bot_labels_test
    #     print(len(labels_test))

    # Initialize the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Tokenize the data
    test_encodings = tokenizer(texts_test, truncation=False, padding=True)
    test_dataset = ChatGPTDataset(test_encodings, labels_test)

    return test_dataset


def pre_process_data_rephrased_trainset(trainset_path):
    # Read the XLSX file
    train_set = pd.read_excel(trainset_path)

    # # Check for missing values and remove those rows
    # rows_with_none = dataframe[dataframe.isnull().any(axis=1)]

    # Specify the columns that contain the text
    human_texts = train_set['Human Review'].dropna()
    bot_texts = train_set['Bot-made counterpart'].dropna()

    human_texts = human_texts.tolist()
    bot_texts = bot_texts.tolist()

    # Assign labels: 0 for human-generated text, 1 for bot-generated text
    human_labels = [0] * len(human_texts)
    bot_labels = [1] * len(bot_texts)

    # Combine human and bot texts, and their corresponding labels
    texts = human_texts + bot_texts
    #     print(len(texts))
    labels = human_labels + bot_labels
    #     print(len(labels))

    # Split the data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, stratify=labels)
    # train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, stratify=labels)

    # Initialize the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Tokenize the data
    train_encodings = tokenizer(train_texts, truncation=False, padding=True)
    val_encodings = tokenizer(val_texts, truncation=False, padding=True)

    train_dataset = ChatGPTDataset(train_encodings, train_labels)
    val_dataset = ChatGPTDataset(val_encodings, val_labels)

    return train_dataset, val_dataset, val_texts, val_labels


class CustomDistilBertForSequenceClassification(DistilBertForSequenceClassification):
    def __init__(self, config, fc1_dim, dropout_rate0, dropout_rate1):
        super(CustomDistilBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(dropout_rate0)

        # Define your custom FC layers here
        self.classifier = nn.Sequential(
            nn.Linear(config.dim, fc1_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate1),
            nn.Linear(fc1_dim, config.num_labels),
        )
        self.init_weights()

        # Initialize FC layer weights using Xavier initialization
        for module in self.pre_classifier.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0.0)

        # Initialize FC layer weights using Xavier initialization
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0.0)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
    }


def wrapper_for_objective(train_dataset_paper, val_dataset_paper):
    """
    Provides the datasets to train on and validate on

    :param train_dataset_paper: training dataset with the type of ChatGPTDataset
    :param val_dataset_paper: validation dataset with the type of ChatGPTDataset
    :return: objective function to optimize on Optuna
    """

    def objective(trial):
        print("################### STARTED TRIAL {} ###################".format(trial.number))
        fc1_dim = trial.suggest_int("fc1_dim", 64, 2048)
        dropout_rate0 = trial.suggest_float("dropout_rate0", 0.05, 0.6)
        dropout_rate1 = trial.suggest_float("dropout_rate1", 0.05, 0.6)

        model = CustomDistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            fc1_dim=fc1_dim,
            dropout_rate0=dropout_rate0,
            dropout_rate1=dropout_rate1)

        # Initialize the accelerator for GPU training
        accelerator = accelerate.Accelerator()
        # Move the model and datasets to GPU
        model, train_dataset_acc, val_dataset_acc = accelerator.prepare(model, train_dataset_paper, val_dataset_paper)

        weight_decay_value = trial.suggest_float('weight_decay', 0.01, 4)
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        warmup_steps_value = trial.suggest_int('warmup_steps', 100, 900)
        batch_size = trial.suggest_int('per_batch_size', 2, 40)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=64,
            warmup_steps=warmup_steps_value,
            weight_decay=weight_decay_value,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_acc,
            eval_dataset=val_dataset_acc,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        train_result = trainer.evaluate(train_dataset_paper)
        val_result = trainer.evaluate(val_dataset_paper)

        for key, value in trial.params.items():
            print("{}:{}".format(key, value))

        print("Train Accuracy:", train_result["eval_accuracy"])
        print("Val Accuracy:", val_result["eval_accuracy"])
        print("################### FINISHED TRIAL {} ###################".format(trial.number))

        return val_result["eval_accuracy"]

    return objective


def run_visualizations(study):
    # Display the plot
    optuna.visualization.plot_param_importances(study).show()

    params_name = optuna.visualization.plot_param_importances(study).data[0].y
    params_value = optuna.visualization.plot_param_importances(study).data[0].x

    # Combine parameters and values into a list of tuples
    combined = list(zip(params_name, params_value))

    # Sort the list based on values in descending order
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)

    # Get the top 3 parameters with the highest values
    top_3 = sorted_combined[:3]

    optuna.visualization.plot_contour(study, params=[top_3[0][0], top_3[1][0]]).show()
    optuna.visualization.plot_contour(study, params=[top_3[0][0], top_3[2][0]]).show()
    optuna.visualization.plot_contour(study, params=[top_3[1][0], top_3[2][0]]).show()


def check_test(params, train_dataset_paper, val_dataset_paper, testset):
    model = CustomDistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        fc1_dim=params["fc1_dim"],
        dropout_rate0=params["dropout_rate0"],
        dropout_rate1=params["dropout_rate1"])

    # Initialize the accelerator for GPU training
    accelerator = accelerate.Accelerator()
    # Move the model and datasets to GPU

    model, train_dataset_acc, val_dataset_acc = accelerator.prepare(model, train_dataset_paper, val_dataset_paper)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        learning_rate=params["lr"],
        per_device_train_batch_size=params["per_batch_size"],
        per_device_eval_batch_size=64,
        warmup_steps=params["warmup_steps"],
        weight_decay=params["weight_decay"],
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_acc,
        eval_dataset=val_dataset_acc,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    test_result = trainer.evaluate(testset)
    print(test_result["eval_accuracy"])
    return test_result


def check_reference_score(train_dataset, val_dataset, test_dataset):
    """
    run reference model
    :param train_dataset:
    :param val_dataset:
    :param test_dataset:
    :return: accuracy score on test dataset
    """
    model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased')

    # Initialize the accelerator for GPU training
    accelerator = accelerate.Accelerator()
    # Move the model and datasets to GPU
    model, train_dataset_acc, val_dataset_acc = accelerator.prepare(model, train_dataset, val_dataset)

    trainer = Trainer(
        model=model,
        # args=training_args, default params
        train_dataset=train_dataset_acc,
        eval_dataset=val_dataset_acc,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    test_result = trainer.evaluate(test_dataset)
    print(test_result["eval_accuracy"])
    return test_result

def models_on_rephrased(training_dataset_relative_path, mode, model=1, trials=50, visualizations=0, training_params=None, test_set_path=None):
    """
    The main function for training, testing and interference of the rephrased models.
    if training_params isn't given in the format of:
    It will perform Optuna with 50 trials by default and will take the best params performed on validation set.

    :param training_dataset_relative_path: the path to the dataset
    :param mode: mode 0 is for training using optuna and then test, mode 1 is for testing a testset given training_params and test_set_path.
    :param model: if mode = 1, then if model = 0 it will test on the custom model with the given training_params and if the model = 1 it will test on reference uncustomized model.
                    on default it will train on uncustomize model (because it doens't need params)
    :param training_params:
    :param test_set_path:
    :param visualizations: flag = 1 to show visualization and 0 otherwise
    :return:
    """
    wandb.login(key="892412d702fd24f7c9ff06cbac4625513f88b27d")
    train_dataset_paper, val_dataset_paper, val_set_paper, val_label_set_paper = pre_process_data_rephrased_trainset(training_dataset_relative_path)
    cur_params = training_params
    if model == 0:
        if (mode == 0) or (mode == 1 and not training_params):
            study_rephrased_paper = optuna.create_study(direction='maximize')
            objective = wrapper_for_objective(train_dataset_paper, val_dataset_paper)
            study_rephrased_paper.optimize(objective, n_trials=trials)

            best_trial_rephrased_paper = study_rephrased_paper.best_trial
            print("Best trial:")
            print("  Value: ", best_trial_rephrased_paper.value)
            print("  Params: ")
            for key, value in best_trial_rephrased_paper.params.items():
                print("    {}: {}".format(key, value))
            print("Raw: ",best_trial_rephrased_paper.params)
            cur_params = best_trial_rephrased_paper.params

            if visualizations:
                run_visualizations(study_rephrased_paper)

    test_dataset_paper = preprocess_testset(test_set_path)
    if model == 0:
        print("Test performance on custom model:")
        check_test(cur_params, train_dataset_paper, val_dataset_paper, test_dataset_paper)
    elif model == 1:
        print("Test performance on reference model:")
        check_reference_score(train_dataset_paper, val_dataset_paper, test_dataset_paper)
    else:
        raise Exception("Unknown Model")


models_on_rephrased("../Datasets/Rephrased_GPT3_paper.xlsx", mode=0, model=1, training_params=None, test_set_path="../Datasets/Rephrased_GPT3_testset_paper.xlsx")