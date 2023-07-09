# Importing the libraries needed
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
import pandas as pd
import torch
from torch import cuda


class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        review = str(self.data.text[index])
        review = " ".join(review.split())
        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_len,
            return_token_type_ids=True,
            truncation=False
            )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.label[index], dtype=torch.long)
        }

    def __len__(self):
        return self.len


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.
class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)

        # Apply Xavier initialization to FC layers
        torch.nn.init.xavier_uniform_(self.pre_classifier.weight)
        torch.nn.init.constant_(self.pre_classifier.bias, 0)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.constant_(self.classifier.bias, 0)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


# Function to calculate the accuracy of the model

def calculate_accuracy(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct


# Defining the training function on the 80% of the dataset for tuning the distilbert model

def train(epoch, model, training_loader, device, loss_function, optimizer):  # 10
    train_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    flag = -1
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask)
        if _ == flag:
            print("outputs:", outputs)
            print("targets:", targets)

        loss = loss_function(outputs, targets)
        if _ == flag:
            print("loss:", loss)
            print("loss item:", loss.item())

        train_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calculate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 100 == 0:
            loss_step = train_loss / nb_tr_steps
            acc_step = (n_correct * 100) / nb_tr_examples
            print(f"Training Loss per 100 steps: {loss_step}")
            print(f"Training Accuracy per 100 steps: {acc_step}")

        optimizer.zero_grad()
        loss.backward()
        # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = train_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")


def custom_trainer(hyperparams: dict = None):
    """
    Function to train the custom model using a different classification head and custom training loop
    which is different from the Trainer() method. Here we have more control on the training process, but the training
    is less efficient and not using multi GPU processors properly. In the end we used the Trainer() class.
    :param hyperparams: the hyper parameters to train, in the format of:
                        hyperparams = {"TRAIN_BATCH_SIZE": 32,
                                            "VALID_BATCH_SIZE": 32,
                                            "EPOCHS": 3,
                                            "LEARNING_RATE": 1e-05}
    :return: The trained model using the hyperparams provided.
    """

    if hyperparams is None:
        hyperparams = {"TRAIN_BATCH_SIZE": 32,
                        "VALID_BATCH_SIZE": 32,
                        "EPOCHS": 3,
                        "LEARNING_RATE": 1e-05}
    # Setting up the device for GPU usage
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)

    dataframe_rephrased_unified = pd.read_excel('../Datasets/Rephrased_GPT3_paper.xlsx')
    dataframe_rephrased_unified = dataframe_rephrased_unified.sample(frac=1).reset_index(drop=True)
    dataframe_rephrased_unified = dataframe_rephrased_unified.dropna()
    print(dataframe_rephrased_unified)

    df_bot = dataframe_rephrased_unified[["Bot-made counterpart"]].copy()
    df_bot.columns = ["text"]
    df_bot["label"] = 1
    df_human = dataframe_rephrased_unified[["Human Review"]].copy()
    df_human.columns = ["text"]
    df_human["label"] = 0
    # Concatenate the two DataFrames
    stacked_df = pd.concat([df_bot, df_human], axis=0)
    # Reset the index
    stacked_df = stacked_df.reset_index(drop=True)

    # Defining some key variables that will be used later on in the training
    MAX_LEN = max(len(review) for review in stacked_df["text"])
    TRAIN_BATCH_SIZE = hyperparams["TRAIN_BATCH_SIZE"]
    VALID_BATCH_SIZE = hyperparams["VALID_BATCH_SIZE"]
    EPOCHS = hyperparams["EPOCHS"]
    LEARNING_RATE = hyperparams["LEARNING_RATE"]
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Creating the dataset and dataloader for the neural network
    train_size = 0.8
    train_dataset = stacked_df.sample(frac=train_size)
    test_dataset = stacked_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(stacked_df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = Triage(train_dataset, tokenizer, MAX_LEN)
    testing_set = Triage(test_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }
    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }

    training_loader = DataLoader(training_set, **train_params)
    # testing_loader = DataLoader(testing_set, **test_params)
    model = DistillBERTClass()
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        train(epoch, model, training_loader, device, loss_function, optimizer)

    return model



if __name__ == "__main__":
    model_output = custom_trainer()

