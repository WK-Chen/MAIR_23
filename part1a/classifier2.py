import os
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, average_precision_score, confusion_matrix, f1_score, \
    precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.utils import *

classes = ['ack', 'affirm', 'bye', 'confirm', 'deny', 'hello', 'inform', 'negate', 'null',
           'repeat', 'reqalts', 'reqmore', 'request', 'restart', 'thankyou']

# turn label to number
label_to_seq = {
    "ack": 0,
    "affirm": 1,
    "bye": 2,
    "confirm": 3,
    "deny": 4,
    "hello": 5,
    "inform": 6,
    "negate": 7,
    "null": 8,
    "repeat": 9,
    "reqalts": 10,
    "reqmore": 11,
    "request": 12,
    "restart": 13,
    "thankyou": 14,
}

seq_to_label = {v: k for k, v in label_to_seq.items()}


class DSTCDataset(Dataset):
    def __init__(self, path, tokenizer, dataset='trn'):
        self.data = self.process(path, dataset)
        self.tokenizer = tokenizer

    def process(self, path, dataset):
        data_ = load_csv(path)
        data = [{"text": sample[1], "label": sample[0]} for sample in data_]

        # split the dataset into train set and test set
        if dataset == 'trn':
            data = data[:int(len(data) * 0.85)]
        elif dataset == 'test':
            data = data[int(len(data) * 0.85):]
        else:
            assert False
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = label_to_seq[item['label']]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=64, return_tensors='pt')

        inputs = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': label
        }

        return inputs


def train(path, model):
    # Define hyperparameters
    batch_size = 16
    learning_rate = 2e-5
    epochs = 3

    # Create data loaders for training
    train_dataset = DSTCDataset(path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")

    model.save_pretrained(os.path.join(root_path, "models/trained_bert"))
    return model


def evaluate(path, model):
    # Define hyperparameters
    batch_size = 16

    # Create data loaders for test
    validation_dataset = DSTCDataset(path, tokenizer, 'test')
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    predicts = []
    labels = []
    # Validation loop
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in tqdm(validation_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predict = torch.argmax(outputs.logits, dim=1)
            predicts.extend([seq_to_label[p] for p in predict.tolist()])
            labels.extend([seq_to_label[l] for l in label.tolist()])
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

    # accuracy = correct / total
    # print(f"Validation Accuracy: {accuracy:.4f}")
    # Print evaluation metrics
    print(f"Accuracy on test data: {accuracy_score(labels, predicts)}")
    print(f"Average precision score: {precision_score(labels, predicts, average='macro', zero_division=1.0)}")
    print(f"Average recall score: {recall_score(labels, predicts, average='macro', zero_division=1.0)}")
    print(f"Average F1 score score: {f1_score(labels, predicts, average='macro', zero_division=1.0)}")
    #
    # Print Condusion Matrix
    confusion = confusion_matrix(labels, predicts, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=classes)
    disp.plot()
    # plt.show(block=False)
    if "dedup" not in path:
        plt.title("Figure 1: Confusion Matrix of the dataset with duplicates")
        plt.savefig('1.png')
    else:
        plt.title("Figure 2: Confusion Matrix of the dataset without duplicates")
        plt.savefig('2.png')


def interaction(tokenizer, model):
    # Start diaglogue
    while True:
        user_input = input("User: ")
        if user_input == "":
            break
        # Preprocess user input, for example, by creating a list of user inputs
        user_inputs = [user_input]
        # Transform the user inputs using encoding
        encoding = tokenizer(user_inputs, truncation=True, padding='max_length', max_length=64, return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze().to(device),
        attention_mask = encoding['attention_mask'].squeeze().to(device)

        # Make predictions using the classifier
        outputs = model(input_ids, attention_mask=attention_mask)
        predicted = torch.argmax(outputs.logits, dim=1)
        print(predicted)
        print(seq_to_label[predicted])


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # model_name = 'bert-base-uncased'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if sys.argv[1] == "train":
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=15)
    else:
        model_path = sys.argv[3]
        model = BertForSequenceClassification.from_pretrained(os.path.join(root_path, model_path), num_labels=15)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if sys.argv[1] == "train":
        model = train(os.path.join(root_path, sys.argv[2]), model)
    evaluate(os.path.join(root_path, sys.argv[2]), model)

    # predict with human input
    # interaction()
