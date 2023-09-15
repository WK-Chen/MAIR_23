import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import *

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

def train(model):
    # Define hyperparameters
    batch_size = 16
    learning_rate = 2e-5
    epochs = 5

    # Create data loaders for training
    train_dataset = DSTCDataset("dialog_acts.csv", tokenizer)
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
    return model

def evaluate(model):
    # Define hyperparameters
    batch_size = 16

    # Create data loaders for test
    validation_dataset = DSTCDataset("dialog_acts.csv", tokenizer, 'test')
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    # Validation loop
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in tqdm(validation_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predicted = torch.argmax(outputs.logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

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
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=15)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model = train(model)
    evaluate(model)

    # predict with human input
    # interaction()