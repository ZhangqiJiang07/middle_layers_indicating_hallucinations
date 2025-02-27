import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.metrics import roc_auc_score
from itertools import product
import numpy as np

# set seed
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, default='./var_data/llava_7b_var_features.pt')
parser.add_argument("--start-layer", type=int, default=5)
parser.add_argument("--end-layer", type=int, default=18)
parser.add_argument("--model", type=str, default='llava-1.5', help="model")
args = parser.parse_known_args()[0]

# load data
data_raw = torch.load(args.data_path)

# statistical info
count_0 = 0
count_1 = 0
for item in data_raw:
    if item['label'] == 0:
        count_0 += 1
    elif item['label'] == 1:
        count_1 += 1
print(f"Labels: 0 = {count_0}, 1 = {count_1}")

# prepare data
data = []
start_layer, end_layer = args.start_layer, args.end_layer # first layer is noted as 0-th layer

if args.model == 'llava-1.5':
    heads_num, layers_num = 32, 32
else:
    ValueError(f"Unknown model: {args.model}")

start_VAR_idx, end_VAR_idx = (32 - 1 - end_layer) * heads_num, (32 - start_layer) * heads_num
for i in range(len(data_raw)):
    data.extend([(torch.tensor(data_raw[i]["atten"][start_VAR_idx:end_VAR_idx]), data_raw[i]["label"])])

inputs, labels = zip(*data)
inputs = torch.stack(inputs)  # convert to Tensor
labels = torch.tensor(labels)

# Split Train/Test sets
train_size = int(0.8 * len(labels))  # 80% for training set
test_size = len(labels) - train_size  # 20% for test set
train_dataset, test_dataset = random_split(TensorDataset(inputs, labels), [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# two-layer MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_model(num_epochs, hidden_size, lr):
    # Initialize model, loss, optimizer
    input_size = inputs.size(1)  # feature dim
    output_size = len(set(labels.numpy()))  # classes

    mlp_model = MLP(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=lr)

    # train model
    for epoch in range(num_epochs):
        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = mlp_model(batch_inputs)

            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    # evaluate model
    mlp_model.eval()
    correct = 0
    total = 0
    correct_label_1 = 0
    total_label_1 = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            outputs = mlp_model(batch_inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

            label_1_indices = (batch_labels == 1)
            total_label_1 += label_1_indices.sum().item()
            correct_label_1 += (predicted[label_1_indices] == batch_labels[label_1_indices]).sum().item()

            all_labels.extend(batch_labels.numpy())
            all_probs.extend(outputs.softmax(dim=1)[:, 1].numpy())

    accuracy = 100 * correct / total
    accuracy_label_1 = 100 * correct_label_1 / total_label_1 if total_label_1 > 0 else 0
    auc_roc = roc_auc_score(all_labels, all_probs)

    return accuracy, accuracy_label_1, auc_roc

# Grid search for hyperparameters
def hyperparameter_search():
    num_epochs_list = [50, 100, 150, 200]
    hidden_size_list = [64, 128, 248]
    lr_list = [0.001, 0.0001, 0.01]

    best_auc = 0.0
    best_params = None

    for num_epochs, hidden_size, lr in product(num_epochs_list, hidden_size_list, lr_list):
        print(f"Training with num_epochs={num_epochs}, hidden_size={hidden_size}, lr={lr}")
        accuracy, accuracy_label_1, auc_roc = train_model(num_epochs, hidden_size, lr)

        print(f"Overall Accuracy: {accuracy:.2f}% | Label 1 Accuracy: {accuracy_label_1:.2f}% | AUC-ROC: {auc_roc:.4f}")

        if auc_roc > best_auc:
            best_auc = auc_roc
            best_params = (num_epochs, hidden_size, lr)

    print(f"Best AUC-ROC: {best_auc:.4f} with parameters num_epochs={best_params[0]}, hidden_size={best_params[1]}, lr={best_params[2]}")


# main function
hyperparameter_search()