import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image
from torch.utils.data import (
    Dataset,
    DataLoader,
) 

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 3
learning_rate = 0.01
batch_size = 32
num_epochs = 10
num_classes = 5
class DiabeticCustom(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0]) +'.jpg'
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


my_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resizes (32,32) to (36,36)
        transforms.ToTensor(),  # Finally converts PIL image to tensor so we can train w. pytorch
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Note: these values aren't optimal
    ]
)

# train_set = DiabeticCustom(csv_file = '/home/bhabesh/iSES 2023 Paper/Indian Dataset/Disease Grading/Retinopathy Original Images/Train_Set/client5/client5Label/Label.csv', root_dir = '/home/bhabesh/iSES 2023 Paper/Indian Dataset/Disease Grading/Retinopathy Original Images/Train_Set/client5/client5Image' , transform = my_transforms)


# test_set = DiabeticCustom(csv_file = '/home/bhabesh/iSES 2023 Paper/Indian Dataset/Disease Grading/Groundtruths/TestingLabels.csv', root_dir = '/home/bhabesh/iSES 2023 Paper/Indian Dataset/Disease Grading/Retinopathy Original Images/Test_Set/Test' , transform = my_transforms)

script_name = os.path.basename(__file__)
client_num = script_name[-1]

train_set = DiabeticCustom(csv_file = f'../Indian Dataset/Disease Grading/Retinopathy Original Images/Train_Set/client{client_num}/client{client_num}Label/Label.csv', root_dir = '../Indian Dataset/Disease Grading/Retinopathy Original Images/Train_Set/client{client_num}/client{client_num}Image' , transform = my_transforms)

test_set = DiabeticCustom(csv_file = '../Indian Dataset/Disease Grading/Groundtruths/TestingLabels.csv', root_dir = '../Indian Dataset/Disease Grading/Retinopathy Original Images/Test_Set/Test' , transform = my_transforms)

trainloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

#Model
model = torchvision.models.resnet50(weights="DEFAULT")

# freeze all layers, change final linear layer with num_classes
for param in model.parameters():
    param.requires_grad = False

# final layer is not frozen
model.fc = nn.Linear(in_features=2048, out_features=512)
# modelfc1relu = nn.Sequential(model.fc, nn.ReLU())
model.fc2 = nn.Linear(in_features=512, out_features=256)
# modelfc2relu = nn.Sequential(model.fc2, nn.ReLU())
model.fc3 = nn.Linear(in_features=256, out_features=128)
# modelfc3relu = nn.Sequential(model.fc3, nn.ReLU())
model.fc4 = nn.Linear(in_features=124, out_features=64)
# modelfc4relu = nn.Sequential(model.fc4, nn.ReLU())
model.fc5 = nn.Linear(in_features=64, out_features=num_classes)

# Add ReLU activation functions after each fully connected layer
model.relu = nn.ReLU(inplace=True)



def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


net = model.to(device)


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
