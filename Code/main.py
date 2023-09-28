import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from custom_dataset_PyTorch import DiabeticCustom

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
in_channel = 3
num_classes = 10
learning_rate = 0.001
batch_size = 32
num_epochs = 1

#Load_Data

# For Macular
train_set = DiabeticCustom(csv_file = '../Indian Dataset/Disease Grading/Groundtruths/TrainingLabelsMacular.csv', root_dir = "../Indian Dataset/Disease Grading/Macular Original Images/Train_Set/Train",transform = transforms.ToTensor())

test_set = DiabeticCustom(csv_file = '../Indian Dataset/Disease Grading/Groundtruths/TestingLabels.csv', root_dir = "../Indian Dataset/Disease Grading/Macular Original Images/Test_Set/Test", transform = transforms.ToTensor())



