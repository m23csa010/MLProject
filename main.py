import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torchvision
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler,label_binarize
import numpy as np

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.USPS(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.USPS(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32*7*7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, criterion, optimizer, train_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

def evaluate_model(model, test_loader, writer, tag):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    all_labels = label_binarize(all_labels, classes=range(10))
    all_preds = label_binarize(all_preds, classes=range(10))


    precision = dict()
    recall = dict()
    for i in range(10):
        precision[i], recall[i], _ = precision_recall_curve(all_labels[:, i], all_preds[:, i])

        writer.add_pr_curve(tag+"Class "+str(i), all_labels[:,i], all_preds[:, 1], global_step=0)

    return accuracy, precision, recall, cm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlp_model = MLP().to(device)
cnn_model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

train_model(mlp_model, criterion, mlp_optimizer, train_loader)
train_model(cnn_model, criterion, cnn_optimizer, train_loader)

writer = SummaryWriter()
mlp_accuracy, mlp_precision, mlp_recall, mlp_cm = evaluate_model(mlp_model, test_loader, writer, 'MLP')
cnn_accuracy, cnn_precision, cnn_recall, cnn_cm = evaluate_model(cnn_model, test_loader, writer, 'CNN')
writer.close()

print("MLP Model Evaluation:")
print("Accuracy:", mlp_accuracy)
print("Precision:", mlp_precision)
print("Recall:", mlp_recall)
print("Confusion Matrix:")
print(mlp_cm)

print("\nCNN Model Evaluation:")
print("Accuracy:", cnn_accuracy)
print("Precision:", cnn_precision)
print("Recall:", cnn_recall)
print("Confusion Matrix:")
print(cnn_cm)

# Commented out IPython magic to ensure Python compatibility.
import tensorboard
# %load_ext tensorboard
# %tensorboard --logdir runs