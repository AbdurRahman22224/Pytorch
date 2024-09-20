# Imports
import torch
import torch.nn.functional as F  
import torchvision.datasets as datasets  
import torchvision.transforms as transforms 
from torch import optim 
from torch import nn  
from torch.utils.data import (
    DataLoader,
)
from tqdm import tqdm

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.MNIST(root = "dataset/", train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)

test_dataset = datasets.MNIST(root = "dataset/", train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = 64, shuffle = True)

model = NN(input_size = 784, num_classes = 10).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

num_epochs = 3
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device = device)
        target = target.to(device = device)

        # Get to correct shape
        data = data.reshape(data.shape[0], -1)

        # Forward
        scores = model(data)
        loss = criterion(scores, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()
    return num_correct / num_samples    


print(f"Accuracy on Training Set:{check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on Test Set: {check_accuracy(test_loader, model)*100:.2f}")