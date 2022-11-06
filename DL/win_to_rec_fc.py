import torch.nn as nn


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()

        # First fully connected layer
        self.fc1 = nn.Linear(55 * 30 * 6, 55 * 30)
        # Second fully connected layer
        self.fc2 = nn.Linear(55 * 30, 55)
        # Thired fully connected layer
        self.fc3 = nn.Linear(55, 1)

    def forward(self, x):
        # Pass data through conv1
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

        # First fully connected layer
        self.fc1 = nn.Linear(55 * 30 * 6, 55 * 30)
        # Second fully connected layer
        self.fc2 = nn.Linear(55 * 30, 55)
        # Thired fully connected layer
        self.fc3 = nn.Linear(55, 2)

    # First fully connected layer

    def forward(self, x):
        # Pass data through conv1
        out = self.fc3(x)
        return out
