import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
    
class PyTorchMLPDropConnect(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super().__init__()
        self.fc1 = LinearDropConnect(input_dim, hidden_dim_1, p=0.3)
        self.fc2 = LinearDropConnect(hidden_dim_1, hidden_dim_2, p=0.3)
        self.fc3 = LinearDropConnect(hidden_dim_2, output_dim, p=0.3)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

class LinearDropConnect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, p=0.5):
        super().__init__(in_features, out_features, bias)
        self.p = p  # probabilité de droppers un poids
        self.dropconnect_active = True

    def forward(self, x):
        if self.dropconnect_active and self.training:
            # dropout sur les poids
            mask = torch.bernoulli((1 - self.p) * torch.ones_like(self.weight))
            weight = self.weight * mask / (1 - self.p)  # scale = keep prob
        else:
            weight = self.weight
        
        return F.linear(x, weight, self.bias)
    
class PyTorchMLPDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)

        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x