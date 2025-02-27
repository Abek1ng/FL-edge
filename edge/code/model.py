import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=10, num_classes=2):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    model = MLPClassifier()
    print(model)
