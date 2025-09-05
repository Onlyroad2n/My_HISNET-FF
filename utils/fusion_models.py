# utils/fusion_models.py
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_class, hidden_dim=1024, dropout=0.5):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_class)
        )

    def forward(self, x):
        return self.model(x)
