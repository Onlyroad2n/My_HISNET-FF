import torch.nn as nn

class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) model for classification."""
    def __init__(self, input_dim, num_classes, hidden_dim=1024, dropout=0.5):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """Defines the forward pass of the MLP."""
        return self.network(x)