import torch
import torch.nn as nn


# For reproducability
torch.manual_seed(0)

class BaseClassifier(nn.Module):
  def __init__(self, in_dim=784, feature_dim=256, out_dim=10):
    super(BaseClassifier, self).__init__()
    self.classifier = nn.Sequential(
        nn.Linear(in_dim, feature_dim, bias=True),
        nn.ReLU(),
        nn.Linear(feature_dim, out_dim, bias=True)
    )
    
  def forward(self, x):
    return self.classifier(x)
    


