import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# For reproducability
torch.manual_seed(0)

class BaseClassifier(nn.Module):
  def __init__(self, in_dim, feature_dim, out_dim):
    super(BaseClassifier, self).__init__()
    self.classifier = nn.Sequential(
        nn.Linear(in_dim, feature_dim, bias=True),
        nn.ReLU(),
        nn.Linear(feature_dim, out_dim, bias=True)
    )
    
  def forward(self, x):
    return self.classifier(x)
    
#needs to have the MNIST folder and all test files in it at this file folder
test_dataset = MNIST(".", train=False, 
                     download=False, transform=ToTensor())
test_loader = DataLoader(test_dataset, 
                         batch_size=64, shuffle=False)


def test():
  in_dim, feature_dim, out_dim = 784, 256, 10
  lr=1e-3
  loss_fn = nn.CrossEntropyLoss()
  epochs=40
  classifier = BaseClassifier(in_dim, feature_dim, out_dim)
  optimizer = optim.SGD(classifier.parameters(), lr=lr)
  
  classifier = BaseClassifier(in_dim, feature_dim, out_dim)
  #NN weight file located at this file folder
  classifier.load_state_dict(torch.load('mnist.pt'))
  classifier.eval()
  accuracy = 0.0
  computed_loss = 0.0
  with torch.no_grad():
      for data, target in test_loader:
          data = data.flatten(start_dim=1)
          out = classifier(data)
          _, preds = out.max(dim=1)

          # Get loss and accuracy
          computed_loss += loss_fn(out, target)
          accuracy += torch.sum(preds==target)
          
      print("Test loss: {}, test accuracy: {}".format(
          computed_loss.item()/(len(test_loader)*64), accuracy*100.0/(len(test_loader)*64)))

