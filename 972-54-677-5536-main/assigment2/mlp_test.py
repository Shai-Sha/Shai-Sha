import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from mlp import BaseClassifier
import numpy

feature_dim=128
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model=BaseClassifier(feature_dim=feature_dim).to(device)
print(model)
#needs to have the MNIST folder and all test files in it at this file folder
test_dataset = MNIST(".", train=False, 
                     download=False, transform=ToTensor())
test_loader = DataLoader(test_dataset, 
                         batch_size=64, shuffle=False)


def test():
  out_dim=10
  confMatrix = numpy.zeros((out_dim, out_dim))

  loss_fn = nn.CrossEntropyLoss()
  
  classifier = BaseClassifier(feature_dim=feature_dim)
  #NN weight file located at this file folder
  classifier.load_state_dict(torch.load('mnist.pt'))
  classifier.eval()
  accuracy = 0.0
  computed_loss = 0.0
  numberOfImages=0
  with torch.no_grad():
      for data, target in test_loader:
          data = data.flatten(start_dim=1)
          out = classifier(data)
          _, preds = out.max(dim=1)
          numberOfImages+=1
          # Get loss and accuracy
          computed_loss += loss_fn(out, target)
          accuracy += torch.sum(preds==target)
#      output=model(data)
#      print(output)
      print("Number of data is {}".format(len(test_loader)*64))    
      print("Test loss: {}, test accuracy: {}".format(
          computed_loss.item()/(len(test_loader)*64), accuracy*100.0/(len(test_loader)*64)))

'''  def showConfusion(self, numberOfData):
    
    for i in range(0,numberOfData):
      O=self.test(self.batchI[i])
      T=self.batchT[i]
      confMatrix[np.argmax(T)][np.argmax(O)]+=1
    print(confMatrix)'''
