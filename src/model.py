import torchvision.models as models
import torch.nn as nn

class ChexNet(nn.Module):
    def __init__(self, out_size):
        super(ChexNet, self).__init__()

        self.denseNet = models.densenet121(weights = models.DenseNet121_Weights.DEFAULT)
        num_filters = self.denseNet.classifier.in_features
        self.denseNet.classifier = nn.Sequential(nn.Linear(num_filters, out_size))
    
    def forward(self,x):
        return self.denseNet(x)