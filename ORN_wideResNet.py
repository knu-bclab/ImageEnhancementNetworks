import torch.nn as nn
from torchvision.models import wide_resnet101_2

class Classifier(nn.Module):
    #change paremeter for transform learning
    def __init__(self, numclasses, extractFeature):
        super(Classifier, self).__init__()
        self.model_ft = wide_resnet101_2(pretrained=False)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, numclasses)
        self.extractFeature = extractFeature

    def forward(self, x):
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)
        x_f = self.model_ft.layer1(x)
        
        #extarct ORN perceptual feature for training
        x = self.model_ft.layer2(x_f)
        x = self.model_ft.layer3(x)
        x = self.model_ft.layer4(x)
        x = self.model_ft.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model_ft.fc(x)
        
        #extracting ORN feature
        if self.extractFeature == True:
            return x_f, x
        elif self.extractFeature == False:
            return x
