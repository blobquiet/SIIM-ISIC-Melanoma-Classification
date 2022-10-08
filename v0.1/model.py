import timm
from torch import nn
class MelanomaClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained, num_classes = n_class)
    def forward(self, x):
        x = self.model(x)
        return x
    def get_classifier(self):
        return self.model.get_classifier()
    def fc(self):
        return self.model.fc()