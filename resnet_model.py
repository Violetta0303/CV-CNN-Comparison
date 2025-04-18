from cnn_model import CNNModel
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetModel(CNNModel):
    def __init__(self, input_shape, num_classes, name='ResNet18', dataset_name=''):
        super().__init__(input_shape, num_classes, model_type='resnet', name=name, dataset_name=dataset_name)
        self.model = self._build_model().to(device)

    def _build_model(self):
        # Pretrained = False
        model = resnet18(weights=None)
        # model.conv1 = nn.Conv2d(self.input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1 = nn.Conv2d(self.input_shape[0], 64, kernel_size=9, stride=2, padding=4, bias=False)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model