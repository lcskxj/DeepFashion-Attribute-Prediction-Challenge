# -*- coding: utf-8 -*-

import torchvision
import torch.nn as nn
import torch.nn.functional as F
from config import *
import torch


class MultiHeadsNetwork(nn.Module):
    '''
    input: N * 3 * 224 * 224
    '''

    def __init__(self, inter_dim=INTER_DIM, category_attributes=CATEGORIES):
        super(MultiHeadsNetwork, self).__init__()
        self.category_attributes = category_attributes

        self.backbone = torchvision.models.resnet50(pretrained=True)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, inter_dim)
        
        self.category_layers = nn.ModuleList()
        for idx in range(len(category_attributes)):
            # self.backbone.fc = nn.Linear(self.backbone.fc.in_features, category_attributes[idx])
            # self.category_layers.append(self.backbone)
            category = nn.Linear(inter_dim, category_attributes[idx])
            self.category_layers.append(category)

    def forward(self, x):
        x = F.relu_(self.backbone(x))

        outputs = []
        for idx in range(len(self.category_attributes)):
            category_out = self.category_layers[idx](x)
            category_out = nn.functional.softmax(category_out, dim=1)
            outputs.append(category_out)

        return outputs


class MultiLabelsNetwork(nn.Module):
    '''
    input: N * 3 * 224 * 224
    '''

    def __init__(self):
        super(MultiLabelsNetwork, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # self.backbone.fc = nn.Linear(self.backbone.fc.in_features, sum(CATEGORIES))

        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, INTER_DIM),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(INTER_DIM, sum(CATEGORIES))
        )


    def forward(self, x):
        output = self.backbone(x)
        output = torch.sigmoid(output)
        # print(output)
        return output


if __name__ == "__main__":
    # model = MultiHeadsNetwork()
    model = MultiLabelsNetwork()
    print(model)
