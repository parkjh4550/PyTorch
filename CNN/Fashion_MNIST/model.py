import torch
from torch import nn, optim
class FlattenLayer(nn.Module):
    # CNN output feature -> flatten
    def forward(self, x):
        sizes= x.size()
        return x.view(sizes[0], -1)

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv2d(in_channels, out_channels, kernel_size)
        # MaxPool2d(kernel_size)
        # ReLU(inplace=False)
        # BatchNorm2d(num_features)
        # Dropout2d(p=0.5, inplace=False)

        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),
            FlattenLayer()
        )

        # check 1d features size
        self.test_input = torch.ones(1,1,28,28)
        self.conv_output_size = self.conv_net(self.test_input).size()[-1]

        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(self.conv_output_size, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Dropout(0.25),
            nn.Linear(200,10)
        )

        self.net = nn.Sequential(
            self.conv_net,
            self.mlp
        )

    def forward(self, x):
        output = self.net(x)
        return output

