import torch
import torch.nn as nn
from torch.nn.functional import conv2d


class SimpleCNN(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.conv1 = self.make_block(3, 8)
        self.conv2 = self.make_block(8, 16)
        self.conv3 = self.make_block(16, 32)
        self.conv4 = self.make_block(32, 64)
        self.conv5 = self.make_block(64, 128)

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 128, 2048),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, "same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, "same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    torch.rand(1, 3, 224, 224)
    model = SimpleCNN()
    output = model(torch.rand(1, 3, 224, 224))
    print(output.shape)

