import torch
import torch.nn as nn

class FaceEmbeddingNet(nn.Module):
    """Simple CNN to extract 128D face embeddings"""
    def __init__(self):
        super(FaceEmbeddingNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 6 * 6, 128),  # Adjust based on image size
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)