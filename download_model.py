import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download pretrained weights (aux_logits must be True)
weights = Inception_V3_Weights.DEFAULT
model = models.inception_v3(weights=weights, aux_logits=True)

# Replace the final fc layer for 2 classes
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 2),
)

# Move to device
model = model.to(device)

# Save weights for offline use
torch.save(model.state_dict(), "inception_v3_weights.pth")
print("✅ Saved InceptionV3 weights for offline use")
