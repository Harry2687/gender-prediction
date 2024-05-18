import os
import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# Download model if not available
modelsave_name = 'model_parameters.pt'
if os.path.isfile(modelsave_name) == False:
    url = 'https://drive.google.com/file/d/1_mYn2LrhG080Xvt26tWBtJ8U_0F2E1-s/view?usp=sharing'
    gdown.download(url, output=modelsave_name, fuzzy=True)

# Set device
if torch.backends.mps.is_available():
    device = torch.device('mps')
    device_name = 'Apple Silicon GPU'
elif torch.cuda.is_available():
    device = torch.device('cuda')
    device_name = 'CUDA'
else:
    device = torch.device('cpu')
    device_name = 'CPU'

torch.set_default_device(device)

print(f'\nDevice: {device_name}')

# Define model
def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    ]
    if pool:
        layers.append(
            nn.MaxPool2d(4)
        )
    return nn.Sequential(*layers)

class resnetModel_128(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = 'resnetModel_128'

        self.conv_1 = conv_block(1, 64)
        self.res_1 = nn.Sequential(
            conv_block(64, 64), 
            conv_block(64, 64)
        )
        self.conv_2 = conv_block(64, 256, pool=True)
        self.res_2 = nn.Sequential(
            conv_block(256, 256),
            conv_block(256, 256)
        )
        self.conv_3 = conv_block(256, 512, pool=True)
        self.res_3 = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512)
        )
        self.conv_4 = conv_block(512, 1024, pool=True)
        self.res_4 = nn.Sequential(
            conv_block(1024, 1024),
            conv_block(1024, 1024)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*2*1024, 2048),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.res_1(x) + x
        x = self.conv_2(x)
        x = self.res_2(x) + x
        x = self.conv_3(x)
        x = self.res_3(x) + x
        x = self.conv_4(x)
        x = self.res_4(x) + x
        x = self.classifier(x)
        x = F.softmax(x, dim=1)
        return x
    
# Make model and load parameters
resnet = resnetModel_128()
resnet.load_state_dict(torch.load(modelsave_name, map_location=device))
resnet.eval()

imsize = 128
classes = ('Female', 'Male')

loader = transforms.Compose([
    transforms.Resize([imsize, imsize]),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
])

my_dataset = datasets.ImageFolder(
    root='images/',
    transform=loader
)

my_dataset_loader = DataLoader(
    my_dataset,
    batch_size=len(my_dataset),
    generator=torch.Generator(device=device)
)

# Make predictions
start_time = time.time()
with torch.no_grad():
    for i, (X, y) in enumerate(my_dataset_loader):
        X = X.to(device)
        y_pred = resnet.forward(X)
        predicted = torch.max(y_pred.data,1)[1]

        for j in range(len(X)):
            print(f'\nImage: {my_dataset.imgs[j][0]}')
            print(f'Prediction: {classes[predicted[j]]}')
            print(f'Actual: {classes[y[j]]}')
            print(f'{classes[0]} weight: {y_pred[j][0]}')
            print(f'{classes[1]} weight: {y_pred[j][1]}')

end_time = time.time()

avg_inference_time = (end_time - start_time)/len(my_dataset)
print(f'\nAverage inference time: {avg_inference_time} seconds per image\n')