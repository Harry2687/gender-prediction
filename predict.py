import os
import gdown
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import modules.model as model

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
    
# Make model and load parameters
resnet = model.resnetModel_128()
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
    batch_size=min(len(my_dataset), 10),
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