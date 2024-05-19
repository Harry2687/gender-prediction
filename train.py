import os
import gdown
import zipfile
import shutil
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import modules.model as model

# Download model if not available
if os.path.exists('celeba/') == False:
    url = 'https://drive.google.com/file/d/13vkq4tFCPE8O78KTj84HHM6kBnYkt8gP/view?usp=sharing'
    output = 'download.zip'
    gdown.download(url, output, fuzzy=True)

    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall()

    os.remove(output)
    shutil.rmtree('__MACOSX')

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

# Define dataset, dataloader and transform
imsize = int(128/0.8)
batch_size = 10

fivecrop_transform = transforms.Compose([
    transforms.Resize([imsize, imsize]),
    transforms.Grayscale(1),
    transforms.FiveCrop(int(imsize*0.8)),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Normalize(0, 1)
])

train_dataset = datasets.CelebA(
    root='',
    split='all',
    target_type='attr',
    transform=fivecrop_transform,
    download=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=torch.Generator(device=device)
)

# Male index
factor = 20

# Define model, optimiser and scheduler
torch.manual_seed(2687)
resnet = model.resnetModel_128()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    resnet.parameters(), 
    lr=0.01,
    momentum=0.9,
    weight_decay=0.001
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer,
    step_size=1,
    gamma=0.1
)

def mins_to_hours(mins):
    hours = int(mins/60)
    rem_mins = mins % 60
    return hours, rem_mins

epochs = 2
train_losses = []
train_accuracy = []
for i in range(epochs):
    epoch_time = 0

    for j, (X_train, y_train) in enumerate(train_loader):
        batch_start = time.time()

        X_train = X_train.to(device)
        y_train = y_train[:, factor]

        bs, ncrops, c, h, w = X_train.size()
        y_pred_crops = resnet.forward(X_train.view(-1, c, h, w))
        y_pred = y_pred_crops.view(bs, ncrops, -1).mean(1)

        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred.data, 1)[1]
        train_batch_accuracy = (predicted == y_train).sum()/len(X_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_accuracy.append(train_batch_accuracy.item())

        batch_end = time.time()

        batch_time = batch_end - batch_start
        epoch_time += batch_time
        avg_batch_time = epoch_time/(j+1)
        batches_remaining = len(train_loader)-(j+1)
        epoch_mins_remaining = round(batches_remaining*avg_batch_time/60)
        epoch_time_remaining = mins_to_hours(epoch_mins_remaining)

        full_epoch = avg_batch_time*len(train_loader)
        epochs_remaining = epochs-(i+1)
        rem_epoch_mins_remaining = epoch_mins_remaining+round(full_epoch*epochs_remaining/60)
        rem_epoch_time_remaining = mins_to_hours(rem_epoch_mins_remaining)
        
        if (j+1) % 10 == 0:
            print(f'\nEpoch: {i+1}/{epochs} | Train Batch: {j+1}/{len(train_loader)}')
            print(f'Current epoch: {epoch_time_remaining[0]} hours {epoch_time_remaining[1]} minutes')
            print(f'Remaining epochs: {rem_epoch_time_remaining[0]} hours {rem_epoch_time_remaining[1]} minutes')
            print(f'Train Loss: {loss}')
            print(f'Train Accuracy: {train_batch_accuracy}')

    scheduler.step()

    trained_model_name = resnet.model_name + '_epoch_' + str(i+1) + '.pt'
    torch.save(
        resnet.state_dict(), 
        trained_model_name
    )