import torch
import torchvision.transforms as transforms
from PIL import Image
from .model import resnetModel_128

def get_backend():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = 'Apple Silicon GPU'
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = 'CUDA'
    else:
        device = torch.device('cpu')
        device_name = 'CPU'

    return [device, device_name]

def predict_gender(image_path: str):
    # Constants
    imsize = 128
    classes = ('Female', 'Male')
    model_name = 'resnetModel_128_epoch_2.pt'

    # Set Backend
    device, device_name = get_backend()

    # Init model
    resnet = resnetModel_128().to(device)
    resnet.load_state_dict(torch.load(model_name, map_location=device))
    resnet.eval()

    # Load and transform image
    loader = transforms.Compose([
        transforms.Resize([imsize, imsize]),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(0, 1)
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = loader(image)
    image_tensor = image_tensor.unsqueeze(0)

    # Predict
    X = image_tensor.to(device)
    y_pred = resnet.forward(X)
    pred_index = torch.max(y_pred.data,1)[1]
    prediction = classes[pred_index]
    weighting = y_pred[0][pred_index].item()

    return {'prediction': prediction, 'weighting': weighting, 'device': device_name}