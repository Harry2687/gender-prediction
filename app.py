from shiny import App, reactive, render, ui
from shiny.types import ImgData

from PIL import Image

import torch
import torchvision.transforms as transforms
import modules.model as model

def forward_prop(image_path):
    imsize = 128
    classes = ('Female', 'Male')
    modelsave_name = 'model_parameters.pt'

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

    resnet = model.resnetModel_128()
    resnet.load_state_dict(torch.load(modelsave_name, map_location=device))
    resnet.eval()

    loader = transforms.Compose([
        transforms.Resize([imsize, imsize]),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(0, 1)
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = loader(image)
    image_tensor = image_tensor.unsqueeze(0)

    X = image_tensor.to(device)
    y_pred = resnet.forward(X)
    predicted = torch.max(y_pred.data,1)[1]

    return f'Prediction: {classes[predicted]} with weight {y_pred[0][predicted].item()}. Predicted using {device_name}.'


app_ui = ui.page_fluid(
    ui.panel_title('Image Uploader'),
    ui.input_file('image', 'Image', accept=['.png', '.jpg', '.jpeg']),
    ui.output_image('show_image'),
    ui.input_action_button('predict_gender', 'Predict'),
    ui.output_text('prediction')
)

def server(input, output, session):
    @render.image
    def show_image():
        if input.image() is None:
            return None
            
        image_path = input.image()[0]['datapath']
        img: ImgData = {'src': image_path}
        return img
    
    @render.text
    @reactive.event(input.predict_gender)
    def prediction():
        if input.image() is None:
            return None
        
        image_path = input.image()[0]['datapath']
        prediction = forward_prop(image_path)

        return prediction

app = App(app_ui, server)