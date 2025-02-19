from shiny import App, reactive, render, ui
from shiny.types import ImgData

from gender_cnn.predict import predict_gender

app_ui = ui.page_fluid(
    ui.panel_title('Gender Classifier'),
    ui.input_file('image', 'Upload image', accept=['.png', '.jpg', '.jpeg']),
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
        img: ImgData = {'src': image_path, 'height': '300px', 'width': '300px'}
        return img
    
    @render.text
    @reactive.event(input.predict_gender)
    def prediction():
        if input.image() is None:
            return None
        
        image_path = input.image()[0]['datapath']
        output = predict_gender(image_path)
        prediction = output['prediction']
        weighting = output['weighting']
        device = output['device']

        return f'Prediction: {prediction}. Weighting: {str(round(weighting, 2))}. Device: {device}.'

app = App(app_ui, server)