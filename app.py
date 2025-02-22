from shiny import App, reactive, render, ui
from shiny.types import ImgData

from gender_cnn.predict import predict_gender
from gender_cnn.predict import get_backend

app_ui = ui.page_fillable(
    ui.panel_title('Gender Classifier'),
    ui.output_text('show_backend'),
    ui.navset_pill_list(
        ui.nav_panel(
            "Input and Prediction",
            ui.layout_columns(
                ui.card(
                    ui.card_header('Input'),
                    ui.input_file('image', 'Upload image', accept=['.png', '.jpg', '.jpeg'])
                ),
                ui.card(
                    ui.card_header('Example Image'),
                    ui.output_image('show_example_image', fill=True)
                )
            ),
            ui.card(
                ui.card_header('Image'),
                ui.output_image('show_image', fill=True)
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header('Predict'),
                    ui.input_action_button('predict_gender', 'Make Prediction')
                ),
                ui.card(
                    ui.card_header('Prediction'),
                    ui.output_text('prediction')
                )
            )
        ),
        widths=(3, 9)
    )
)

def server(input, output, session):
    @render.image
    def show_image():
        if input.image() is None:
            return None
            
        image_path = input.image()[0]['datapath']
        img: ImgData = {'src': image_path, 'height': '100%'}
        return img
    
    @render.image
    def show_example_image():
        image_path = 'images/Male/kratos.png'
        img: ImgData = {'src': image_path, 'height': '100%'}
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

        return f'Prediction: {prediction}. Weighting: {str(round(weighting, 2))}.'
    
    @render.text
    def show_backend():
        return f'Using device: {get_backend()[1]}.'

app = App(app_ui, server)