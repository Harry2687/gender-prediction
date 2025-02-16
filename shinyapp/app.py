from shiny import App, render, ui
from shiny.types import ImgData

app_ui = ui.page_fluid(
    ui.panel_title('Image Uploader'),
    ui.input_file('image', 'Image', multiple=True, accept=['.png', '.jpg', '.jpeg']),
    ui.output_image('show_image')
)

def server(input, output, session):
    @render.image
    def show_image():
        if input.image() is None:
            return None
            
        image_path = input.image()[0]['datapath']
        img: ImgData = {'src': image_path}
        return img

app = App(app_ui, server)