from shiny import render, ui
from shiny.types import ImgData
from shiny.express import input

ui.panel_title('Image Uploader')
ui.input_file('image', 'Image', multiple=True, accept=['.png', '.jpg', '.jpeg'])

@render.image
def render_image():
    if input.image() is None:
        return None
        
    image_path = input.image()[0]['datapath']
    img: ImgData = {'src': image_path}
    return img
