import os
import gdown

modelsave_name = 'model_parameters.pt'
if os.path.isfile(modelsave_name) == False:
    url = 'https://drive.google.com/file/d/1_mYn2LrhG080Xvt26tWBtJ8U_0F2E1-s/view?usp=sharing'
    gdown.download(url, output=modelsave_name, fuzzy=True)