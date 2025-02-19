import os
from huggingface_hub import hf_hub_download

model_name = 'resnetModel_128_epoch_2.pt'
if os.path.isfile(model_name) == False:
    hf_hub_download(
        repo_id='Harry2687/Gender-CNN',
        filename=model_name,
        local_dir='./'
    )