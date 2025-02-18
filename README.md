---
title: Gender CNN Demo
emoji: 👁
colorFrom: pink
colorTo: yellow
sdk: docker
pinned: true
short_description: Shiny application to demo my gender classification CNN.
---
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Gender-Prediction

Scripts to train my neural network, or use my pre-trained parameters to make predictions.

## Setup

1. Set up virtual environment.
```
python3 -m venv venv
```
2. Activate virtual environment.
    - On Mac/Linux:
    ```
    source venv/bin/activate
    ```
    - On Windows:
    ```
    # In cmd.exe
    venv\Scripts\activate.bat
    # In PowerShell
    venv\Scripts\Activate.ps1
    ```
3. Install requirements.
```
pip3 install -r requirements.txt
```

## Prediction

Run `predict.py`, this will download pre-trained model parameters and make gender predictions on images in the `images` folder.

## Training

Run `train.py`, this will download the training data used to train my pre-trained model and start training the neural network on the your local device.