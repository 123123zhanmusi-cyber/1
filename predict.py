import torch
import numpy as np
from model_lstm import ECGLSTM
from ecg_image_to_signal import extract_ecg_from_image

classes = [
"正常心律",
"房颤",
"室性早搏",
"融合搏"
]

def predict_ecg(image_path):

    signal = extract_ecg_from_image(image_path)

    x = torch.tensor(signal,dtype=torch.float32).view(1,300,1)

    model = ECGLSTM()

    model.load_state_dict(
        torch.load("ecg_lstm_model.pth",map_location="cpu")
    )

    model.eval()

    with torch.no_grad():

        output = model(x)

        pred = torch.argmax(output,1).item()

    return classes[pred]