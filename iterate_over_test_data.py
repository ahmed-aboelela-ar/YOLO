import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from neural_network_backbone import TinyYoloV1
from utils import draw_and_interpret

if __name__ == '__main__':
    data_path = "synthetic_voc/JPEGImages"
    test_path = "synthetic_voc/ImageSets/Main/test.txt"
    test_result_path = "synthetic_voc/JPEGImagesPred"
    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)


    test_path_file = open(test_path, mode='r')

    with open(test_path, 'r') as f:
        ids = f.read().splitlines()

    model = TinyYoloV1()
    model.load_state_dict(torch.load("tiny_yolo_v1_best.pth"))
    model.eval()

    for id in ids:
        img_path = os.path.join(data_path, f"{id}.jpg")

        img = torch.zeros((1, 3, 448, 448))
        img[0] = torch.tensor(np.array(Image.open(img_path))).permute(2, 0, 1)
        img = img.float() / 255.0

        pred = model(img)

        img = draw_and_interpret(img[0].detach().numpy(), pred.detach().numpy())
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(test_result_path, f"{id}.jpg"))

