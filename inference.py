import torch
import torch.nn as nn
from neural_network_backbone import TinyYoloV1
from synthetic_voc_dataset import VocDataset
from PIL import Image
from utils import draw_and_interpret
import numpy as np
if __name__ == '__main__':
    model = TinyYoloV1()
    model.load_state_dict(torch.load("tiny_yolo_v1_best.pth"))
    model.eval()


    img_path = input("Please enter the image path: ")
    
    img = torch.zeros((1, 3, 448, 448))
    img[0] = torch.tensor(np.array(Image.open(img_path))).permute(2, 0, 1)
    img = img.float() / 255.0

    pred = model(img)
    

    img = draw_and_interpret(img[0].detach().numpy(), pred.detach().numpy())
    img_pil = Image.fromarray(img)
    img_pil.show()
    img_pil.save("test.png")
    


