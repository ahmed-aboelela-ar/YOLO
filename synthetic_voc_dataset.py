import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path
import os
import xml.etree.ElementTree as ET
from constants import CLASS_MAPPING, S, B, C
import numpy as np


class VocDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        self._annopath = self.root_dir / 'Annotations'
        self._imgpath = self.root_dir / 'JPEGImages'
        self._splitpath = self.root_dir / 'ImageSets' / 'Main' / f'{split}.txt'
        
        with open(self._splitpath, 'r') as f:
            self.ids = f.read().splitlines()

        self.class_mapping = CLASS_MAPPING

        self.s = S
        self.b = B
        self.c = C



    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        name = self.ids[idx]
        img = self.read_image(name)
        anno = self.read_annotation(name)

        if self.transform:
            bbox = np.array([a['bbox'] for a in anno])
            class_labels = np.array([a['class'] for a in anno])

            transformed = self.transform(image=img, bboxes=bbox, class_labels=class_labels)
            img = transformed['image']
            anno = [{'bbox': b, 'class': c} for b, c in zip(transformed['bboxes'], transformed['class_labels'])]
            

        img = self.convert_image_to_tensor(img)
        grid = self.encode_to_grid(anno)

        

        return img, grid


    def encode_to_grid(self, boxes):
        grid = torch.zeros(self.s, self.s, self.b * 5 + self.c)
        for box in boxes:
            class_idx = box['class']
            x_yolo, y_yolo, w_yolo, h_yolo = box['bbox']

            i = int(y_yolo * self.s)
            j = int(x_yolo * self.s)

            i = min(i, self.s - 1)
            j = min(j, self.s - 1)

            x_cell = x_yolo * self.s - j
            y_cell = y_yolo * self.s - i

            w_cell = w_yolo
            h_cell = h_yolo
            
            grid[i, j, 0] = x_cell
            grid[i, j, 1] = y_cell
            grid[i, j, 2] = w_cell
            grid[i, j, 3] = h_cell
            grid[i, j, 4] = 1
            grid[i, j, self.b * 5 + class_idx] = 1

        
        return grid
            


    def read_annotation(self, anno_name):
        path = self._annopath / f"{anno_name}.xml"
        tree = ET.parse(path)
        root = tree.getroot()

        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)

        objects = []
        for obj in root.iter('object'):
            obj_struct = {}
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            box_width = xmax - xmin
            box_height = ymax - ymin

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            x_yolo = x_center / width
            y_yolo = y_center / height
            w_yolo = box_width / width
            h_yolo = box_height / height

            obj_struct['bbox'] = [x_yolo, y_yolo, w_yolo, h_yolo]
            obj_struct['class'] = self.class_mapping[obj.find('name').text]

            objects.append(obj_struct)
        return objects  



    def read_image(self, img_name):
        path = self._imgpath / f"{img_name}.jpg"
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        

        return img
    
    def convert_image_to_tensor(self, img):
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        img_tensor = img_tensor.float() / 255.0
        return img_tensor