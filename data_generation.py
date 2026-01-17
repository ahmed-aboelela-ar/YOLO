import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import cv2


ROOT_DIR = os.path.join(os.getcwd(), 'synthetic_voc')
IMG_SIZE = 448
IMG_COUNT = 1000
CLASSES = ["circle", 'rectangle']


def generate_directories_of_VOC():
    os.makedirs(os.path.join(ROOT_DIR, 'JPEGImages'), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, 'Annotations'), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, 'ImageSets', 'Main'), exist_ok=True)


def generate_annotations(filename, shape_list, img_size):
    root = ET.Element('annotation')
    
    ET.SubElement(root, 'folder').text = 'VOC2007'

    ET.SubElement(root, 'filename').text = filename

    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    ET.SubElement(source, 'annotation').text = 'Unknown'
    ET.SubElement(source, 'image').text = 'Unknown'    

    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(img_size)
    ET.SubElement(size, 'height').text = str(img_size)
    ET.SubElement(size, 'depth').text = '3'


    for shape in shape_list:
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = shape['class']
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(shape['xmin'])
        ET.SubElement(bndbox, 'ymin').text = str(shape['ymin'])
        ET.SubElement(bndbox, 'xmax').text = str(shape['xmax'])
        ET.SubElement(bndbox, 'ymax').text = str(shape['ymax'])

    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    return xmlstr


def generate_data():
    generate_directories_of_VOC()
    train_file = open(os.path.join(ROOT_DIR, 'ImageSets', 'Main', 'train.txt'), 'w')

    for i in range(IMG_COUNT):
        file_id = f"syn_{i:05d}"
        filename = f"{file_id}.jpg"

        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        
        shapes_in_image = []

        num_obj = np.random.randint(len(CLASSES)) + 1
        
        for j in range(num_obj):
            obj_type = np.random.choice(CLASSES)
            
            if obj_type == 'circle':
                r = np.random.randint(15, 50)
                cx = np.random.randint(r, IMG_SIZE - r)
                cy = np.random.randint(r, IMG_SIZE - r)
                color = (255, 0, 0)
                cv2.circle(img, (cx, cy), r, color, -1)
                xmin, xmax = cx - r, cx + r
                ymin, ymax = cy - r, cy + r

            elif obj_type == 'rectangle':
                w, h = np.random.randint(30, 100), np.random.randint(30, 100)
                x = np.random.randint(w, IMG_SIZE - w)
                y = np.random.randint(h, IMG_SIZE - h)
                color = (0, 0, 255)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
                xmin, xmax = x, x + w
                ymin, ymax = y, y + h

            shapes_in_image.append({
                'class': obj_type,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })

        cv2.imwrite(os.path.join(ROOT_DIR, 'JPEGImages', filename), img)
        annotation = generate_annotations(filename, shapes_in_image, IMG_SIZE)
        with open(os.path.join(ROOT_DIR, 'Annotations', f"{file_id}.xml"), 'w') as f:
            f.write(annotation)
        train_file.write(f"{file_id}\n")
    train_file.close()




if __name__ == '__main__':
    generate_data()