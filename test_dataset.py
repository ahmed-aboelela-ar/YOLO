import unittest
import os
import cv2
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from data_generation import generate_data, ROOT_DIR
# Import your generation logic here. 
# Assuming the previous script is named 'data_generator.py'
# from data_generator import generate_data, ROOT_DIR
# For this standalone test, we define the config directly:

class TestSyntheticDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Runs once before all tests. 
        Generates a fresh batch of data to test against.
        """
        # Clean up any existing data to ensure a fresh test
        if os.path.exists(ROOT_DIR):
            shutil.rmtree(ROOT_DIR)
        
        generate_data(image_count=10)
        # Trigger the generation (assuming the function is available)
        # If imported: generate_data()
        # For this test file to run standalone, I'm assuming you run the 
        # generator script first, or we can mock the generation here.
        if not os.path.exists(ROOT_DIR):
            raise RuntimeError(f"Data not found. Please run the generation script first to create '{ROOT_DIR}'.")

    def test_directory_structure(self):
        """Checks if the PASCAL VOC folder hierarchy is created correctly."""
        required_folders = [
            "Annotations",
            "JPEGImages",
            os.path.join("ImageSets", "Main")
        ]
        for folder in required_folders:
            path = os.path.join(ROOT_DIR, folder)
            self.assertTrue(os.path.isdir(path), f"Missing directory: {path}")

    def test_file_pairing_and_splits(self):
        """
        Checks if every ID in train.txt has a corresponding JPG and XML,
        and that they are not empty.
        """
        split_path = os.path.join(ROOT_DIR, "ImageSets", "Main", "train.txt")
        self.assertTrue(os.path.exists(split_path), "train.txt is missing")

        with open(split_path, 'r') as f:
            file_ids = [line.strip() for line in f.readlines() if line.strip()]

        self.assertGreater(len(file_ids), 0, "train.txt is empty")

        for file_id in file_ids:
            img_path = os.path.join(ROOT_DIR, "JPEGImages", f"{file_id}.jpg")
            xml_path = os.path.join(ROOT_DIR, "Annotations", f"{file_id}.xml")

            self.assertTrue(os.path.exists(img_path), f"Missing image for ID: {file_id}")
            self.assertTrue(os.path.exists(xml_path), f"Missing XML for ID: {file_id}")

    def test_xml_schema_validity(self):
        """
        Parses XML files to ensure they contain the required PASCAL fields
        and that coordinate values are sane (non-negative, xmin < xmax).
        """
        xml_dir = os.path.join(ROOT_DIR, "Annotations")
        xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
        
        for xml_file in xml_files:
            tree = ET.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()

            # Check filename tag matches actual file
            xml_filename = root.find('filename').text
            self.assertEqual(xml_filename, xml_file.replace('.xml', '.jpg'), 
                             f"Filename mismatch in {xml_file}")

            # Check image size
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            self.assertGreater(width, 0)
            self.assertGreater(height, 0)

            # Check objects
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # Logical sanity checks
                self.assertGreaterEqual(xmin, 0)
                self.assertGreaterEqual(ymin, 0)
                self.assertLessEqual(xmax, width)
                self.assertLessEqual(ymax, height)
                self.assertLess(xmin, xmax, f"xmin >= xmax in {xml_file}")
                self.assertLess(ymin, ymax, f"ymin >= ymax in {xml_file}")

    def test_pixel_color_consistency(self):
        """
        The 'Sanity Check': Opens the image and checks if the pixel 
        at the center of the bounding box matches the class color.
        Rectangle (Blue) expected: (255, 0, 0) in BGR
        Circle (Red) expected: (0, 0, 255) in BGR
        """
        xml_dir = os.path.join(ROOT_DIR, "Annotations")
        img_dir = os.path.join(ROOT_DIR, "JPEGImages")
        xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

        # Sample 5 random files to test (to save time)
        sample_files = xml_files[:5]

        for xml_file in sample_files:
            tree = ET.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()
            
            img_filename = root.find('filename').text
            img = cv2.imread(os.path.join(img_dir, img_filename))
            self.assertIsNotNone(img, f"Could not load image {img_filename}")

            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # Calculate center point of the box
                center_x = (xmin + xmax) // 2
                center_y = (ymin + ymax) // 2

                # Get pixel color at center (BGR format)
                pixel_color = img[center_y, center_x]

                if cls_name == "rectangle":
                    # Expect Blue (255, 0, 0)
                    # We allow small tolerance if you add compression/noise later
                    self.assertTrue(np.allclose(pixel_color, [255, 0, 0], atol=5),
                                    f"Expected Blue for rectangle at {center_x},{center_y}, got {pixel_color}")
                
                elif cls_name == "circle":
                    # Expect Red (0, 0, 255)
                    self.assertTrue(np.allclose(pixel_color, [0, 0, 255], atol=5),
                                    f"Expected Red for circle at {center_x},{center_y}, got {pixel_color}")

if __name__ == '__main__':
    unittest.main()