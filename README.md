# Custom Tiny YOLOv1 for Synthetic Shape Detection

This project is a complete, from-scratch implementation of a Tiny YOLOv1 object detection model using PyTorch. It includes a custom synthetic data generator that creates geometric shapes (circles and rectangles) in PASCAL VOC format, along with scripts for training, testing, and inference.

## Project Structure

* **`data_generation.py`**: A script that generates a synthetic dataset of 16,384 images. Each image contains random red circles and blue rectangles. It creates the standard PASCAL VOC directory structure (`JPEGImages`, `Annotations`, `ImageSets/Main`).
* **`test_data_generation.py`**: Unit tests to verify the integrity of the generated data, checking for correct directory structures, file pairings, XML schema validity, and pixel color consistency.
* **`synthetic_voc_dataset.py`**: A custom PyTorch `Dataset` class (`VocDataset`) that reads the generated images and XML annotations. It encodes the ground truth into the $S \times S \times (B \times 5 + C)$ grid format required by the YOLO loss function.
* **`neural_network_backbone.py`**: Defines the `TinyYoloV1` model architecture. It consists of a custom backbone of convolutional blocks (Conv2d + BatchNorm + LeakyReLU) followed by a fully connected head that outputs the prediction tensor.
* **`loss.py`**: Implements the custom `YoloLoss` function. It calculates the loss based on localization (coordinates), object confidence, and class prediction, using specific masking to ensure only the responsible bounding box predictor is penalized.
* **`train.py`**: The main training loop. It sets up the model, dataloaders, and optimizer (Adam). It trains for 100 epochs, tracking validation loss and saving the best model state to `tiny_yolo_v1_best.pth`.
* **`inference.py`**: An interactive script that loads a trained model and asks the user for an image path to detect objects in. It visualizes the result using OpenCV and PIL.
* **`iterate_over_test_data.py`**: A script that runs inference on the entire test set (defined in `test.txt`) and saves the resulting images with drawn bounding boxes to a `JPEGImagesPred` directory.
* **`utils.py`**: Contains utility functions for Intersection Over Union (IOU) calculation, Non-Maximum Suppression (NMS), coordinate conversion, and visualization functions (`draw_and_interpret`).
* **`constants.py`**: Configurations including the class names (`circle`, `rectangle`) and confidence threshold.

## Model Details

* **Architecture**: Tiny YOLOv1 (Modified)
* **Grid Size (S)**: 7
* **Bounding Boxes per Cell (B)**: 2
* **Classes (C)**: 2 ("circle", "rectangle")
* **Input Image Size**: 448x448

## Getting Started

### 1. Prerequisites
Ensure you have the following Python packages installed:
* `torch`
* `torchvision`
* `numpy`
* `opencv-python`
* `Pillow`

### 2. Generate the Dataset
First, generate the synthetic dataset. This will create a `synthetic_voc` folder in your root directory.
```bash
python data_generation.py
```

### 3. Train the Model
Start the training process. The script will automatically detect if a CUDA device is available.
```bash
python train.py
```
* **Output**: The script saves `tiny_yolo_v1_best.pth` (best validation loss) and `tiny_yolo_v1.pth` (final epoch) to the current directory.

### 4. Inference
**Single Image:**
To test the model on a specific image, run:
```bash
python inference.py
```
Enter the path to an image when prompted (e.g., `synthetic_voc/JPEGImages/syn_00001.jpg`).

**Full Test Set:**
To generate predictions for all test images:
```bash
python iterate_over_test_data.py
```
The output images with drawn bounding boxes will be saved in the `synthetic_voc/JPEGImagesPred` folder.
