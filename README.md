# Two-Wheeler Traffic Rule Violation Detection
This project is aimed at detecting and identifying traffic rule violations by two-wheeler riders, with a focus on the detection of violations related to helmet usage (and more in future). The project utilizes the [YOLOv5](https://github.com/ultralytics/yolov5) deep learning model to detect objects of interest, namely helmets, license plates, and motorcycles, in images or video frames.

## Dataset
The dataset used for this project is available at [Roboflow here](https://universe.roboflow.com/nckh-2023/helmet-detection-project/dataset/19). It includes images of two-wheelers captured under various conditions, with annotations for the detection of helmets, license plates, and motorcycles.

For usage in this project the *data.yaml* file is modified as:
```yaml
train: ../Helmet Detection Dataset/train/images/
val: ../Helmet Detection Dataset/valid/images/
test: ../Helmet Detection Dataset/test/images/

nc: 3
names: ['helmet', 'license plate', 'motorcycle']

roboflow:
  workspace: nckh-2023
  project: helmet-detection-project
  version: 19
  license: MIT
  url: https://universe.roboflow.com/nckh-2023/helmet-detection-project/dataset/19
```

## Local File Structure
```bash
Two-Wheeler Traffic Rule Violation Detection
├───Helmet Detection Dataset
│   ├───test
│   │   ├───images
│   │   └───labels
│   ├───train
│   │   ├───images
│   │   └───labels
│   └───valid
│       ├───images
│       └───labels
├───models
├───nohelmet lp
├───videos
├───.env
└───yolov5
```

- *Helmet Detection Dataset* - Downloaded and extracted dataset
- *models* - Stores models trained on the dataset
- *nohelmet lp* - Stores traffic rule violated two wheeler image and its license plate image
- *videos* - Stores videos to test the trained models on
- *.env* - Stores environment variable
- *yolov5* - Git clone of ***https://github.com/ultralytics/yolov5***

## Google Drive File Structure
```bash
Two-Wheeler Traffic Rule Violation Detection
├───Helmet Detection Dataset
├───yolov5
├───Helmet Detection Dataset.rar
└───yolov5m.pt
```

- *Helmet Detection Dataset.rar* - Downloaded dataset
- *Helmet Detection Dataset* - Extracted dataset (*Helmet Detection Dataset.rar*)
- *yolov5* - Git clone of ***https://github.com/ultralytics/yolov5***
- *yolov5m.pt* - Pretrained [yolov5m model](https://github.com/ultralytics/yolov5#pretrained-checkpoints) 

## Object Classes
The YOLOv5 model is trained to recognize three classes of objects:
- Helmet: To detect whether the rider is wearing a helmet.
- License Plate: To detect and readthe content of license plates on the vehicles.
- Motorcycle: To detect motorcycles.

## Model Training - *modelTraining.ipynb*
This Jupyter Notebook contains the code for training the YOLOv5 model using the provided dataset. The key steps include:

- Mounting Google Drive to access the dataset and model files.
- Installing required packages.
- Training the YOLOv5 model with custom parameters, including image size, batch size, and the number of training epochs.
    ```bash
    !python /content/drive/MyDrive/tarp/yolov5/train.py --img 512 --batch 16 --epochs 3 --data "/content/drive/MyDrive/tarp/Helmet Detection Dataset/data.yaml" --weights /content/drive/MyDrive/tarp/yolov5m.pt
    ```
- Fine-tuning the model using pre-trained weights.
    ```bash
    !python /content/drive/MyDrive/tarp/yolov5/train.py --img 512 --batch 4 --epochs 3 --data "/content/drive/MyDrive/tarp/Helmet Detection Dataset/data.yaml" --weights '/content/drive/MyDrive/tarp/yolov5/runs/train/exp3/weights/best.pt'
    ```
    > **Note:** The directory (*/content/drive/MyDrive/tarp/yolov5/runs/train/exp3/weights/best.pt*) should keep on changing (*/exp/*, */exp1/*, */exp2/*, and so on) as per where the latest trained model got saved.
- Download the latest *best.pt* file and rename it as per your need (here *512_2_5m.pt* placed in *models* folder).

## Detection and Violation Identification - *main.py*
This Python script utilizes the trained YOLOv5 model to detect objects in images, identify potential traffic rule violations, and capture relevant images for documentation. The main functionalities include:
- Using the trained YOLOv5 model to detect two wheelers, helmets, and license plates.
- Saves images of two wheelers with no helmet and their license plate images.
- Extracting license plate numbers of above using [OCR.Space API](https://ocr.space/OCRAPI).

## Getting Started
- Please follow the file structures mentioned above.
- Clone this repository,
    ```bash
    git clone https://github.com/pratham-jaiswal/two-wheeler-traffic-rule-violation.git
    ```
- Clong yolov5 repository by ultralytics,
    ```bash
    git clone https://github.com/ultralytics/yolov5
    ```
- Download the dataset, both locally and in google drive, from [here](https://universe.roboflow.com/nckh-2023/helmet-detection-project/dataset/19).
- Get your OCR.Space API key from [here](https://ocr.space/OCRAPI).
- Create a .env file with the following environment variables
    ```env
    OCR_SPACE_API = YOUR_API_KEY
    ```
- Extract the dataset into *Helmet Detection Dataset* or anywhere you may like (be sure to make changes in the code too).
- Download the pretrained yolov5 model from [here](https://github.com/ultralytics/yolov5#pretrained-checkpoints) into the google drive.
- Run the *modelTraining.ipynb* and fine tune the models as you want, then download the latest trained model (here *512_2_5m.pt*) and place it in *models* folder.
- In *main.py*, comment/uncomment video/image parts to run the model on either a video or image (keep either one commented always).
- Run the *main.py*.

## License
This project is provided under the MIT License.