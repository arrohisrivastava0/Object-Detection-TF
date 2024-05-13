# Object-Detection and Distance Estimation with Voice Outputs for the Blind
A Python based real time object detection application utilizing SSD_MOBILENET algorithm and Tensorflow APIs, designed specifically for the blind victims.
<p>
  <img src="https://github.com/arrohisrivastava0/Object-Detection-TF/blob/main/images/test_output1.jpg" width="50%">
  <img src="https://github.com/arrohisrivastava0/Object-Detection-TF/blob/main/images/test_output2.jpg" width="50%">
  <img src="https://github.com/arrohisrivastava0/Object-Detection-TF/blob/main/images/test_output3.jpg" width="50%">
</p>

## Features
+ Real time interface
+ Can be interacted with a laptop webcam and Android Mobile webcam as well
+ Voice activated outputs
+ Real time Distance estimation
+ Warning alert system for closing objects

## Libraries
+ OpenCv - Open Source Computer Vision Library for image and video processing.
+ Pyttsx3 - Text-to-speech conversion library in Python, for voice genneration module
+ Pytesseract - Python wrapper for Google's Tesseract-OCR Engine, allowing you to extract text from images

## Tools
+ Uses Pre-trained SSD detection model trained on [COCO DATASETS](https://www.tensorflow.org/datasets/catalog/coco)
+ Uses Resnet based architectural approach for feature extraction
+ Implemented it by using [TensorFlow Object Detection API](https://github.com/tensorflow/models)
+ Install all the required Libraries as per your system’s requirement. Visit [Tensorflow](https://www.tensorflow.org/install/) for detailed installation analysis

## Model Setup
### TensorFlow Installation
+ You can simply install it on Anaconda Prompt with following commands.
```
#for cpu
pip install tensorflow
#for gpu
pip install tensorflow-gpu
```
### Download the [TensorFlow model](https://github.com/tensorflow/models) repository

### Protobuf compilation
+ Convert the .protos file to .py file extensions.
+ Head on to the [Google Protobuf Releases](https://github.com/protocolbuffers/protobuf/releases)
+ Download the protobuf version which satisfies your system compatibility.
+ Add it’s path in the environment variable
+ Inside of [tensorflow/models/research/](https://github.com/tensorflow/models/tree/master/research/) directory hit the following command:

  ```
  #From tensorflow/models/research/
  protoc object_detection/protos/*.proto --python_out=.
  ```
  + By the end of this you will have successfully converted your protos file into python files.

### Model
+ You can use any one of the pre-trained models provided by Tensorflow depending upon your system specifications from [here](https://github.com/tensorflow/models/tree/master/research/object_detection/models)
+ For a faster accuracy you can go with SSD DETECTION and for better accuracy you can go with MASK RCNN
+ Siince most of the system shows smooth performance with SSD Mobile_Net DETECTION, we will be using that.
+ This model is based on the ideology of THE MobileNet model

## Project Setup
+ Fork the repository at your profile
+ Git Clone the repository to your local machine.
+ pip install - r [requirements](https://github.com/arrohisrivastava0/Object-Detection-TF/blob/main/requirements.txt)
+ Run [webcam_blind_voice.py](https://github.com/arrohisrivastava0/Object-Detection-TF/blob/main/webcam_blind_voice.py)

### Distance Estimation
+ Utilizes values of average real height of the object(mm), focal length(mm) and sensor height(mmm) of the camera, image height in pixels.
+ Formula:

  ```python
  distance_apx=round(((f_mm*real_height_mm*img_height_px)/(object_height_px*sensor_height_mm)))
  ```

