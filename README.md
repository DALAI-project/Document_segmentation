# segmentation

This repository contains a segmentation model for segmenting different part of a document. The model segments unified text parts and classifies them into typewritten and handwritten classes. In addition to this, the model segments tabular sections, pictures and signatures in a document. This model is most likely not production ready and might need some finetuning on your own data.  

## Environment installation and code

The model uses Yolov5m model developed by Ultralytics. Installation guide can be found from https://github.com/ultralytics/yolov5. A separate virtual environment is recommended for the installation. You can create a conda virtual environment by following commands. 

```conda create -n yolo python=3.7```

```conda activate yolo```

## Training

This section explains how you can train a yolov5 model. You can train by using the Ultralytics models or you can further train with the provided model. The provided model is already trained on roughly 1100 document images and can perform reasonably well. 

### Training with Ultralytics models. 

Ultralytics provides multiple models and you can find their specifications here: https://github.com/ultralytics/yolov5#pretrained-checkpoints. Once you have selected the desired, you need to prepare your dataset. Our model uses 5 different classes: "typewritten", "handwritten", "signature", "table" and "image". If you decide to use Ultralytics models and train from "scratch", feel free to choose your own classes also outside of the scope provided here. Then you need annotated data. This link (https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/) provides information about that. In addition to 
