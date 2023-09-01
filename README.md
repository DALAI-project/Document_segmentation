# segmentation

This repository contains a segmentation model for segmenting different part of a document. The model segments unified text parts and classifies them into typewritten and handwritten classes. In addition to this, the model segments tabular sections, pictures and signatures in a document. This model is most likely not production ready and might need some finetuning on your own data.  

## Environment installation and code

The model uses Yolov5m model developed by Ultralytics. Installation guide can be found from https://github.com/ultralytics/yolov5. A separate virtual environment is recommended for the installation. You can create a conda virtual environment by following commands. 

```conda create -n yolo python=3.7```

```conda activate yolo```

## Training

This section explains how you can train a yolov5 model. You can train by using the Ultralytics models or you can further train with the provided model. The provided model is already trained on roughly 1100 document images and can perform reasonably well. 

### Data preparation

First, you need to prepare your dataset. Our model uses 5 different classes: "typewritten", "handwritten", "signature", "table" and "image". If you decide to use Ultralytics models and train from "scratch", feel free to choose your own classes also outside of the scope provided here. 

Then you need annotated data. This link (https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/) provides information about that. The yolo models use yolo annotation format.

In addition to training data, you are going to need to create a yaml file, that describes where the training data is located and the classes used in training. You can find an example yaml found in this repository. 

### Training with Ultralytics models. 

Ultralytics provides multiple models and you can find their specifications here: https://github.com/ultralytics/yolov5#pretrained-checkpoints.

Once you have your training data and your yaml file ready, you can start training. This can be done by following command.

```python train.py --data '/path/to/yaml/file/doc_segment.yaml' --epochs 300 --weights '' --cfg yolov5m.yaml  --batch-size 40```

### Training with our pre-trained model

The process before training with our model is pretty much the same as with Ultralytics models. The only difference is that you should use our provided yaml file with our defined classes (NB! you can change the paths to the training and validation data) and define the path to our pretrained model in the training script command. An example is provided below.

```python train.py --data '/path/to/yaml/file/doc_segment.yaml' --epochs 300 --weights '/path/to/pretrained/model/model.pt'  --batch-size 40```

### How to organize training and validation data

```
├──yolov5
      ├──datasets
      |   ├──images
      |   |   ├──train
      |   |   ├──validation
      |   └──labels
      |   |   ├──train
      |   |   ├──validation
```


## Detection

Ultralytics repository has a function for inference. 

```python detect.py --weights '/path/to/model/folder/model.pt' --source '/path/to/image/folder/'```

If you are using our pretrained model, you could benefit from using and keyword argument called `--conf-thres`. This sets the confidence threshold. An optimal value for confidence threshold is 0.35. Below is and example of this.

```python detect.py --weights '/path/to/model/folder/model.pt' --source '/path/to/image/folder/' --conf-thres 0.35```


### Note about language

The model is trained to predict in Finnish language. In inference you can change the language of predictions in the following way:

```
#load model
model = torch.hub.load('/path/to/yolov5/','custom', path='/path/to/model.pt',force_reload=True,source='local')

# change names of the classes
model.names = {0:'typewritten', 1:'handwritten', 2:'signature', 3:'image', 4:'table'}

# predict
results = model('/path/to/img.jpg')
results.show()

```
