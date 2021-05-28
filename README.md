# SSD_omniDetection<br>
# Explain<br>
This is Object detection for omnidirectional camera.<br>
SSD (Single Shot Detector) is one of the object detection in Deep Learning.<br>
I modify this code for panorama image.<br>
<br>
# Install<br>
```pip install torch==1.7.0+cu101 torchvision==0.8.0+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html```<br>
```pip install numpy```<br>
```pip install python-opencv```<br>
<br>
# Usage<br>
## ・Train<br>
```python train.py```<br>
You can train custom dataset (VOC).<br>

```python:train.py
dataset_root = "data/1~5_split4"
vgg16_weightPath = "weight/vgg16_reducedfc.pth"
```

And When you train dataset, This dataset transform is not panorama image. <br>
Please check example of dataset in ```data/1~5_split4```

## ・Evaluation
```python eval.py```<br>
You can detection object from panorama image.<br>

```python:eval.py
dataset_root = "data/20.09.30_annotation"
weight = "weight/ssd300_210523.pth"
```
