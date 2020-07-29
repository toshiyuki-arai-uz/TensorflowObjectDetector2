<html>
<body>
<h1>TensorflowObjectDetector2</h1>
<font size=3><b>
This is a simple python class TensorflowObjectDetector based on Tensorflow2 Object Detection API.<br>
</b></font>
<br>
<font size=2>
We have downloaded <a href="https://github.com/tensorflow/models/tree/master/research/object_detection">Tensorflow2 Object Detection API</a>.
and installed tensorflow==2.2.0.<br>

<table style="border: 1px solid red;">
<tr><td>
<font size=2>
git clone https://github.com/tensorflow/models.git<br>
pip install tensorflow==2.2.0<br>
pip install Cython<br>
pip install tf_slim<br>
pip install protobuf<br>
protoc object_detection\protos\*.proto --python_out=.'<br>

</font>
</td></tr>


</table>

<br>
Run TensorflowObjectDetector2.py script to detect objects in an image in the following way.<br><br>
<b>
>python TensorflowObjectDetector2.py .\images\img.png<br>
</b>
<br>
<img src="./detected/img.png" width="80%">
<br>
In this case, we use CocoModelDownloader class and download the followng file:
  'faster_rcnn_inception_v2_coco_2018_01_28.tar.gz'<br>
from 'http://download.tensorflow.org/models/object_detection/'.
<br>
<br>
<br>
See also: https://github.com/atlan-antillia/TensorflowObjectDetector

