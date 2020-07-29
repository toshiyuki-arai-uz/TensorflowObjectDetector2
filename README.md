<html>
<body>
<h1>TensorflowObjectDetector2</h1>
<font size=3><b>
This is a simple python class TensorflowObjectDetector based on Tensorflow2 Object Detection API.<br>
</b></font>
<br>
<font size=2>
We have downloaded <a href="https://github.com/tensorflow/models/tree/master/research/object_detection">Tensorflow2 Object Detection API</a>,  
You have to instal tensorflow>=2.0.0<br>

<table style="border: 1px solid red;">
<tr><td>
<font size=2>
git clone https://github.com/tensorflow/models.git<br>
pip install tensorflow==2.2.0<br>
pip install Cython<br>
pip install tf_slim
pip install protobuf
protoc object_detection\protos\*.proto --python_out=.'

</font>
</td></tr>


</table>


Run TensorflowObjectDetector2.py script to detect objects in an image in the following way.<br><br>
<b>
>python TensorflowObjectDetector2.py .\images\img.png<br>
</b>

See also: https://github.com/atlan-antillia/TensorflowObjectDetector

