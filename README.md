<html>
<body>
<h1>TensorflowObjectDetector2</h1>
<font size=3><b>
This is a simple python class TensorflowObjectDetector2 based on Tensorflow2 Object Detection API.<br>
</b></font>
<br>
<h2> 1 Installation </h2>

<font size=2>
 We have downloaded <a href="https://github.com/tensorflow/models/tree/master/research/object_detection">Tensorflow2 Object Detection API</a>.
and installed tensorflow==2.2.0.<br>
<br>
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
Please clone <a href="https://github.com/atlan-antillia/TensorflowObjectDetector2.git">TensorflowObjectDetector2.git</a> in a working folder.
<pre>
>git clone  https://github.com/atlan-antillia/TensorflowObjectDetector2.git
</pre>
Copy the files in that folder to <i>somewhere_cloned_folder/models/research/object_detection/</i> folder.
<br>


<h2>2 TensorflowObjectDetector2 class</h2>
 We have updated TensorflowObjectDetector2 class and added a new <i>vis_utils2.py</i> file to be able to save detected objects information of the list of 
each detected object data (id, label, score) to a text file.

<br>
Run TensorflowObjectDetector2.py script to detect objects in an image in the following way.<br><br>
<b>
<pre>
>python TensorflowObjectDetector2.py .\images\img.png detected<br>
</pre>
</b>
<br>
<img src="./detected/img.png" width="80%">
<br>
In this case, we use CocoModelDownloader class and download the followng file:
  'faster_rcnn_inception_v2_coco_2018_01_28.tar.gz'<br>
from 'http://download.tensorflow.org/models/object_detection/'.
<br>
<br>

<img src="./detected/img.png.txt.png" width="80%">


<br>
See also: https://github.com/atlan-antillia/TensorflowObjectDetector



<h2>3 Filters </h2>
 We have also updated TensorflowObjectDetector2 class to accept filters to select some classes
 only specified by filters.<br>
We use the following list format to define filters:<br>
<pre>
 [person,car]
</pre>

For example, you can run the following command to select persons or cars only from an imput image file.<br>
<pre> 
python TensorflowObjectDetector2.py .\images\img.png detected [person,car]
</pre>

<br>
<img src="./detected/person_car_img.png" width="80%">
<br>


<br>
<img src="./detected/person_car_img.png.txt.png" width="80%">
<br>


 