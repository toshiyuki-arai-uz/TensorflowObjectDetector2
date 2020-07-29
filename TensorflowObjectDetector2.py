# TensorflowObjectDetection2.py
#
# Copyright (c) 2020 Antillia.com TOSHIYUKI ARAI. ALL RIGHTS RESERVED.
#
# This is based on Tensorflow Object Detection API
# https://github.com/tensorflow/models
#    research/object_detection/object_detection_tutorial.ipynb
#
# 2020/06/20
# 2020/07/25 Modified to be run on Tensorflow 2.2.0


#You have to run the following command to get tf_slim
#pip install tf_slim

#pip install protobuf
#Run protoc
#protoc object_detection\protos\*.proto --python_out=.'

import os
import glob
import numpy as np

#2020/07/25
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import sys
import traceback

from collections import defaultdict
from io import StringIO
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

#<modified date="2020/07/27"> 
# To avoid the following error:
# AttributeError: module 'tensorflow' has no attribute 'get_default_graph'
# We have changed the following line.
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#</modified>

from utils import label_map_util
from utils import visualization_utils as vis_util

from FiltersParser       import FiltersParser

from CocoModelDownloader import CocoModelDownloader

class TensorflowObjectDetector2:
  #Constructor
  def __init__(self, frozen_graph, labels):
    self.frozen_graph = frozen_graph
    self.labels = labels
    
    #Load a frozen Tensorflow model into memory.
    self.detection_graph = tf.Graph()
    with self.detection_graph.as_default():
      od_graph_def = tf.compat.v1.GraphDef()   
      with tf.compat.v2.io.gfile.GFile(self.frozen_graph, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    
    #Loading label map    
    self.category_index = label_map_util.create_category_index_from_labelmap(
              self.labels, use_display_name=True)

    self.NUM_DETECTIONS    = 'num_detections'
    self.DETECTION_CLASSES = 'detection_classes'
    self.DETECTION_BOXES   = 'detection_boxes'
    self.DETECTION_MASKS   = 'detection_masks'
    self.DETECTION_SCORES  = 'detection_scores'


  def load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


  #2020/06/20
  # Detect objects in each image in input_image_dir, and save the detected image 
  # to output_image_dir.
  
  def detect_all(self, input_image_dir, output_image_dir):
  
      
      image_list = []

      if os.path.isdir(input_image_dir):
        image_list.extend(glob.glob(os.path.join(input_image_dir, "*.png")) )
        image_list.extend(glob.glob(os.path.join(input_image_dir, "*.jpg")) )

      print("image_list {}".format(image_list) )
          
      for image_filename in image_list:
          #image_filename will take images/foo.png
          image_file_path = os.path.abspath(image_filename)
          
          print("filename {}".format(image_file_path))
          
          self.detect(image_file_path, output_image_dir)
    
  
  ## Object detection for a single image to image_path  
  def detect(self, image_path, image_output_dir):
    image = Image.open(image_path)
    
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    
    image_np = self.load_image_into_numpy_array(image)
    
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]   
    image_np_expanded = np.expand_dims(image_np, axis=0)
    
    # Actual detection.
    #output_dict = run_inference_for_single_image(image_np, detection_graph)
        
    with self.detection_graph.as_default():
      with tf.compat.v1.Session() as sess:
      
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            self.NUM_DETECTIONS,
            self.DETECTION_CLASSES,
            self.DETECTION_BOXES,
            self.DETECTION_MASKS,
            self.DETECTION_SCORES,
        ]:
          tensor_name = key + ':0'
        
          if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)

        if self.DETECTION_MASKS in tensor_dict:
          # The following processing is only for single image
          detection_boxes = tf.squeeze(tensor_dict[self.DETECTION_BOXES], [0])
          detection_masks = tf.squeeze(tensor_dict[self.DETECTION_MASKS], [0])
          
          # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
          real_num_detection = tf.cast(tensor_dict[self.NUM_DETECTIONS][0], tf.int32)

          detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])

          detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])

          detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])

          detection_masks_reframed = tf.cast(
              tf.greater(detection_masks_reframed, 0.5), tf.uint8)

          # Follow the convention by adding back the batch dimension
          tensor_dict[self.DETECTION_MASKS] = tf.expand_dims(
              detection_masks_reframed, 0)
              
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = sess.run(tensor_dict,
                               feed_dict={image_tensor: np.expand_dims(image_np, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict[self.NUM_DETECTIONS] = int(output_dict[self.NUM_DETECTIONS][0])
        
        output_dict[self.DETECTION_CLASSES] = output_dict[self.DETECTION_CLASSES][0].astype(np.uint8)
        
        output_dict[self.DETECTION_BOXES]   = output_dict[self.DETECTION_BOXES][0]
        
        output_dict[self.DETECTION_SCORES]  = output_dict[self.DETECTION_SCORES][0]
        
        if self.DETECTION_MASKS in output_dict:
          output_dict[self.DETECTION_MASKS] = output_dict[self.DETECTION_MASKS][0]

    filename_only = self.get_filename_only(image_path)

    output_image_filepath = os.path.join(image_output_dir, filename_only)

    # Draw detected boxes, classes, scores onto image_np,
    # and save it to the output_image_filepath
    self.visualize(image_np, output_dict, output_image_filepath)


  def visualize(self, image_np, output_dict, output_image_filepath):
  
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict[self.DETECTION_BOXES],
        output_dict[self.DETECTION_CLASSES],
        output_dict[self.DETECTION_SCORES],
        self.category_index,
        instance_masks=output_dict.get(self.DETECTION_MASKS),
        use_normalized_coordinates=True,
        line_thickness=4)
    pil_img = Image.fromarray(image_np)

    pil_img.save(output_image_filepath) 


  def get_filename_only(self, input_image_filename):

     rpos  = input_image_filename.rfind("/")
     fname = input_image_filename

     if rpos >0:
         fname = input_image_filename[rpos+1:]
     else:
         rpos = input_image_filename.rfind("\\")
         if rpos >0:
            fname = input_image_filename[rpos+1:]
     return fname


def parse_args(argv):
  input_image_path  = None #"images/img.png"
  output_image_dir  = None
  filters           = None
  frozen_graph_path = None
  label_path        = None
  if len(argv) >= 2:
    input_image_path = argv[1]
        
  if len(argv) >= 3:
    output_image_dir = argv[2]

  if len(argv) >= 4:
    str_filters = argv[3]
    filtersParser = FiltersParser(str_filter)
    filters = filtersParser.get_filters()
    
  if len(argv) >= 5:
    frozen_graph_path = argv[4]

  if len(argv) == 6:
    label_path = argv[5]

  if not os.path.exists(input_image_path):
    raise Exception("Not found input_image_path {}".format(input_image_path))

  output_image_dir = os.path.join(os.getcwd(), output_image_dir)
  if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

  return (input_image_path, output_image_dir, filters, frozen_graph_path, label_path)


#
#
#
if __name__ == "__main__":
  use_coco_model = False
  """
  python TensorflowObjectDetector.py input_image_dir output_image_dir filters frozen_graph.pb  label_map.pbtxt
  """
  try:
     input_image_path, output_image_dir, filters, frozen_graph_path, label_path = parse_args(sys.argv)
     
     if len(sys.argv) <=3:
       use_coco_model = True

     if use_coco_model==True:
       print("Using CocoModel-----------")
       downloader = CocoModelDownloader()
       downloader.download()
       frozen_graph_path = downloader.get_frozen_graph_path()
       if not filters ==None:
         label_path        = downloader.get_filtered_label_path(filters)
       else:
         label_path        = downloader.get_label_path()
       print("frozen_graph_path {}".format(frozen_graph_path))
       print("label_path        {}".format(label_path))
 
     detector = TensorflowObjectDetector2(frozen_graph_path, label_path)

     if os.path.isfile(input_image_path):
       # If input_image_path is a file
       detector.detect(input_image_path, output_image_dir)
         
     elif os.path.isdir(input_image_path):
       detector.detect_all(input_image_path, output_image_dir)
       
     else:
        print("Inavlid input_file type {}".format(input_image_path))
     
  except Exception as ex:
    traceback.print_exc()

