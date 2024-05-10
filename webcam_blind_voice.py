# -*- coding: utf-8 -*-


import numpy as np
import os
import six.moves.urllib as urllib
import urllib.request as allib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import pytesseract
import engineio
import torch
from torch.autograd import Variable as V
import models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import pyttsx3
#from .engine import Engine
engine =pyttsx3.init()

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

arch = 'resnet50'

model_file = 'whole_%s_places365.pth.tar' % arch
if not os.path.exists(model_file):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract\\tesseract.exe'

from utils import label_map_util
#/object_detection/'   m2

from utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
	print ('Downloading the model')
	opener = urllib.request.URLopener()
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
	  file_name = os.path.basename(file.name)
	  if 'frozen_inference_graph.pb' in file_name:
	    tar_file.extract(file, os.getcwd())
	print ('Download complete')
else:
	print ('Model already exists')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

url='http://10.67.208.240:8080//shot.jpg'

import cv2
cap = cv2.VideoCapture(1)

with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph) as sess:
   ret = True
   while (ret):
      ret,image_np = cap.read()
      if cv2.waitKey(20) & 0xFF == ord('b'): 
          cv2.imwrite('opencv'+'.jpg', image_np) 
          model_file = 'whole_%s_places365.pth.tar' % arch
          
          if not os.access(model_file, os.W_OK):
              weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
              os.system('wget ' + weight_url)
              
          useGPU = 1
          
          if useGPU == 1:
              model = torch.load(model_file)
          else:
              model = torch.load(model_file, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!
        
          model.eval()
       
          centre_crop = trn.Compose([
                trn.Resize((256,256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ])

          file_name = 'categories_places365.txt'
          if not os.access(file_name, os.W_OK):
              synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
              os.system('wget ' + synset_url)
          classes = list()
          with open(file_name) as class_file:
              for line in class_file:
                  classes.append(line.strip().split(' ')[0][3:])
          classes = tuple(classes)

          img_name = 'opencv.jpg'
          if not os.access(img_name, os.W_OK):
              img_url = 'http://places2.csail.mit.edu/demo/' + img_name
              os.system('wget ' + img_url)
    
          img = Image.open(img_name)
          input_img = V(centre_crop(img).unsqueeze(0), volatile=True)

          logit = model.forward(input_img)
          h_x = F.softmax(logit, 1).data.squeeze()
          probs, idx = h_x.sort(0, True)
        
          print('POSSIBLE SCENES ARE: ' + img_name)
          engine.say("Possible Scene may be")
          engine.say(img_name)

          for i in range(0, 5):
              engine.say(classes[idx[i]])
              print('{}'.format(classes[idx[i]]))
      
       
      # Expand dimensions: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      
      # Visualization of the results of a detection.
      if cv2.waitKey(2) & 0xFF == ord('a'):
          vis_util.vislize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      else:    
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
      if cv2.waitKey(2) & 0xFF == ord('r'):
          gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
          text=pytesseract.image_to_string(image_np)
          print(text)
          engine.say(text)
          engine.runAndWait()
      
      for i,b in enumerate(boxes[0]):
        ymin, xmin, ymax, xmax = b
        img_height_px=image_np.shape[0]
        f_mm=4.17
        sensor_height_mm=25.0
        
        #                 bus                  truck
        if classes[0][i] == 6 or classes[0][i] == 8:
          if scores[0][i] >= 0.5:
              object_height_px=(ymax-ymin)*img_height_px
              real_height_mm=3500
              distance_apx=round(((f_mm*real_height_mm*img_height_px)/(object_height_px*sensor_height_mm)))
              mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
              mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
              distance_apx_m=round((distance_apx*0.001),2)
              cv2.putText(image_np, '{}'.format(distance_apx_m), (int(mid_x*500),int(mid_y*500)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
              print(str(distance_apx_m)+"m")
              
              if distance_apx <=300:
                  if distance_apx <=300:
                      cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                      print("Warning -Vehicles Approaching")
                      engine.say("Warning -Vehicles Approaching")
                      engine.say(distance_apx)
                      engine.say("metres")
                      engine.runAndWait()
                      
        #               car
        if classes[0][i]==3:
            if scores[0][i] >= 0.5:
                img_height_px=image_np.shape[0]
                object_height_px=(ymax-ymin)*img_height_px
                real_height_mm=1550
                f_mm=4.17
                sensor_height_mm=25.0
                distance_apx=round(((f_mm*real_height_mm*img_height_px)/(object_height_px*sensor_height_mm)))
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                distance_apx_m=round((distance_apx*0.001),2)
                cv2.putText(image_np, '{}'.format(distance_apx_m), (int(mid_x*500),int(mid_y*500)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(str(distance_apx_m)+" m")
                if distance_apx <=300:
                    cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                    print("Warning vehicles approaching")
                    engine.say("Warning vehicles approaching "+str(distance_apx_m)+" meters")
                    engine.runAndWait()
                    
        #               bottle        
        if classes[0][i] ==44:
            if scores[0][i] >= 0.5:
                img_height_px=image_np.shape[0]
                object_height_px=(ymax-ymin)*img_height_px
                real_height_mm=300
                f_mm=4.17
                sensor_height_mm=25.0
                distance_apx=round(((f_mm*real_height_mm*img_height_px)/(object_height_px*sensor_height_mm)))
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                distance_apx_m=round((distance_apx*0.001),2)
                cv2.putText(image_np, '{}'.format(distance_apx_m), (int(mid_x*500),int(mid_y*500)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(str(distance_apx_m)+" m")
                if distance_apx <=300:
                    cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                    print("Warning -BOTTLE very close to the frame")
                    engine.say("Warning -BOTTLE very close to the frame "+str(distance_apx_m)+" meters")
                    engine.runAndWait()
                    
        #               person
        if classes[0][i] ==1:
            if scores[0][i] >= 0.5:
                img_height_px=image_np.shape[0]
                x=0.0
                if((ymax-ymin)>(xmax-xmin)):
                    x=(ymax-ymin)
                else:
                    x=(xmax-xmin)
                object_height_px=(x)*img_height_px
                real_height_mm=1650
                f_mm=4.17
                sensor_height_mm=25.0
                distance_apx=round(((f_mm*real_height_mm*img_height_px)/(object_height_px*sensor_height_mm)))
                mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                distance_apx_m=round((distance_apx*0.001),2)
                cv2.putText(image_np, '{}'.format(distance_apx_m), (int(mid_x*500),int(mid_y*500)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(str(distance_apx_m)+" m")
                if distance_apx <=300:
                    cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                    print("Warning -Person very close to the frame")
                    engine.say("Warning -Person very close to the frame "+str(distance_apx_m)+"meters")
                    engine.runAndWait()
                
      cv2.imshow('image',cv2.resize(image_np,(1024,768)))
      if cv2.waitKey(2) & 0xFF == ord('x'):
          cv2.destroyAllWindows()
          cap.release()
          ret=False
          break

	
#{1: {'id': 1, 'name': 'person'}, 2: {'id': 2, 'name': 'bicycle'}, 3: {'id': 3, 'name': 'car'}, 4: {'id': 4, 'name': 'motorcycle'}, 5: {'id': 5, 'name': 'airplane'}, 6: {'id': 6, 'name': 'bus'}, 7: {'id': 7, 'name': 'train'}, 8: {'id': 8, 'name': 'truck'}, 9: {'id': 9, 'name': 'boat'}, 10: {'id': 10, 'name': 'traffic light'}, 11: {'id': 11, 'name': 'fire hydrant'}, 13: {'id': 13, 'name': 'stop sign'}, 14: {'id': 14, 'name': 'parking meter'}, 15: {'id': 15, 'name': 'bench'}, 16: {'id': 16, 'name': 'bird'}, 17: {'id': 17, 'name': 'cat'}, 18: {'id': 18, 'name': 'dog'}, 19: {'id': 19, 'name': 'horse'}, 20: {'id': 20, 'name': 'sheep'}, 21: {'id': 21, 'name': 'cow'}, 22: {'id': 22, 'name': 'elephant'}, 23: {'id': 23, 'name': 'bear'}, 24: {'id': 24, 'name': 'zebra'}, 25: {'id': 25, 'name': 'giraffe'}, 27: {'id': 27, 'name': 'backpack'}, 28: {'id': 28, 'name': 'umbrella'}, 31: {'id': 31, 'name': 'handbag'}, 32: {'id': 32, 'name': 'tie'}, 33: {'id': 33, 'name': 'suitcase'}, 34: {'id': 34, 'name': 'frisbee'}, 35: {'id': 35, 'name': 'skis'}, 36: {'id': 36, 'name': 'snowboard'}, 37: {'id': 37, 'name': 'sports ball'}, 38: {'id': 38, 'name': 'kite'}, 39: {'id': 39, 'name': 'baseball bat'}, 40: {'id': 40, 'name': 'baseball glove'}, 41: {'id': 41, 'name': 'skateboard'}, 42: {'id': 42, 'name': 'surfboard'}, 43: {'id': 43, 'name': 'tennis racket'}, 44: {'id': 44, 'name': 'bottle'}, 46: {'id': 46, 'name': 'wine glass'}, 47: {'id': 47, 'name': 'cup'}, 48: {'id': 48, 'name': 'fork'}, 49: {'id': 49, 'name': 'knife'}, 50: {'id': 50, 'name': 'spoon'}, 51: {'id': 51, 'name': 'bowl'}, 52: {'id': 52, 'name': 'banana'}, 53: {'id': 53, 'name': 'apple'}, 54: {'id': 54, 'name': 'sandwich'}, 55: {'id': 55, 'name': 'orange'}, 56: {'id': 56, 'name': 'broccoli'}, 57: {'id': 57, 'name': 'carrot'}, 58: {'id': 58, 'name': 'hot dog'}, 59: {'id': 59, 'name': 'pizza'}, 60: {'id': 60, 'name': 'donut'}, 61: {'id': 61, 'name': 'cake'}, 62: {'id': 62, 'name': 'chair'}, 63: {'id': 63, 'name': 'couch'}, 64: {'id': 64, 'name': 'potted plant'}, 65: {'id': 65, 'name': 'bed'}, 67: {'id': 67, 'name': 'dining table'}, 70: {'id': 70, 'name': 'toilet'}, 72: {'id': 72, 'name': 'tv'}, 73: {'id': 73, 'name': 'laptop'}, 74: {'id': 74, 'name': 'mouse'}, 75: {'id': 75, 'name': 'remote'}, 76: {'id': 76, 'name': 'keyboard'}, 77: {'id': 77, 'name': 'cell phone'}, 78: {'id': 78, 'name': 'microwave'}, 79: {'id': 79, 'name': 'oven'}, 80: {'id': 80, 'name': 'toaster'}, 81: {'id': 81, 'name': 'sink'}, 82: {'id': 82, 'name': 'refrigerator'}, 84: {'id': 84, 'name': 'book'}, 85: {'id': 85, 'name': 'clock'}, 86: {'id': 86, 'name': 'vase'}, 87: {'id': 87, 'name': 'scissors'}, 88: {'id': 88, 'name': 'teddy bear'}, 89: {'id': 89, 'name': 'hair drier'}, 90: {'id': 90, 'name': 'toothbrush'}}

