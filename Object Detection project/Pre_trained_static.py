
import cv2
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

width = 640
height = 640

# Load image using openCV
img = cv2.imread("cars-on-motorway.jpg")
# resize the image so we can put it into the neural net
inp = cv2.resize(img, (width, height))

# convert the img to RGB
rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

#convert to uint8
rgb_tensor = tf.convert_to_tensor(rgb, dtype = tf.uint8)

#Add dimenstions to the rgb_tensor
rgb_tensor = tf.expand_dims(rgb_tensor, 0)

#os.environ['TFHUB_CACHE_DIR'] = '/tmp/tfhub_modules'
#Load pre-trained model
detector = hub.load(r'C:\Users\parth\Downloads\efficientdet_lite3_detection_1.tar\1')
#Load csv with labels of classes
labels = pd.read_csv("labels.csv", sep = ";", index_col = "ID")
labels = labels["OBJECT (2017 REL.)"]

#Create prediction 
boxes, scores, classes, num_detections = detector(rgb_tensor)

#Processing the outputs
pred_labels = classes.numpy().astype("int")[0]
pred_labels = [labels[i] for i in pred_labels]
pred_boxes = boxes.numpy()[0].astype("int")
pred_scores = scores.numpy()[0]

#Putting the boxes and labels on the image
for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
    if score<0.5:
        continue
    score_txt = f'{100*round(score)}%'
    img_boxes = cv2.rectangle(rgb, (xmin, ymax), (xmax, ymin), (0,255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_boxes, label, (xmin, ymax-10), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img_boxes, score_txt, (xmax, ymax-10), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
    
plt.imshow(img_boxes)
