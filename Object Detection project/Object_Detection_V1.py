# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 22:14:16 2021

@author: parth
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


import fiftyone as fo
import fiftyone.zoo as foz

#
# Load the COCO-2017 validation split into a FiftyOne dataset
#
# This will download the dataset from the web, if necessary
#
coco_train= foz.load_zoo_dataset("coco-2017", split="train")
coco_val = foz.load_zoo_dataset("coco-2017", split="validation")
coco_test = foz.load_zoo_dataset("coco-2017", split="test") 
# Give the dataset a new name, and make it persistent so that you can
# work with it in future sessions
coco_train.name = "coco-2017-train"
coco_val.name = "coco-2017-validation"
coco_test.name = "coco-2017-test"
coco_train.persistent = True
coco_val.persistent = True
coco_test.persistent = True

print(coco_train.default_classes)

