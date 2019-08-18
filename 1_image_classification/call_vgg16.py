import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import vgg16_read_file_predict as vgg16
from torchvision import models, transforms
import sys

args = sys.argv
vgg16_predict = vgg16.vgg16()
vgg16_predict.predict(args[1])

#
#files = os.listdir("./data/")
#for file in files:
#    if "jpg" in file:
#        print("入力画像           : ",file[:-4])
#        vgg16_predict.predict("./data/"+file)
