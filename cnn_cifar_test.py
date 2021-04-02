#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:41:06 2021

@author: liuyi
"""

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
i=5

# Normalize pixel values to be between 0 and 1
#归一化
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

model = load_model('/Users/liuyi/Desktop/cifar10/cifar10_data/my_test_model.ckpt')
from keras.preprocessing import image
img = test_images[i]
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=32)
print(classes)
y_test_pred = np.argmax(classes, axis=1)
print(y_test_pred,class_names[y_test_pred[0]])

plt.figure(figsize=(10,10))

plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(test_images[i], cmap=plt.cm.binary)
# The CIFAR labels happen to be arrays, 
# which is why you need the extra index
plt.xlabel(class_names[test_labels[i][0]])
plt.show()