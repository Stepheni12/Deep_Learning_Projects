
# coding: utf-8

# In[6]:

from keras.models import load_model
import numpy as np
import cv2
from mss import mss
from PIL import Image
import pyautogui as pya

model = load_model("model")
bbox = {'top': 140, 'left': 710, 'width': 480, 'height': 610}
sct = mss()
training_data = []

while 1:
    sct_img = sct.grab(bbox)
    black_img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(black_img, (480,270))
    img = img.reshape(1,270,480,1)
    pred = model.predict(img)
    if pred[0].argmax() == 0:
        pya.press('space')


# In[ ]:



