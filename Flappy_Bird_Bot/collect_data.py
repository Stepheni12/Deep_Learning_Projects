
# coding: utf-8

# In[2]:

import numpy as np
import cv2
from mss import mss
from PIL import Image
from getkeys import key_check

spc = [1,0]
nk = [0,1]

#### CHANGE THESE ####
starting_value = 1

def keys_to_output(keys):
    '''
    Convert keys to a one-hot array
       0      1 
    [Space, NOKEY] boolean values.
    '''
    output = [0,0]

    if ' ' in keys:
        output = spc
    else:
        output = nk
    return output

bbox = {'top': 140, 'left': 710, 'width': 480, 'height': 610}
sct = mss()
training_data = []

while 1:
    sct_img = sct.grab(bbox)
    black_img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(black_img, (480,270))
    cv2.imshow('window', np.array(img))
    
    keys = key_check()
    output = keys_to_output(keys)
    training_data.append([img, output])
    
    if len(training_data) % 1000 == 0:
        print(len(training_data))
    
    if len(training_data) == 5000:
        file_name = 'train_data_{}.npy'.format(starting_value)
        np.save(file_name, training_data)
        training_data = []
        starting_value = starting_value + 1
        
    #To quit collecting training data, navigate to open cv window and press letter 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break


# In[ ]:



