#Importing relevant libraries:::
import cv2
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization



#Data Preprocessing:::
#add in data here, augmentation here too
def data_preprocessing(image):
    return image

#CNN Model Architetcure:::
    #function that contains architecture
def building_model():
    pass

    #model = sequential
    #crop any unecessary parts of images 
    # normalise 

    #5x5 layers - depth = (24, 36, 48)
    #3x3 layers - depth = (64, 64)


    #for flattening + connecting layers:
    #flatter - dense - dropout - dense - dense - dense
    #use elu activation instead of relu activation


    #return the model for training 



#Training the Model:::
#functions to train the model
def training_model():
    pass



#Hyperparameter tuning and experimentation:::



#Evalute and Iterate:::



