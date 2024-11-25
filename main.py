#Importing relevant libraries:::
import cv2
import numpy as np
import pandas as pd
import matplotlib as plt

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import os



#upload data here 
#then process it before augmentation + model training 




#Data Preprocessing:::
#add in data here, augmentation here too
def data_preprocessing(image):
    #resizing and normalizing data on image
    image = cv2.resize(image, (64, 64))  
    image = image / 255.0

    return image


#ADD DATA AUGMENTATION LATER!!!
# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


#CNN Model Architetcure:::
    #function that contains architecture
def building_model():
    #model = sequential
    model = Sequential()

    
    #crop any unecessary parts of images 
    # normalise 

    #5x5 layers - depth = (24, 36, 48)
    model.add(Conv2D(24, (5, 5), activation='elu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(36, (5, 5), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(48, (5, 5), activation='elu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))


    #3x3 layers - depth = (64, 64)

    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='elu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))


    #for flattening + connecting layers:
    #flatter - dense - dropout - dense - dense - dense
    #use elu activation instead of relu activation
    model.add(Flatten())
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(32, activation='elu'))
    model.add(Dense(10, activation='softmax'))


    #return the model for training 
    return model


#making model globally accessible
model = building_model()


#Training the Model:::
#functions to train the model
#use ADAM optimizer for training

def training_model(train_data, val_data, epochs=10, batch_size=32):
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    updated_model = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size
    )

    return updated_model




# Function to plot training & validation accuracy and loss
def plot_training_history(updated_model):
    # Extract values for accuracy and loss
    acc = updated_model.updated_model['accuracy']
    val_acc = updated_model.updated_model['val_accuracy']
    loss =  updated_model.updated_model['loss']
    val_loss = updated_model.updated_model['val_loss']

    epochs = range(1, len(acc) + 1)

    # Plot training and validation accuracy
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


train_data = None  # Replace with actual training data
val_data = None    # Replace with actual validation data

# Train and evaluate the model
if train_data and val_data:
    history = training_model(model, train_data, val_data, epochs=10, batch_size=32)
    plot_training_history(history)



#Hyperparameter tuning and experimentation:::
#--- later 


#save model for evaluation:
#model.save("model.h5")
model.save("model.h5")

#Evalute and Iterate:::
def evaluate_model(model, test_data):
    loss, accuracy = model.evaluate(test_data)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")


