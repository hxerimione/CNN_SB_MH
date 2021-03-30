import os
from google.colab import files
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten,Dropout,BatchNormalization
import tensorflow as tf
import keras


rootPath = '/content'



trainGenerator = ImageDataGenerator(validation_split=0.2)

trainGen=trainGenerator.flow_from_directory(os.path.join(rootPath,'datadata'),
                                            target_size=(200,200),
                                            subset='training')

validationGen = trainGenerator.flow_from_directory(os.path.join(rootPath, 'datadata'),
                                                   target_size=(200, 200),
                                                   subset='validation')

model = Sequential([
        Input(shape=(200,200,3)),

        Conv2D(32,kernel_size=3,activation='relu'),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),

        Conv2D(64,kernel_size=3,activation='relu'),
        MaxPooling2D(pool_size=2),

        Conv2D(128,kernel_size=3,activation='relu'),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),
        Dropout(0.3),
        
        Flatten(),
        Dense(256,activation='relu'),
        
        Dense(128,activation='relu'),
        Dropout(0.5),
        Dense(2,activation='softmax')
        ])
model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])


epochs=8
batch_size=32
history = model.fit(
    trainGen, 
    epochs=epochs,
    steps_per_epoch=trainGen.samples/batch_size,
    validation_data=validationGen,
    validation_steps=validationGen.samples/batch_size,
)
print(history.history)
print("train loss=", history.history['loss'][-1])
print("validation loss=", history.history['val_loss'][-1]) 

def show_graph(history_dict):
    accuracy = history_dict['acc']
    val_accuracy = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(loss) + 1)
    
    plt.figure(figsize=(16, 1))
    
    plt.subplot(121)
    plt.subplots_adjust(top=2)
    plt.plot(epochs, accuracy, 'ro', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.title('Trainging and validation accuracy ')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy and Loss')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True, ncol=5)

    plt.subplot(122)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)

    plt.show()
show_graph(history.history)
