import urllib.request
import zipfile
import numpy as np
from IPython.display import Image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
urllib.request.urlretrieve(url, 'rps.zip')
local_zip = 'rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/')
zip_ref.close()


def solution_model():

    TRAINING_DIR = "tmp/rps/"

    training_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=5,
    width_shift_range=0.01,
    height_shift_range=0.01,
    shear_range=0.01,
    zoom_range=0.01,
    horizontal_flip=True,
    fill_mode='nearest', 
    validation_split=0.2
    )
    training_generator = training_datagen.flow_from_directory(TRAINING_DIR, 
                                                          batch_size=128, 
                                                          target_size=(150, 150), 
                                                          class_mode='categorical', 
                                                          subset='training',
                                                         )
    validation_generator = training_datagen.flow_from_directory(TRAINING_DIR, 
                                                          batch_size=128, 
                                                          target_size=(150, 150), 
                                                          class_mode='categorical',
                                                          subset='validation', 
                                                         )

    c_path = "cp.ckpt"
    checkpoint = ModelCheckpoint(filepath=c_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss')
    model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2), 
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2), 
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2), 
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2), 
    Flatten(), 
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax'),
])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    history = model.fit(training_generator, validation_data=(validation_generator),
                        epochs=25, callbacks=[checkpoint])
    model.load_weights(c_path)
    model.evaluate(validation_generator)
    return model

if __name__ == '__main__':
    model = solution_model()
    model.save('TF3-rps.h5')
