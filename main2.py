import tensorflow as tf
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


base_dir = '/eye-dataset/'
train_val_dir = os.path.join(base_dir)

train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.4)
    
validation_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.4)

train_generator = train_datagen.flow_from_directory(
    train_val_dir,
    target_size=(150, 150),
    class_mode='categorical',
    subset='training')

validation_generator = validation_datagen.flow_from_directory(
    train_val_dir,
    target_size=(150, 150),
    class_mode='categorical',
    subset='validation')